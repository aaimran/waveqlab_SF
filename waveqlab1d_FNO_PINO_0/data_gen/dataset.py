"""
dataset.py — PyTorch Dataset for waveqlab1d PINO-FNO (Plans A and B)
=====================================================================

Each .npz sample saved by generate_dataset.py is loaded here and converted
into the (input_tensor, target_tensor, meta_dict) triplet required by the
FNO models.

Supports:
  - Plan A (UnifiedFNO2d): returns (fric_idx, bc_idx) as extra integer labels
  - Plan B (SeparateFNO2d): no discrete labels needed
  - Multi-resolution: select 'lr' or 'hr' suffix at construction time
  - Channel normalization: per-channel zero-mean / unit-variance

Channel layout of input_tensor (B, C_in, NX, NT):
  Spatial (6):    cs, rho, mu, Qs_inv, d_l, d_r          broadcast over t
  Scalars Plan A (14): Tau_0, sigma_n, alp_s, alp_d, D_c,
                        f0, a, b, V0, L0, psi_init, c, weight_exp, r0_l
  Scalars Plan B SW (7): Tau_0, sigma_n, alp_s, alp_d, D_c, c, weight_exp
  Scalars Plan B RS (8): Tau_0, sigma_n, f0, a, b, V0, L0, psi_init, c, weight_exp
  Coords (2):     x_norm, t_norm
"""

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Index maps ──────────────────────────────────────────────────────────────

FRIC_IDX = {'SW': 0, 'RS': 1}
BC_IDX   = {'free': 0, 'absorbing': 1, 'pml': 2}


# ─── Normalizer ──────────────────────────────────────────────────────────────

class Normalizer:
    """Channel-wise zero-mean unit-variance normalizer."""

    def __init__(self, eps: float = 1e-8):
        self.eps  = eps
        self.mean = None
        self.std  = None

    def fit(self, data: np.ndarray):
        """data: (N, C) or (N, C, H, W) — collapses all dims except C."""
        if data.ndim == 2:
            self.mean = data.mean(axis=0, keepdims=True).astype(np.float32)
            self.std  = data.std(axis=0,  keepdims=True).astype(np.float32)
        else:
            flat = data.reshape(data.shape[0], data.shape[1], -1)
            self.mean = flat.mean(axis=(0, 2), keepdims=False).astype(np.float32)
            self.std  = flat.std(axis=(0, 2),  keepdims=False).astype(np.float32)
        return self

    def transform(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            s = torch.as_tensor(self.std,  dtype=x.dtype, device=x.device)
            while m.ndim < x.ndim:
                m = m.unsqueeze(-1)
                s = s.unsqueeze(-1)
            return (x - m) / (s + self.eps)
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            s = torch.as_tensor(self.std,  dtype=x.dtype, device=x.device)
            while m.ndim < x.ndim:
                m = m.unsqueeze(-1)
                s = s.unsqueeze(-1)
            return x * (s + self.eps) + m
        return x * (self.std + self.eps) + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'eps': self.eps}

    def load_state_dict(self, d):
        self.mean, self.std, self.eps = d['mean'], d['std'], d['eps']


def save_normalizers(path: str, norms: dict):
    with open(path, 'wb') as f:
        pickle.dump({k: v.state_dict() for k, v in norms.items()}, f)


def load_normalizers(path: str, keys: list) -> dict:
    with open(path, 'rb') as f:
        sd = pickle.load(f)
    norms = {}
    for k in keys:
        n = Normalizer()
        n.load_state_dict(sd[k])
        norms[k] = n
    return norms


def fit_normalizers(data_dir: str, plan: str = 'A',
                    fric_law: str = 'SW', max_samples: int = 500) -> dict:
    """
    Fit normalizers from training data.
    Returns {'input': Normalizer, 'output': Normalizer}.
    """
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith('sample_') and f.endswith('.npz')
    )[:max_samples]

    all_inputs  = []
    all_outputs = []

    for fname in files:
        d = np.load(os.path.join(data_dir, fname), allow_pickle=True)
        inp  = _build_input_channels(d, plan, fric_law, res='lr',
                                     return_numpy=True)      # (C_in, H, W)
        out  = _build_output_channels(d, res='lr')           # (4, H, W)
        all_inputs.append(inp)
        all_outputs.append(out)

    inp_stack = np.stack(all_inputs, axis=0)   # (N, C_in, H, W)
    out_stack = np.stack(all_outputs, axis=0)   # (N, 4, H, W)

    in_norm  = Normalizer().fit(inp_stack)
    out_norm = Normalizer().fit(out_stack)
    return {'input': in_norm, 'output': out_norm}


# ─── Channel builders ─────────────────────────────────────────────────────────

def _build_output_channels(d: dict, res: str = 'lr') -> np.ndarray:
    """Returns (4, NX, NT) float32 array: v_l, s_l, v_r, s_r."""
    return np.stack([
        d[f'v_l_{res}'], d[f's_l_{res}'],
        d[f'v_r_{res}'], d[f's_r_{res}'],
    ], axis=0).astype(np.float32)


def _build_input_channels(
    d          : dict,
    plan       : str  = 'A',
    fric_law   : str  = 'SW',
    res        : str  = 'lr',
    return_numpy: bool = False,
) -> np.ndarray:
    """
    Assemble input channel tensor (C_in, NX, NT) from a loaded .npz sample dict.
    All arrays cast to float32.
    """
    # Determine output grid size from one field
    suffix  = f'v_l_{res}'
    nx, nt = d[suffix].shape

    def bcast_x(v1d: np.ndarray) -> np.ndarray:
        """(NX_SIM,) → (NX, NT) via nearest-neighbor resize then broadcast."""
        if len(v1d) != nx:
            v1d = _resize1d(v1d, nx)
        return np.tile(v1d[:, None], (1, nt)).astype(np.float32)

    def bcast_scalar(val: float) -> np.ndarray:
        return np.full((nx, nt), val, dtype=np.float32)

    def coord_x() -> np.ndarray:
        return np.tile(
            np.linspace(0, 1, nx, dtype=np.float32)[:, None], (1, nt))

    def coord_t() -> np.ndarray:
        return np.tile(
            np.linspace(0, 1, nt, dtype=np.float32)[None, :], (nx, 1))

    # Load parameters JSON
    params = json.loads(str(d['params']))

    # ── Spatial profiles (6) ──
    spatial = [
        bcast_x(d['cs_arr']),
        bcast_x(d['rho_arr']),
        bcast_x(d['mu_arr']),
        bcast_x(d['Qs_inv']),
        bcast_x(d['d_l']),
        bcast_x(d['d_r']),
    ]

    # ── Scalar channels ──
    fl = params.get('fric_law', fric_law).upper()

    if plan == 'A':
        # 14 scalars — unused ones are zero
        is_rs = 1.0 if fl == 'RS' else 0.0
        r0_l  = float(params.get('r0_l', 1))
        # encode PML as r0_l=2
        if params.get('pml', False):
            r0_l = 2.0
        scalars = [
            bcast_scalar(params.get('Tau_0',    81.6)),
            bcast_scalar(params.get('sigma_n',  120.0)),
            bcast_scalar(params.get('alp_s',    0.0) if fl == 'SW' else 0.0),
            bcast_scalar(params.get('alp_d',    0.0) if fl == 'SW' else 0.0),
            bcast_scalar(params.get('D_c',      0.0) if fl == 'SW' else 0.0),
            bcast_scalar(params.get('f0',       0.0) if fl == 'RS' else 0.0),
            bcast_scalar(params.get('a',        0.0) if fl == 'RS' else 0.0),
            bcast_scalar(params.get('b',        0.0) if fl == 'RS' else 0.0),
            bcast_scalar(params.get('V0',       0.0) if fl == 'RS' else 0.0),
            bcast_scalar(params.get('L0',       0.0) if fl == 'RS' else 0.0),
            bcast_scalar(params.get('psi_init', 0.0) if fl == 'RS' else 0.0),
            bcast_scalar(params.get('c',        0.0)),
            bcast_scalar(params.get('weight_exp', 0.0)),
            bcast_scalar(r0_l),
        ]
    else:
        # Plan B: law-specific smaller set
        scalars = [
            bcast_scalar(params.get('Tau_0',   81.6)),
            bcast_scalar(params.get('sigma_n', 120.0)),
        ]
        if fl == 'SW':
            scalars += [
                bcast_scalar(params.get('alp_s',  0.677)),
                bcast_scalar(params.get('alp_d',  0.525)),
                bcast_scalar(params.get('D_c',    0.4)),
            ]
        else:
            scalars += [
                bcast_scalar(params.get('f0',       0.6)),
                bcast_scalar(params.get('a',        0.008)),
                bcast_scalar(params.get('b',        0.012)),
                bcast_scalar(params.get('V0',       1e-6)),
                bcast_scalar(params.get('L0',       0.02)),
                bcast_scalar(params.get('psi_init', 0.4367)),
            ]
        scalars += [
            bcast_scalar(params.get('c',          0.0)),
            bcast_scalar(params.get('weight_exp', 0.0)),
        ]
        # PML: add pml_alpha channel
        if params.get('pml', False):
            scalars.append(bcast_scalar(params.get('pml_alpha', 10.0)))

    coords = [coord_x(), coord_t()]
    channels = spatial + scalars + coords
    result   = np.stack(channels, axis=0).astype(np.float32)   # (C_in, NX, NT)
    return result


def _resize1d(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Nearest-neighbour resize of 1D array to n_out points."""
    n_in  = len(arr)
    idx   = np.round(np.linspace(0, n_in - 1, n_out)).astype(int)
    return arr[idx].astype(np.float32)


# ─── Dataset class ────────────────────────────────────────────────────────────

class RuptureDataset(Dataset):
    """
    PyTorch Dataset for a directory of .npz sample files.

    Parameters
    ----------
    data_dir    : str    path to train/ val/ or test/ directory
    plan        : str    'A' or 'B' — controls input channel layout
    fric_law    : str    friction law for Plan B ('SW' or 'RS')
    res         : str    'lr' or 'hr' — which resolution to load
    in_norm     : Normalizer  optional input normalizer
    out_norm    : Normalizer  optional output normalizer
    return_meta : bool   if True, __getitem__ returns (inp, out, meta_dict)
    """

    def __init__(
        self,
        data_dir    : str,
        plan        : str  = 'A',
        fric_law    : str  = 'SW',
        res         : str  = 'lr',
        in_norm     : Normalizer | None = None,
        out_norm    : Normalizer | None = None,
        return_meta : bool = False,
    ):
        self.data_dir    = data_dir
        self.plan        = plan
        self.fric_law    = fric_law.upper()
        self.res         = res
        self.in_norm     = in_norm
        self.out_norm    = out_norm
        self.return_meta = return_meta

        self.files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith('sample_') and f.endswith('.npz')
        )
        if not self.files:
            raise FileNotFoundError(f'No sample*.npz in {data_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.data_dir, self.files[idx])
        d    = np.load(path, allow_pickle=True)
        params = json.loads(str(d['params']))

        # ── Input tensor
        inp_np  = _build_input_channels(d, plan=self.plan,
                                        fric_law=self.fric_law, res=self.res)
        inp     = torch.from_numpy(inp_np)   # (C_in, NX, NT)
        if self.in_norm is not None:
            inp = self.in_norm.transform(inp)

        # ── Output tensor
        out_np  = _build_output_channels(d, res=self.res)
        out     = torch.from_numpy(out_np)   # (4, NX, NT)
        if self.out_norm is not None:
            out = self.out_norm.transform(out)

        # ── Discrete labels (Plan A)
        fl = params.get('fric_law', self.fric_law).upper()
        bc = params.get('bc_mode', 'absorbing').lower()
        fric_idx = torch.tensor(FRIC_IDX.get(fl, 0), dtype=torch.long)
        bc_idx   = torch.tensor(BC_IDX.get(bc, 1),   dtype=torch.long)

        # ── Collateral tensors for physics loss
        nx = inp_np.shape[1]
        nt = inp_np.shape[2]
        meta = dict(
            rho         = torch.from_numpy(d['rho_arr'].astype(np.float32)),
            mu          = torch.from_numpy(d['mu_arr'].astype(np.float32)),
            cs          = torch.tensor([params.get('cs', 3.464)],  dtype=torch.float32),
            Tau_0       = torch.tensor([params.get('Tau_0', 81.6)], dtype=torch.float32),
            sigma_n     = torch.tensor([params.get('sigma_n', 120.0)], dtype=torch.float32),
            slip        = torch.from_numpy(
                            _resize1d(d['sliprate'], nt).astype(np.float32)).unsqueeze(0),
            sliprate    = torch.from_numpy(
                            _resize1d(d['sliprate'], nt).astype(np.float32)).unsqueeze(0),
            traction    = torch.from_numpy(
                            _resize1d(d['traction'], nt).astype(np.float32)).unsqueeze(0),
            fric_idx    = fric_idx,
            bc_idx      = bc_idx,
            fric_law    = fl,
            bc_mode     = bc,
            dx          = float(d['dx']),
            dt          = float(d['dt']),
        )
        # SW-specific
        if fl == 'SW':
            meta['alp_s'] = torch.tensor([params.get('alp_s', 0.677)], dtype=torch.float32)
            meta['alp_d'] = torch.tensor([params.get('alp_d', 0.525)], dtype=torch.float32)
            meta['D_c']   = torch.tensor([params.get('D_c',   0.4)],   dtype=torch.float32)
        # RS-specific
        else:
            meta['f0']   = torch.tensor([params.get('f0',   0.6)],    dtype=torch.float32)
            meta['a']    = torch.tensor([params.get('a',    0.008)],   dtype=torch.float32)
            meta['b']    = torch.tensor([params.get('b',    0.012)],   dtype=torch.float32)
            meta['V0']   = torch.tensor([params.get('V0',   1e-6)],    dtype=torch.float32)
            meta['L0']   = torch.tensor([params.get('L0',   0.02)],    dtype=torch.float32)

        if self.return_meta:
            return inp, out, meta
        return inp, out, fric_idx, bc_idx


def collate_with_meta(batch):
    """
    Custom collate_fn for DataLoader when return_meta=True.
    Stacks inp and out tensors; merges meta dicts with batch dimension.
    """
    inps, outs, metas = zip(*[(b[0], b[1], b[2]) for b in batch])
    inp_batch = torch.stack(inps, dim=0)
    out_batch = torch.stack(outs, dim=0)

    # Merge metas: stack tensors, keep scalars as lists
    merged = {}
    for key in metas[0].keys():
        vals = [m[key] for m in metas]
        if isinstance(vals[0], torch.Tensor):
            try:
                merged[key] = torch.stack(vals, dim=0)
            except Exception:
                merged[key] = vals
        else:
            merged[key] = vals

    return inp_batch, out_batch, merged
