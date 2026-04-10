#!/usr/bin/env python3
"""
planC/data_gen/dataset.py
==========================
PyTorch dataset for Plan-C multi-resolution SW rupture data.

Each .npz file stored by generate_dataset.py contains fields at three
resolutions (r80/r40/r20).  This dataset:
  1. Loads files for a requested resolution.
  2. Normalises params and fields (fit on training data).
  3. Builds the 9-channel FNO input tensor via build_input_tensor().
  4. Returns (input_tensor, target_tensor, meta) tuples suitable for
     PINOLoss.forward().

Usage:
    from data_gen.dataset import RuptureDatasetPlanC, Normalizer, fit_normalizers

    # Fit normalisers on training split:
    tr_params, tr_fields = fit_normalizers(train_dir, res='r80')

    # Build datasets:
    ds_train = RuptureDatasetPlanC(train_dir, res='r80',
                                   param_norm=tr_params, field_norm=tr_fields)
    ds_val   = RuptureDatasetPlanC(val_dir,   res='r80',
                                   param_norm=tr_params, field_norm=tr_fields)
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE  = os.path.dirname(os.path.abspath(__file__))
_PLANC = os.path.dirname(_HERE)

try:
    from model.fno import build_input_tensor
except ImportError:
    sys.path.insert(0, _PLANC)
    from model.fno import build_input_tensor


# ---------------------------------------------------------------------------
# Resolution meta
# ---------------------------------------------------------------------------
RES_TO_NX = {'r80': 376, 'r40': 751, 'r20': 1501}

# Param names (must match generate_dataset.py)
PARAM_NAMES = ['Tau_0', 'alp_s', 'alp_d', 'D_c', 'sigma_n', 'cs', 'rho']

# Target output fields: v_l, s_l, v_r, s_r stacked along channel 0
FIELD_KEYS  = ['v_l', 's_l', 'v_r', 's_r']


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """Mean-std normalizer; uses torch.Tensor or ndarray inputs."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, epsilon: float = 1e-8):
        self.mean    = torch.tensor(mean, dtype=torch.float32)
        self.std     = torch.tensor(std,  dtype=torch.float32)
        self.epsilon = epsilon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(x.device)
        s = self.std.to(x.device)
        # x can be (..., C) or (B, C, ...)
        return (x - m) / (s + self.epsilon)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(x.device)
        s = self.std.to(x.device)
        return x * (s + self.epsilon) + m

    def save(self, path: str):
        np.savez(path,
                 mean=self.mean.numpy(),
                 std=self.std.numpy(),
                 epsilon=np.float64(self.epsilon))

    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        d = np.load(path)
        return cls(d['mean'], d['std'], float(d['epsilon']))


# ---------------------------------------------------------------------------
# Helpers to gather all samples from a directory
# ---------------------------------------------------------------------------

def _list_samples(data_dir: str):
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith('sample_') and f.endswith('.npz')
    )
    if not files:
        raise FileNotFoundError(f'No sample_*.npz found in {data_dir}')
    return [os.path.join(data_dir, f) for f in files]


def fit_normalizers(data_dir: str, res: str = 'r80'):
    """
    Compute per-channel mean/std for params and field tensors from all
    samples in data_dir.  Returns (param_norm, field_norm).
    """
    paths  = _list_samples(data_dir)
    nx_out = RES_TO_NX[res]

    param_list  = []
    field_list  = []   # flattened space-time for per-field stats

    for p in paths:
        d = np.load(p, allow_pickle=True)
        param_list.append(d['params_raw'].astype(np.float32))  # (7,)
        # Stack 4 fields → (4, nx, nt)
        fields = np.stack([d[f'{k}_{res}'] for k in FIELD_KEYS], axis=0)  # (4,nx,nt)
        field_list.append(fields)

    params = np.stack(param_list, axis=0)   # (N, 7)
    fields = np.stack(field_list, axis=0)   # (N, 4, nx, nt)

    param_mean = params.mean(axis=0)
    param_std  = params.std(axis=0) + 1e-8

    # Per-channel (axis 1 of fields) mean/std over N, nx, nt
    field_mean = fields.mean(axis=(0, 2, 3))   # (4,)
    field_std  = fields.std(axis=(0, 2, 3))   # (4,)
    field_std  = np.where(field_std < 1e-12, 1.0, field_std)

    return (Normalizer(param_mean, param_std),
            Normalizer(field_mean, field_std))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RuptureDatasetPlanC(Dataset):
    """
    Parameters
    ----------
    data_dir  : directory with sample_*.npz files
    res       : resolution to load — 'r80', 'r40', or 'r20'
    param_norm: Normalizer for the 7 scalar params (fit on training set)
    field_norm: Normalizer for the 4 field channels  (fit on training set)
    nt_out    : number of time steps (default: as in data → 128)

    Returns
    -------
    (inp_tensor, target, meta)
      inp_tensor : (C_in, nx, nt) float32 — 7 normalised params + 2 coords
      target     : (4, nx, nt) float32 — normalised [v_l, s_l, v_r, s_r]
      meta       : dict with raw scalars needed by PINOLoss
                      keys: Tau_0, alp_s, alp_d, D_c, sigma_n, rho,
                            cs, dx_km, dt_s, L_km, r0_l, r1_r
    """

    def __init__(self,
                 data_dir: str,
                 res: str = 'r80',
                 param_norm: Normalizer = None,
                 field_norm: Normalizer = None):
        self.paths     = _list_samples(data_dir)
        self.res       = res
        self.nx_out    = RES_TO_NX[res]
        self.param_norm = param_norm
        self.field_norm = field_norm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx], allow_pickle=True)

        # Raw params tensor (7,)
        params_raw = torch.from_numpy(d['params_raw'].astype(np.float32))   # (7,)

        # Normalise params
        if self.param_norm is not None:
            params_n = self.param_norm.encode(params_raw)
        else:
            params_n = params_raw

        # Field target (4, nx, nt)
        fields = torch.stack(
            [torch.from_numpy(d[f'{k}_{self.res}'].astype(np.float32))
             for k in FIELD_KEYS], dim=0)   # (4, nx, nt)

        if self.field_norm is not None:
            # Per-channel normalise: field_norm.mean/std are (4,)
            m = self.field_norm.mean.view(4, 1, 1)
            s = self.field_norm.std.view(4, 1, 1)
            fields_n = (fields - m) / (s + self.field_norm.epsilon)
        else:
            fields_n = fields

        # Spatial coords (normalised to [0,1])
        nx  = fields.shape[1]
        nt  = fields.shape[2]
        x_n = torch.linspace(0.0, 1.0, nx)
        t_n = torch.linspace(0.0, 1.0, nt)
        # Build (2, nx, nt) coord channels
        x_ch = x_n.view(nx, 1).expand(nx, nt)
        t_ch = t_n.view(1, nt).expand(nx, nt)
        coords = torch.stack([x_ch, t_ch], dim=0)   # (2, nx, nt)

        # Build FNO input tensor using build_input_tensor
        # params_n is (7,) → unsqueeze batch dim → (1,7) → build → (1,9,nx,nt)
        inp = build_input_tensor(
            params_n.unsqueeze(0),   # (1,7)
            coords.unsqueeze(0),     # (1,2,nx,nt)
        ).squeeze(0)                 # (9, nx, nt)

        # Metadata for PINOLoss
        pnames = json.loads(str(d['param_names']))
        pvals  = {k: float(d['params_raw'][i]) for i, k in enumerate(pnames)}
        dx_km  = float(d[f'dx_km_{self.res}'])
        dt_s   = float(d['dt_native_s'])
        meta   = dict(
            Tau_0   = torch.tensor(pvals['Tau_0'],   dtype=torch.float32),
            alp_s   = torch.tensor(pvals['alp_s'],   dtype=torch.float32),
            alp_d   = torch.tensor(pvals['alp_d'],   dtype=torch.float32),
            D_c     = torch.tensor(pvals['D_c'],     dtype=torch.float32),
            sigma_n = torch.tensor(pvals['sigma_n'], dtype=torch.float32),
            cs      = torch.tensor(pvals['cs'],      dtype=torch.float32),
            rho     = torch.tensor(pvals['rho'],     dtype=torch.float32),
            dx_km   = torch.tensor(dx_km, dtype=torch.float32),
            dt_s    = torch.tensor(dt_s,  dtype=torch.float32),
            L_km    = torch.tensor(float(d['L_km']), dtype=torch.float32),
            r0_l    = int(d['r0_l']),
            r1_r    = int(d['r1_r']),
        )

        return inp, fields_n, meta


# ---------------------------------------------------------------------------
# Collate helper for variable-length meta dicts
# ---------------------------------------------------------------------------

def collate_with_meta(batch):
    """torch DataLoader collate_fn that handles meta dicts."""
    inps, targets, metas = zip(*batch)
    inp_batch    = torch.stack(inps,    dim=0)
    target_batch = torch.stack(targets, dim=0)

    keys = metas[0].keys()
    meta_batch = {}
    for k in keys:
        if isinstance(metas[0][k], torch.Tensor):
            meta_batch[k] = torch.stack([m[k] for m in metas], dim=0)
        else:
            meta_batch[k] = [m[k] for m in metas]

    return inp_batch, target_batch, meta_batch
