#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset and Normalizer for PINO-FNO rupture training.

Each .npz sample contains:
    params_raw   : (N_params,)     raw physical parameter values
    param_bounds : JSON str        bounds used for normalization
    v_l, s_l     : (nx_out, nt_out) left domain velocity / stress
    v_r, s_r     : (nx_out, nt_out) right domain velocity / stress
    x_l, x_r     : (nx_out,)       spatial coordinates (km)
    time         : (nt_out,)        time coordinates (s)

RuptureDataset returns:
    params  : Tensor (N_params,)           — normalized parameters [0, 1]
    fields  : Tensor (4, NX_OUT, NT_OUT)   — (v_l, s_l, v_r, s_r) normalized
    coords  : Tensor (2, NX_OUT, NT_OUT)   — (x, t) grid broadcast to full domain

Normalizer:
    fit(data)                — compute channel-wise mean/std from list of arrays
    transform(x)             — (x - mean) / std
    inverse_transform(x)     — x * std + mean
    state_dict / load_state_dict — for saving/loading with checkpoints
"""

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Channel-wise zero-mean unit-variance normalizer (works on numpy or torch)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray):
        """
        Fit from a 2D array of shape (N_samples, N_channels) or a list of 1-D arrays.
        For spatial/temporal fields call fit on flattened data.
        """
        if isinstance(data, list):
            data = np.stack(data, axis=0)
        self.mean = np.mean(data, axis=0, keepdims=True).astype(np.float32)
        self.std  = np.std(data,  axis=0, keepdims=True).astype(np.float32)
        return self

    def transform(self, x):
        """Normalize x. Accepts numpy array or torch Tensor."""
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std  = torch.as_tensor(self.std,  dtype=x.dtype, device=x.device)
            return (x - mean) / (std + self.eps)
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std  = torch.as_tensor(self.std,  dtype=x.dtype, device=x.device)
            return x * (std + self.eps) + mean
        return x * (self.std + self.eps) + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'eps': self.eps}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.std  = d['std']
        self.eps  = d['eps']


class RuptureDataset(Dataset):
    """
    Loads pre-generated .npz samples for (params → space-time field) prediction.

    Returns a 3-tuple:
        params  : Tensor (N_params,)
        fields  : Tensor (4, NX_OUT, NT_OUT)   — (v_l, s_l, v_r, s_r)
        coords  : Tensor (2, NX_OUT, NT_OUT)   — normalised (x, t) in [0,1]

    Parameters
    ----------
    data_dir : str
        Directory containing sample_00000.npz, sample_00001.npz, …
    param_normalizer : Normalizer, optional
        If provided, params are transformed; otherwise raw values are returned.
    field_normalizer : dict[str, Normalizer], optional
        Keys: 'v_l', 's_l', 'v_r', 's_r'; if provided, each field is transformed.
    """

    FIELD_NAMES = ['v_l', 's_l', 'v_r', 's_r']

    def __init__(
        self,
        data_dir: str,
        param_normalizer: Normalizer | None = None,
        field_normalizer: dict | None = None,
    ):
        self.data_dir = data_dir
        self.param_normalizer = param_normalizer
        self.field_normalizer = field_normalizer

        # Discover all sample files
        self.files = sorted(
            f for f in os.listdir(data_dir) if f.startswith('sample_') and f.endswith('.npz')
        )
        if not self.files:
            raise FileNotFoundError(f"No sample*.npz files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        d = np.load(path, allow_pickle=True)

        # -- Parameters --
        params = d['params_raw'].astype(np.float32)           # (N_params,)
        if self.param_normalizer is not None:
            params = self.param_normalizer.transform(params)
        params_t = torch.from_numpy(params)                   # (N_params,)

        # -- Fields: stack into (4, NX, NT) --
        fields = np.stack(
            [d[k].astype(np.float32) for k in self.FIELD_NAMES], axis=0
        )  # (4, NX_OUT, NT_OUT)
        if self.field_normalizer is not None:
            for ci, k in enumerate(self.FIELD_NAMES):
                if k in self.field_normalizer:
                    fields[ci] = self.field_normalizer[k].transform(fields[ci])
        fields_t = torch.from_numpy(fields)                   # (4, NX, NT)

        # -- Coordinate grid (normalised to [0,1]) --
        nx_out, nt_out = fields_t.shape[1], fields_t.shape[2]
        x_grid = torch.linspace(0.0, 1.0, nx_out)            # (NX,)
        t_grid = torch.linspace(0.0, 1.0, nt_out)            # (NT,)
        # Broadcast to 2D
        x_2d = x_grid.unsqueeze(1).expand(nx_out, nt_out)    # (NX, NT)
        t_2d = t_grid.unsqueeze(0).expand(nx_out, nt_out)    # (NX, NT)
        coords_t = torch.stack([x_2d, t_2d], dim=0)          # (2, NX, NT)

        return params_t, fields_t, coords_t


# ---------------------------------------------------------------------------
# Helper: fit normalizers from a training dataset directory
# ---------------------------------------------------------------------------

def fit_normalizers(
    train_dir: str,
    max_samples: int = 500,
) -> tuple[Normalizer, dict]:
    """
    Scan up to max_samples .npz files from train_dir and fit normalizers.

    Returns
    -------
    param_norm  : Normalizer   shape (N_params,)
    field_norms : dict[str -> Normalizer]  one per field, each fit on all pixel values
    """
    files = sorted(
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.startswith('sample_') and f.endswith('.npz')
    )[:max_samples]

    if not files:
        raise FileNotFoundError(f"No sample files in {train_dir}")

    all_params = []
    all_fields = {k: [] for k in RuptureDataset.FIELD_NAMES}

    for fpath in files:
        d = np.load(fpath, allow_pickle=True)
        all_params.append(d['params_raw'].astype(np.float32))
        for k in RuptureDataset.FIELD_NAMES:
            all_fields[k].append(d[k].astype(np.float32).ravel())

    # Param normalizer — fit jointly over (N_samples, N_params)
    param_norm = Normalizer().fit(np.stack(all_params, axis=0))

    # Field normalizers — each fit on all pixel values (scalar mean/std)
    field_norms = {}
    for k in RuptureDataset.FIELD_NAMES:
        vals = np.concatenate(all_fields[k]).reshape(-1, 1)
        field_norms[k] = Normalizer().fit(vals)

    return param_norm, field_norms


def save_normalizers(param_norm: Normalizer, field_norms: dict, path: str):
    """Save normalizers to a pickle file (stored alongside checkpoints)."""
    with open(path, 'wb') as f:
        pickle.dump({'param': param_norm.state_dict(),
                     'fields': {k: v.state_dict() for k, v in field_norms.items()}}, f)


def load_normalizers(path: str) -> tuple[Normalizer, dict]:
    """Load normalizers previously saved with save_normalizers."""
    with open(path, 'rb') as f:
        d = pickle.load(f)
    param_norm = Normalizer()
    param_norm.load_state_dict(d['param'])
    field_norms = {}
    for k, sd in d['fields'].items():
        fn = Normalizer()
        fn.load_state_dict(sd)
        field_norms[k] = fn
    return param_norm, field_norms
