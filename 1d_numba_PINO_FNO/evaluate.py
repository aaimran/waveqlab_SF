#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained FNO-2D model on the test set.

Usage:
    python3 evaluate.py --checkpoint checkpoints/best.pt
    python3 evaluate.py --checkpoint checkpoints/best.pt --config configs/fno_sw_baseline.yaml
    python3 evaluate.py --checkpoint checkpoints/best.pt --plot

Metrics:
    Relative L2 error per field (v_l, s_l, v_r, s_r)
    Peak slip-rate error
    Inference speedup vs. simulator

Outputs:
    Prints a summary table to stdout.
    If --plot: saves comparison PNGs to plots/eval_{run_id}/
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_gen.dataset import RuptureDataset, load_normalizers
from model.fno import FNO2d
from model.physics_loss import relative_l2_loss

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml is required: pip install pyyaml")


FIELD_NAMES = ['v_l', 's_l', 'v_r', 's_r']


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def per_field_rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Returns relative L2 error for each field channel.  Shape: (C,)."""
    diff = (pred - target).pow(2).sum(dim=(-2, -1))
    norm = target.pow(2).sum(dim=(-2, -1)).clamp(min=eps)
    return (diff / norm).sqrt().mean(dim=0).cpu().numpy()   # (C,)


def peak_sliprate_error(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """
    Relative error in peak slip rate:
        slip_rate ≈ |v_l[:, -1, :] - v_r[:, 0, :]|
    Returns (mean_rel_error, max_rel_error).
    """
    eps = 1e-10
    sr_pred   = (pred[:, 0, -1, :] - pred[:, 2, 0, :]).abs()    # (B, NT)
    sr_target = (target[:, 0, -1, :] - target[:, 2, 0, :]).abs()

    peak_pred   = sr_pred.max(dim=-1).values    # (B,)
    peak_target = sr_target.max(dim=-1).values

    rel_err = ((peak_pred - peak_target).abs() / (peak_target + eps)).cpu().numpy()
    return float(rel_err.mean()), float(rel_err.max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pt)')
    parser.add_argument('--config', default=None,
                        help='Config YAML (default: checkpoints/config.yaml)')
    parser.add_argument('--plot',  action='store_true', help='Save comparison plots')
    parser.add_argument('--n_plot', type=int, default=4, help='Number of samples to plot')
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(args.checkpoint)

    # Load config
    config_path = args.config or os.path.join(ckpt_dir, 'config.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load normalizers
    norm_path = os.path.join(ckpt_dir, 'normalizers.pkl')
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f'Normalizers not found at {norm_path}')
    param_norm, field_norms = load_normalizers(norm_path)

    # Test dataset
    test_dir = cfg['data']['test_dir']
    test_ds  = RuptureDataset(test_dir, param_norm, field_norms)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f'Test: {len(test_ds)} samples')

    # Build model
    md = cfg['model']
    model = FNO2d(
        n_params = md['n_params'],
        modes_x  = md['modes_x'],
        modes_t  = md['modes_t'],
        width    = md['width'],
        n_layers = md['n_layers'],
        c_out    = md['c_out'],
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt["epoch"]+1}, val_loss={ckpt["val_loss"]:.6f})')

    # ── Evaluation ─────────────────────────────────────────────────────────
    all_rel_l2     = []
    all_peak_mean  = []
    all_peak_max   = []
    total_samples  = 0

    t_infer_start = time.perf_counter()

    with torch.no_grad():
        for params, fields, coords in test_loader:
            params = params.to(device)
            fields = fields.to(device)
            coords = coords.to(device)

            pred = model(params, coords)

            # Inverse-transform predictions and targets for metric computation in physical units
            # (optional; relative errors are scale-invariant, so normalised space works too)
            rel_err  = per_field_rel_l2(pred, fields)   # (4,)
            pm, pmax = peak_sliprate_error(pred, fields)

            all_rel_l2.append(rel_err)
            all_peak_mean.append(pm)
            all_peak_max.append(pmax)
            total_samples += params.shape[0]

    t_infer = time.perf_counter() - t_infer_start
    infer_per_sample = t_infer / total_samples * 1000   # ms

    mean_rel_l2 = np.mean(all_rel_l2, axis=0)   # (4,)

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 55)
    print(f'  Test samples : {total_samples}')
    print(f'  Inference    : {t_infer * 1000:.1f} ms total  |  {infer_per_sample:.2f} ms/sample')
    print(f'  Simulator    : ~86 ms/sample  (Numba, np=1)')
    print(f'  Speedup      : ~{86.0 / max(infer_per_sample, 0.01):.0f}×')
    print('─' * 55)
    print('  Relative L2 error per field:')
    for name, err in zip(FIELD_NAMES, mean_rel_l2):
        print(f'    {name:6s}  :  {err:.4f}  ({err*100:.2f}%)')
    print('─' * 55)
    print(f'  Peak slip-rate error (mean): {np.mean(all_peak_mean):.4f}')
    print(f'  Peak slip-rate error (max) : {np.max(all_peak_max):.4f}')
    print('=' * 55)

    # ── Optional plots ──────────────────────────────────────────────────────
    if args.plot:
        _make_plots(model, test_ds, param_norm, field_norms, cfg, device, args.n_plot)


def _make_plots(model, test_ds, param_norm, field_norms, cfg, device, n_plot):
    """Save side-by-side comparison plots (prediction vs. ground truth)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('matplotlib not available; skipping plots.')
        return

    plot_dir = os.path.join('plots', 'eval')
    os.makedirs(plot_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(min(n_plot, len(test_ds))):
            params, fields, coords = test_ds[i]
            params = params.unsqueeze(0).to(device)
            fields = fields.unsqueeze(0).to(device)
            coords = coords.unsqueeze(0).to(device)

            pred = model(params, coords).squeeze(0).cpu().numpy()    # (4, NX, NT)
            gt   = fields.squeeze(0).cpu().numpy()

            fig, axes = plt.subplots(4, 2, figsize=(10, 12))
            titles = FIELD_NAMES
            for row, name in enumerate(titles):
                vmin = min(gt[row].min(), pred[row].min())
                vmax = max(gt[row].max(), pred[row].max())
                axes[row, 0].imshow(gt[row],   aspect='auto', origin='lower',
                                    vmin=vmin, vmax=vmax, interpolation='nearest')
                axes[row, 0].set_title(f'GT {name}')
                im = axes[row, 1].imshow(pred[row], aspect='auto', origin='lower',
                                          vmin=vmin, vmax=vmax, interpolation='nearest')
                axes[row, 1].set_title(f'Pred {name}')
                plt.colorbar(im, ax=axes[row, 1])

            fig.suptitle(f'Test sample {i}')
            plt.tight_layout()
            out = os.path.join(plot_dir, f'sample_{i:04d}.png')
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f'  Saved {out}')


if __name__ == '__main__':
    main()
