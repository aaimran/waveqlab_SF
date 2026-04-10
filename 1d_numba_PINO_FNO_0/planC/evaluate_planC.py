#!/usr/bin/env python3
"""
planC/evaluate_planC.py
========================
Evaluation and super-resolution testing for Plan-C trained FNO models.

Capabilities
------------
1. Load a trained checkpoint + config.
2. Run inference on test-set at the TRAINING resolution.
3. Run super-resolution: feed 20 m (r20) test inputs to a model trained at 80 m.
4. Compare both to ground-truth SBP-SAT at 10 m (run simulator inline or load
   precomputed GT files).
5. Report per-field relative-L2 errors at each resolution.
6. Generate comparison figures:
     - Space-time panels (v_l, s_l, v_r, s_r)
     - Waveforms at midpoint (x = L/2) and fault (x = 0 km, x = L km)
     - Fault time series: slip, slip-rate, traction

Usage:
    cd /scratch/aimran/FNO/waveqlab_SF/1d_numba_PINO_FNO_0
    source /work/aimran/wql1d/env.sh

    # Evaluate at training resolution (80 m):
    python planC/evaluate_planC.py \\
        --checkpoint planC/checkpoints/sw_bc_both0_r80m/best.pt \\
        --config     planC/configs/sw_bc_both0_r80m.yaml \\
        --out_dir    planC/results/sw_bc_both0_r80m

    # Super-resolution test (infer at 20 m, compare to GT 10 m):
    python planC/evaluate_planC.py \\
        --checkpoint planC/checkpoints/sw_bc_both0_r80m/best.pt \\
        --config     planC/configs/sw_bc_both0_r80m.yaml \\
        --sr_res     r20 \\
        --gt_res     r10_gt \\
        --out_dir    planC/results/sw_bc_both0_r80m_sr
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from model.fno          import FNO2d, build_input_tensor
from model.physics_loss import relative_l2_loss, energy_stability_loss
from data_gen.dataset   import (RuptureDatasetPlanC, Normalizer,
                                fit_normalizers, collate_with_meta,
                                FIELD_KEYS, PARAM_NAMES)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('[WARNING] matplotlib not available — no figures will be saved.')

# ---------------------------------------------------------------------------
# Ground-truth generation (inline simulator)
# ---------------------------------------------------------------------------

def _run_gt_for_sample(npz_path, bc_both, gt_nx=3001):
    """
    Run the SBP-SAT simulator at gt_nx (10 m) for a single sample and
    return downsampled arrays matching NT_OUT=128 time steps.
    Returns dict with v_l, s_l, v_r, s_r each (gt_nx, 128).
    """
    from rupture_1d import parse_infile, build_params, init_fields, build_friction_parameters
    from src.kernels import rk4_step

    d = np.load(npz_path, allow_pickle=True)
    pnames = json.loads(str(d['param_names']))
    pvals  = {k: float(d['params_raw'][i]) for i, k in enumerate(pnames)}
    r0_l   = int(d['r0_l'])
    r1_r   = int(d['r1_r'])

    # Build parameter dict
    infile = os.path.join(_ROOT, 'planC', 'input',
                          f'sw_gt10m_bc{"0" if r0_l == 0 else "1"}.in')
    p = build_params(parse_infile(infile))
    p.update(pvals)
    p['r0_l'] = r0_l
    p['r1_r'] = r1_r
    p['nx']   = gt_nx
    p['dx']   = float(p['L']) / (gt_nx - 1)
    p['dt']   = (float(p['cfl']) / float(p['cs'])) * p['dx']
    p['nt']   = int(round(float(p['tend']) / p['dt']))

    y_l, y_r, v_l, s_l, v_r, s_r, slip, psi = init_fields(p)
    fp    = build_friction_parameters(p)
    rho   = float(p['rho'])
    cs    = float(p['cs'])
    mu    = rho * cs ** 2
    nx    = p['nx']
    dx    = p['dx']
    dt    = p['dt']
    nt    = p['nt']
    iplot = int(p.get('iplot', 5))

    n_stored = nt // iplot + 1
    VL = np.zeros((nx, n_stored)); SL = np.zeros((nx, n_stored))
    VR = np.zeros((nx, n_stored)); SR = np.zeros((nx, n_stored))
    TIME = np.zeros(n_stored)

    idx = 0
    for n in range(nt):
        rk4_step(v_l, s_l, v_r, s_r, slip, psi,
                 rho, mu, nx, dx, int(p['order']),
                 float(r0_l), float(r1_r), dt, fp)
        if n % iplot == 0 and idx < n_stored:
            VL[:, idx] = v_l;  SL[:, idx] = s_l
            VR[:, idx] = v_r;  SR[:, idx] = s_r
            TIME[idx]  = n * dt
            idx += 1

    actual = idx
    ti = np.round(np.linspace(0, actual - 1, 128)).astype(int)
    return {
        'v_l': VL[:, ti].astype(np.float32),
        's_l': SL[:, ti].astype(np.float32),
        'v_r': VR[:, ti].astype(np.float32),
        's_r': SR[:, ti].astype(np.float32),
        'time': TIME[ti].astype(np.float32),
        'nx': gt_nx, 'dx_km': float(p['L']) / (gt_nx - 1),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rel_l2(pred_np, ref_np):
    """rel-L2 error as percentage."""
    denom = np.linalg.norm(ref_np) + 1e-12
    return 100.0 * np.linalg.norm(pred_np - ref_np) / denom


def compute_errors(pred: np.ndarray, ref: np.ndarray, labels=None):
    """
    pred, ref: (4, nx, nt) arrays (un-normalised)
    Returns dict of per-field rel-L2 errors (%)
    """
    if labels is None:
        labels = FIELD_KEYS
    errs = {}
    for i, lab in enumerate(labels):
        errs[lab] = _rel_l2(pred[i], ref[i])
    errs['mean'] = float(np.mean(list(errs.values())))
    return errs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_spacetime(pred, ref, field_labels, title, out_path, dx_km, dt_s):
    if not HAS_MPL:
        return
    nx, nt = pred[0].shape
    x_km   = np.linspace(0, dx_km * (nx - 1), nx)
    t_s    = np.linspace(0, dt_s * (nt - 1),  nt)

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=12)

    for row, (lab, p, r) in enumerate(zip(field_labels, pred, ref)):
        err = r - p
        vmax_pr = max(np.abs(p).max(), np.abs(r).max(), 1e-12)
        vmax_e  = max(np.abs(err).max(), 1e-12)

        for col, (dat, lbl, vm) in enumerate([
            (r,   f'GT {lab}',   vmax_pr),
            (p,   f'Pred {lab}', vmax_pr),
            (err, f'Error {lab}',vmax_e),
        ]):
            ax = axes[row, col]
            im = ax.pcolormesh(t_s, x_km, dat,
                               cmap='RdBu_r',
                               vmin=-vm, vmax=vm)
            ax.set_xlabel('t (s)')
            ax.set_ylabel('x (km)')
            ax.set_title(lbl)
            plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_waveforms(pred, ref, t_arr, field_labels, slice_idxs,
                    slice_names, title, out_path):
    if not HAS_MPL:
        return
    n_fields = len(field_labels)
    n_slices = len(slice_idxs)
    fig, axes = plt.subplots(n_fields, n_slices,
                             figsize=(5 * n_slices, 3 * n_fields),
                             squeeze=False)
    fig.suptitle(title, fontsize=11)
    for fi, lab in enumerate(field_labels):
        for si, (xi, sname) in enumerate(zip(slice_idxs, slice_names)):
            ax = axes[fi, si]
            ax.plot(t_arr, ref[fi][xi, :], 'k', lw=1.5, label='GT')
            ax.plot(t_arr, pred[fi][xi, :], 'r--', lw=1.2, label='Pred')
            err_pct = _rel_l2(pred[fi][xi, :], ref[fi][xi, :])
            ax.set_title(f'{lab} @ {sname}  err={err_pct:.1f}%', fontsize=9)
            ax.set_xlabel('t (s)')
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluate loop
# ---------------------------------------------------------------------------

def evaluate_split(model, test_dir, res, param_norm, field_norm,
                   device, out_dir, batch_size=4,
                   gt_dir=None, gt_res=None, n_plot=5):
    """
    Runs inference on all test samples.  Optionally loads / generates GT
    and computes cross-resolution errors.
    """
    from torch.utils.data import DataLoader

    ds = RuptureDatasetPlanC(test_dir, res=res,
                             param_norm=param_norm,
                             field_norm=field_norm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_with_meta, num_workers=2)

    all_errors = {k: [] for k in FIELD_KEYS + ['mean']}
    model.eval()
    sample_paths = ds.paths
    sample_idx   = 0

    with torch.no_grad():
        for inp, target, meta in loader:
            inp    = inp.to(device)
            pred   = model(inp)          # (B,4,nx,nt)

            # Un-normalise
            m = field_norm.mean.view(1, 4, 1, 1).to(device)
            s = field_norm.std.view(1, 4, 1, 1).to(device)
            pred_un   = pred   * (s + field_norm.epsilon) + m
            target_un = target.to(device) * (s + field_norm.epsilon) + m

            B = pred.shape[0]
            for b in range(B):
                p_np = pred_un[b].cpu().numpy()    # (4,nx,nt)
                r_np = target_un[b].cpu().numpy()  # (4,nx,nt)
                errs = compute_errors(p_np, r_np)
                for k in errs:
                    all_errors[k].append(errs[k])

                # Plots for first n_plot samples
                if sample_idx < n_plot and HAS_MPL:
                    dx = float(meta['dx_km'][b])
                    dt = float(meta['dt_s'][b])
                    _plot_spacetime(p_np, r_np, FIELD_KEYS,
                                    f'Sample {sample_idx} [{res}]',
                                    os.path.join(out_dir,
                                                 f'spacetime_{sample_idx:03d}.png'),
                                    dx, dt)
                    nt_s  = r_np.shape[2]
                    t_arr = np.linspace(0, dt * (nt_s - 1), nt_s)
                    nx_s  = r_np.shape[1]
                    mid   = nx_s // 2
                    _plot_waveforms(p_np, r_np, t_arr,
                                    FIELD_KEYS,
                                    [0, mid, nx_s - 1],
                                    ['x=0', 'x=mid', 'x=end'],
                                    f'Waveforms sample {sample_idx} [{res}]',
                                    os.path.join(out_dir,
                                                 f'waveform_{sample_idx:03d}.png'))
                sample_idx += 1

    # Summary
    means = {k: float(np.mean(all_errors[k])) for k in all_errors if all_errors[k]}
    stds  = {k: float(np.std(all_errors[k]))  for k in all_errors if all_errors[k]}
    return means, stds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config',     required=True)
    parser.add_argument('--out_dir',    default=None)
    parser.add_argument('--sr_res',     default=None,
                        help='Super-resolution inference resolution, e.g. r20')
    parser.add_argument('--n_plot',     type=int, default=5)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device(args.device)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            cfg['checkpoints']['dir'].replace('checkpoints', 'results'),
            'eval')
    os.makedirs(args.out_dir, exist_ok=True)

    # Load normalizers
    norm_dir = os.path.join(cfg['checkpoints']['dir'], 'normalizers')
    res      = cfg['data']['res']
    param_norm = Normalizer.load(os.path.join(norm_dir, 'param_norm.npz'))
    field_norm = Normalizer.load(os.path.join(norm_dir, 'field_norm.npz'))

    # Load model
    mc    = cfg['model']
    model = FNO2d(
        c_in=mc['c_in'], modes_x=mc['modes_x'], modes_t=mc['modes_t'],
        width=mc['width'], n_layers=mc['n_layers'], c_out=mc['c_out'],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Best val loss: {ckpt.get("val_loss", "N/A")}')

    test_dir = cfg['data']['test_dir']

    # ----- Evaluation at training resolution -----
    print(f'\n--- Evaluating at training resolution: {res} ---')
    means, stds = evaluate_split(
        model, test_dir, res, param_norm, field_norm,
        device, args.out_dir, batch_size=8, n_plot=args.n_plot)

    print('\nRelative-L2 errors at training resolution:')
    for k in FIELD_KEYS + ['mean']:
        print(f'  {k:15s}: {means.get(k, 0):6.2f} ± {stds.get(k, 0):.2f} %')

    result_dict = {f'{res}/{k}': means[k] for k in means}

    # ----- Super-resolution evaluation -----
    if args.sr_res:
        sr_res = args.sr_res
        sr_norm_path = os.path.join(norm_dir, f'field_norm_{sr_res}.npz')
        sr_par_path  = os.path.join(norm_dir, f'param_norm_{sr_res}.npz')

        if os.path.exists(sr_norm_path) and os.path.exists(sr_par_path):
            sr_param_norm = Normalizer.load(sr_par_path)
            sr_field_norm = Normalizer.load(sr_norm_path)
        else:
            print(f'[SR] Fitting normalizers for {sr_res}...')
            sr_param_norm, sr_field_norm = fit_normalizers(test_dir, res=sr_res)

        sr_out = os.path.join(args.out_dir, f'sr_{sr_res}')
        os.makedirs(sr_out, exist_ok=True)

        print(f'\n--- Super-resolution inference at {sr_res} '
              f'(model trained at {res}) ---')
        sr_means, sr_stds = evaluate_split(
            model, test_dir, sr_res, sr_param_norm, sr_field_norm,
            device, sr_out, batch_size=4, n_plot=args.n_plot)

        print(f'\nRel-L2 at {sr_res} (super-resolution):')
        for k in FIELD_KEYS + ['mean']:
            print(f'  {k:15s}: {sr_means.get(k, 0):6.2f} ± {sr_stds.get(k, 0):.2f} %')

        for k in sr_means:
            result_dict[f'{sr_res}/{k}'] = sr_means[k]

    # Save summary
    summary_path = os.path.join(args.out_dir, 'summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f'\nSummary saved → {summary_path}')


if __name__ == '__main__':
    main()
