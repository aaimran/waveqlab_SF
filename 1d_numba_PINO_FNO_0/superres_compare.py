#!/usr/bin/env python3
"""
superres_compare.py — Super-resolution evaluation of the trained FNO.

The FNO was trained to predict fields at 64×64 (NX_TRAIN × NT_TRAIN).
Because FNO is resolution-invariant (Fourier modes << grid size), we can
run the forward pass at 128×128 without retraining.

Experiment:
  gt_64   — simulator downsampled to 64×64 (training resolution)
  gt_128  — simulator downsampled to 128×128 (double resolution = reference)
  fno_64  — FNO inference at 64×64 (training resolution)
  fno_128 — FNO inference at 128×128 (resolution generalisation, no retraining)

  fno_64_up  — fno_64 bilinear-upsampled to 128×128
  gt_64_up   — gt_64  bilinear-upsampled to 128×128 (naive interpolation baseline)

All three are compared against gt_128 at 128×128:
  1. gt_64_up  vs gt_128  → baseline error (naive bicubic rescaling of coarse sim)
  2. fno_64_up vs gt_128  → FNO error after standard upsampling
  3. fno_128   vs gt_128  → FNO resolution-generalisation error (best case)

Usage:
    python3 superres_compare.py input/rupture_1d_SW.in \\
            --checkpoint checkpoints/best.pt \\
            --config     configs/fno_sw_baseline.yaml \\
            --outdir     output/superres_SW
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml required: pip install pyyaml")

from rupture_1d import parse_infile, build_params, validate, init_fields, build_friction_parameters
from data_gen.dataset import load_normalizers
from model.fno import FNO2d
from src.kernels import rk4_step

# ── Inlined constants (avoids slow scipy import from generate_dataset.py) ───
PARAM_NAMES = ['Tau_0', 'alp_s', 'alp_d', 'D_c']
PARAM_BOUNDS = {
    'Tau_0': (78.0, 88.0),
    'alp_s': (0.62, 0.74),
    'alp_d': (0.45, 0.55),
    'D_c'  : (0.2,  0.8),
}
NX_TRAIN = 64
NT_TRAIN = 64

FIELD_NAMES  = ['v_l', 's_l', 'v_r', 's_r']
FIELD_LABELS = [r'$v_l$ (m/s)', r'$s_l$ (MPa)', r'$v_r$ (m/s)', r'$s_r$ (MPa)']


# ─────────────────────────────────────────────────────────────────────────────
# Simulator runner — parameterised by output resolution
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(p, nx_out: int, nt_out: int) -> dict:
    """Run simulator, return fields downsampled to (nx_out, nt_out)."""
    nx    = p['nx']
    nt    = p['nt']
    dx    = p['dx']
    dt    = p['dt']
    order = int(p['order'])
    r0_l  = float(p['r0_l'])
    r1_r  = float(p['r1_r'])
    rho   = float(p['rho'])
    mu    = float(p['rho']) * float(p['cs']) ** 2
    Tau_0 = float(p['Tau_0'])
    iplot = int(p['iplot'])

    y_l, y_r, v_l, s_l, v_r, s_r, slip, psi = init_fields(p)
    fp = build_friction_parameters(p)

    n_stored = nt // iplot + 1
    VL = np.zeros((nx, n_stored), dtype=np.float32)
    SL = np.zeros((nx, n_stored), dtype=np.float32)
    VR = np.zeros((nx, n_stored), dtype=np.float32)
    SR = np.zeros((nx, n_stored), dtype=np.float32)
    SLIP  = np.zeros(n_stored, dtype=np.float32)
    SLIPR = np.zeros(n_stored, dtype=np.float32)
    TRACT = np.zeros(n_stored, dtype=np.float32)
    TIME  = np.zeros(n_stored, dtype=np.float32)

    idx = 0
    for n in range(nt):
        # rk4_step updates arrays in-place
        rk4_step(v_l, s_l, v_r, s_r, slip, psi,
                 rho, mu, nx, dx, order, r0_l, r1_r, dt, fp)
        if n % iplot == 0 and idx < n_stored:
            VL[:, idx] = v_l.astype(np.float32)
            SL[:, idx] = s_l.astype(np.float32)
            VR[:, idx] = v_r.astype(np.float32)
            SR[:, idx] = s_r.astype(np.float32)
            sliprate_val = abs(v_l[nx-1] - v_r[0])
            traction_val = Tau_0 + s_l[nx-1]
            SLIP[idx]  = float(slip[0])
            SLIPR[idx] = float(sliprate_val)
            TRACT[idx] = float(traction_val)
            TIME[idx]  = n * dt
            idx += 1

    n_stored_actual = idx
    xi = np.linspace(0, nx - 1,           nx_out, dtype=int)
    ti = np.linspace(0, n_stored_actual-1, nt_out, dtype=int)

    return dict(
        v_l      = VL[np.ix_(xi, ti)],
        s_l      = SL[np.ix_(xi, ti)],
        v_r      = VR[np.ix_(xi, ti)],
        s_r      = SR[np.ix_(xi, ti)],
        slip     = SLIP[ti],
        sliprate = SLIPR[ti],
        traction = TRACT[ti],
        time     = TIME[ti].astype(np.float32),
        x_l      = np.linspace(0,          float(p['L']),   nx_out, dtype=np.float32),
        x_r      = np.linspace(float(p['L']), 2*float(p['L']), nx_out, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FNO inference — parameterised by output resolution
# ─────────────────────────────────────────────────────────────────────────────

def fno_predict(params_raw: np.ndarray,
                param_norm, field_norms,
                model: FNO2d,
                device: torch.device,
                nx_out: int,
                nt_out: int) -> dict:
    """
    Forward pass through the FNO at any (nx_out, nt_out) resolution.

    Because FNO uses Fourier modes (modes_x=16, modes_t=16), it can run at
    any grid size ≥ 2*modes without retraining — this is the resolution
    generalisation property of the FNO architecture.

    The normalizer inverse_transform uses scalar mean/std (fit on all pixels),
    so it is shape-agnostic and works at any resolution.
    """
    # Normalise parameters (param_norm uses shape (1, N_params) → safe for any grid)
    params_norm = param_norm.transform(params_raw.reshape(1, -1)).squeeze(0)
    params_t    = torch.from_numpy(params_norm).float().unsqueeze(0).to(device)  # (1, 4)

    # Build (1, 2, nx_out, nt_out) coordinate grid in [0, 1]²
    x_grid = torch.linspace(0.0, 1.0, nx_out)
    t_grid = torch.linspace(0.0, 1.0, nt_out)
    x_2d   = x_grid.unsqueeze(1).expand(nx_out, nt_out)
    t_2d   = t_grid.unsqueeze(0).expand(nx_out, nt_out)
    coords  = torch.stack([x_2d, t_2d], dim=0).unsqueeze(0).to(device)  # (1, 2, NX, NT)

    model.eval()
    with torch.no_grad():
        pred_norm = model(params_t, coords)   # (1, 4, nx_out, nt_out)

    pred_norm = pred_norm.squeeze(0).cpu().numpy()   # (4, nx_out, nt_out)

    out = {}
    for ci, fname in enumerate(FIELD_NAMES):
        if fname in field_norms:
            out[fname] = field_norms[fname].inverse_transform(pred_norm[ci])
        else:
            out[fname] = pred_norm[ci]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bilinear upsampling helper
# ─────────────────────────────────────────────────────────────────────────────

def bilinear_upsample(field: np.ndarray, nx_out: int, nt_out: int) -> np.ndarray:
    """Bilinear upsample a 2-D (NX, NT) array to (nx_out, nt_out)."""
    t = torch.from_numpy(field).float().unsqueeze(0).unsqueeze(0)   # (1,1,NX,NT)
    upsampled = F.interpolate(t, size=(nx_out, nt_out),
                               mode='bilinear', align_corners=True)
    return upsampled.squeeze().numpy()


def upsample_dict(fields: dict, nx_out: int, nt_out: int) -> dict:
    """Upsample all four fields in a dict to (nx_out, nt_out)."""
    return {k: bilinear_upsample(fields[k], nx_out, nt_out) for k in FIELD_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    return float(np.sqrt(np.sum((pred - ref)**2)) /
                 (np.sqrt(np.sum(ref**2)) + eps))


def phys(fname: str, arr: np.ndarray) -> np.ndarray:
    """Convert stress from Pa to MPa for display."""
    return arr / 1e6 if 's_' in fname else arr


def unit(fname: str) -> str:
    return 'MPa' if 's_' in fname else 'm/s'


def _vlim(arr: np.ndarray, pct: float = 99.5) -> float:
    return float(np.percentile(np.abs(arr), pct)) + 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_superres_fields(gt_128, fno_128, fno_64_up, gt_64_up,
                         t_128, x_128, outpath: str, params_raw: np.ndarray):
    """
    5-column × 4-row grid:
      Coarse GT (up-scaled) | Fine GT | FNO@64 (up-scaled) | FNO@128 | FNO@128 error
    """
    n_fields = 4
    fig, axes = plt.subplots(n_fields, 5, figsize=(22, 4.5 * n_fields),
                             constrained_layout=True)
    fig.suptitle(
        'Super-Resolution Comparison: Simulator vs FNO\n'
        f'$\\tau_0$={params_raw[0]:.2f} MPa, '
        f'$\\alpha_s$={params_raw[1]:.3f}, '
        f'$\\alpha_d$={params_raw[2]:.3f}, '
        f'$D_c$={params_raw[3]:.2f} m',
        fontsize=13, fontweight='bold'
    )

    col_titles = [
        'Coarse Sim (64→128\nbicubic ↑) Baseline',
        'Fine Sim (128×128)\nGround Truth',
        'FNO@64 (→128\nbicubic ↑)',
        'FNO@128\nResolution Gen.',
        'FNO@128 Error\nvs Fine GT',
    ]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, fontweight='bold')

    extent = [t_128[0], t_128[-1], x_128[0], x_128[-1]]

    for row, fname in enumerate(FIELD_NAMES):
        g64u = phys(fname, gt_64_up[fname])
        g128 = phys(fname, gt_128[fname])
        f64u = phys(fname, fno_64_up[fname])
        f128 = phys(fname, fno_128[fname])
        diff = f128 - g128

        vmax  = _vlim(g128)
        vmax_d = _vlim(diff)

        cols = [
            (g64u, 'RdBu_r', -vmax,  vmax),
            (g128, 'RdBu_r', -vmax,  vmax),
            (f64u, 'RdBu_r', -vmax,  vmax),
            (f128, 'RdBu_r', -vmax,  vmax),
            (diff, 'bwr',    -vmax_d, vmax_d),
        ]

        for col, (data, cmap, vl, vh) in enumerate(cols):
            ax = axes[row, col]
            im = ax.imshow(data, origin='lower', aspect='auto',
                           extent=extent, cmap=cmap, vmin=vl, vmax=vh)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(unit(fname), fontsize=7)
            cb.ax.tick_params(labelsize=6)
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.set_ylabel('x (km)',   fontsize=7)
            ax.tick_params(labelsize=6)

            if col == 4:
                err = rel_l2(f128, g128)
                ax.set_ylabel(f'rel L2={err*100:.2f}%', fontsize=8, color='darkred')

            label = FIELD_LABELS[row]
            if col == 0:
                ax.annotate(label, xy=(-0.25, 0.5), xycoords='axes fraction',
                            fontsize=9, rotation=90, va='center', ha='right')

    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


def plot_error_comparison(gt_128, fno_128, fno_64_up, gt_64_up,
                          t_128, x_128, outpath: str):
    """
    Per-field L2 error vs the fine ground truth for all three methods.
    Top row: spatial RMS error profile.  Bottom row: temporal RMS error profile.
    """
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    fig.suptitle('Super-Resolution Error Profiles vs Fine Ground Truth (128×128)',
                 fontsize=13, fontweight='bold')

    methods = {
        'Coarse Sim ↑': (gt_64_up, 'gray',     '--'),
        'FNO@64 ↑':     (fno_64_up, 'steelblue', '-'),
        'FNO@128':      (fno_128,   'tomato',    '-'),
    }

    for ci, fname in enumerate(FIELD_NAMES):
        ref  = phys(fname, gt_128[fname])
        # Use global RMS of reference as scale (avoids division by near-zero at t=0)
        global_scale = float(np.sqrt(np.mean(ref**2))) + 1e-15

        for label, (pred_dict, color, ls) in methods.items():
            pred = phys(fname, pred_dict[fname])
            diff = pred - ref
            # RMS error along each axis, normalised by global_scale
            err_x = np.sqrt(np.mean(diff**2, axis=1)) / global_scale * 100
            err_t = np.sqrt(np.mean(diff**2, axis=0)) / global_scale * 100

            axes[0, ci].plot(x_128, err_x, color=color, ls=ls, lw=1.8, label=label)
            axes[1, ci].plot(t_128, err_t, color=color, ls=ls, lw=1.8, label=label)

        for row in range(2):
            ax = axes[row, ci]
            ax.set_title(f'{FIELD_LABELS[ci]}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_xlabel('x (km)', fontsize=8)
                ax.set_ylabel('RMS error / global RMS (%)', fontsize=8)
            else:
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('RMS error / global RMS (%)', fontsize=8)

    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


def plot_waveform_convergence(gt_128, fno_128, fno_64_up, gt_64_up,
                              t_128, x_128, outpath: str):
    """
    1-D slices through the space-time field — shows FNO@128 hits sharp fronts
    that the upsampled coarse solution misses.
    """
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    fig.suptitle('1-D Waveform Slices: Fine GT vs FNO@128 vs FNO@64↑ vs Coarse↑',
                 fontsize=12, fontweight='bold')

    # Slice positions
    ix_mid = len(x_128) // 2
    it_mid = len(t_128) // 4

    for ci, fname in enumerate(FIELD_NAMES):
        ref  = phys(fname, gt_128[fname])
        f128 = phys(fname, fno_128[fname])
        f64u = phys(fname, fno_64_up[fname])
        g64u = phys(fname, gt_64_up[fname])
        u    = unit(fname)

        # Temporal trace at x_mid
        ax = axes[0, ci]
        ax.plot(t_128, ref[ix_mid,  :], 'k-',  lw=2.5, label='Fine GT 128')
        ax.plot(t_128, f128[ix_mid, :], 'r-',  lw=1.8, label='FNO@128')
        ax.plot(t_128, f64u[ix_mid, :], 'b--', lw=1.5, label='FNO@64↑')
        ax.plot(t_128, g64u[ix_mid, :], 'g:',  lw=1.5, label='Coarse↑')
        ax.set_title(f'{FIELD_LABELS[ci]}  x={x_128[ix_mid]:.1f} km', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel(u, fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.35)

        # Spatial snapshot at t_mid
        ax2 = axes[1, ci]
        ax2.plot(x_128, ref[:, it_mid],  'k-',  lw=2.5, label=f'Fine GT 128')
        ax2.plot(x_128, f128[:, it_mid], 'r-',  lw=1.8, label='FNO@128')
        ax2.plot(x_128, f64u[:, it_mid], 'b--', lw=1.5, label='FNO@64↑')
        ax2.plot(x_128, g64u[:, it_mid], 'g:',  lw=1.5, label='Coarse↑')
        ax2.set_title(f'{FIELD_LABELS[ci]}  t={t_128[it_mid]:.2f} s', fontsize=9)
        ax2.set_xlabel('x (km)', fontsize=8)
        ax2.set_ylabel(u, fontsize=8)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.35)

    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(gt_128, fno_128, fno_64_up, gt_64_up,
                           t_fno_64, t_fno_128, t_sim,
                           params_raw: np.ndarray,
                           outpath: str):
    lines = []
    lines.append('=' * 72)
    lines.append('  SUPER-RESOLUTION COMPARISON SUMMARY')
    lines.append('  Reference: Fine simulator at 128×128')
    lines.append('=' * 72)
    lines.append(f'  Input parameters:')
    lines.append(f'    Tau_0={params_raw[0]:.4f} MPa  alp_s={params_raw[1]:.4f}  '
                 f'alp_d={params_raw[2]:.4f}  D_c={params_raw[3]:.4f} m')
    lines.append('-' * 72)
    lines.append(f'  {"Method":<25} {"Res":>8}  {"Tau (ms)":>10}  '
                 f'{"v_l err":>9}  {"s_l err":>9}  {"v_r err":>9}  {"s_r err":>9}')
    lines.append(f'  {"-"*25:<25} {"-"*8:>8}  {"-"*10:>10}  '
                 f'{"-"*9:>9}  {"-"*9:>9}  {"-"*9:>9}  {"-"*9:>9}')

    rows = [
        ('Coarse Sim (64→128↑)',  gt_64_up,  '128×128',  t_sim * 1000),
        ('FNO@64 (→128↑)',        fno_64_up, '64→128',   t_fno_64 * 1000),
        ('FNO@128 (no retrain)',  fno_128,   '128×128',  t_fno_128 * 1000),
    ]

    for name, pred, res, tau_ms in rows:
        errs = []
        for fn in FIELD_NAMES:
            ref  = phys(fn, gt_128[fn])
            p    = phys(fn, pred[fn])
            errs.append(rel_l2(p, ref) * 100)
        lines.append(f'  {name:<25} {res:>8}  {tau_ms:>9.2f}ms  '
                     + '  '.join(f'{e:>8.3f}%' for e in errs))

    lines.append('=' * 72)
    text = '\n'.join(lines)
    print(text)
    with open(outpath, 'w') as fh:
        fh.write(text + '\n')
    print(f'  → Saved: {outpath}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Super-resolution comparison: FNO resolution generalisation vs '
                    'coarse-sim baseline')
    parser.add_argument('infile',       help='Path to .in parameter file')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--config',     default='configs/fno_sw_baseline.yaml')
    parser.add_argument('--outdir',     default='output/superres_SW')
    parser.add_argument('--hires',      type=int, default=128,
                        help='High-resolution grid size (default 128)')
    args = parser.parse_args()

    NX_HI = args.hires
    NT_HI = args.hires

    os.makedirs(args.outdir, exist_ok=True)

    # ── 1. Parse input file ───────────────────────────────────────────────────
    print('\n[1] Parsing input file:', args.infile)
    user = parse_infile(args.infile)
    p    = build_params(user)
    validate(p)

    params_raw = np.array([
        float(p['Tau_0']),
        float(p['alp_s']),
        float(p['alp_d']),
        float(p['D_c']),
    ], dtype=np.float32)
    print(f'    Tau_0={params_raw[0]:.4f}  alp_s={params_raw[1]:.4f}  '
          f'alp_d={params_raw[2]:.4f}  D_c={params_raw[3]:.4f}')

    # ── 2. Run simulator at 64×64 (training res) ─────────────────────────────
    print(f'\n[2] Simulator JIT warmup + run at {NX_TRAIN}×{NT_TRAIN}...')
    _ = run_sim(p, NX_TRAIN, NT_TRAIN)   # warmup — compiles Numba kernels
    t0 = time.perf_counter()
    gt_64  = run_sim(p, NX_TRAIN, NT_TRAIN)
    t_sim_64 = time.perf_counter() - t0
    print(f'    Done in {t_sim_64*1000:.1f} ms')

    # ── 3. Run simulator at 128×128 (fine ref) ───────────────────────────────
    print(f'\n[3] Simulator run at {NX_HI}×{NT_HI} (fine reference)...')
    t0 = time.perf_counter()
    gt_128 = run_sim(p, NX_HI, NT_HI)
    t_sim_128 = time.perf_counter() - t0
    print(f'    Done in {t_sim_128*1000:.1f} ms')

    t_128 = gt_128['time']
    x_128 = gt_128['x_l']

    # ── 4. Load FNO ───────────────────────────────────────────────────────────
    print('\n[4] Loading FNO checkpoint:', args.checkpoint)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'    Device: {device}')

    m_cfg = cfg['model']
    model = FNO2d(
        n_params = m_cfg['n_params'],
        modes_x  = m_cfg['modes_x'],
        modes_t  = m_cfg['modes_t'],
        width    = m_cfg['width'],
        n_layers = m_cfg['n_layers'],
        c_out    = m_cfg['c_out'],
    ).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f'    FNO loaded ({n_params:,} parameters, modes={m_cfg["modes_x"]})')

    ckpt_dir  = os.path.dirname(args.checkpoint)
    norm_path = os.path.join(ckpt_dir, 'normalizers.pkl')
    param_norm, field_norms = load_normalizers(norm_path)
    print('    Normalizers loaded (field norms are scalar — resolution-agnostic)')

    # ── 5. FNO at training resolution 64×64 ──────────────────────────────────
    print(f'\n[5] FNO inference at {NX_TRAIN}×{NT_TRAIN} (training resolution)...')
    _ = fno_predict(params_raw, param_norm, field_norms, model, device,
                    NX_TRAIN, NT_TRAIN)          # warmup
    t0 = time.perf_counter()
    fno_64 = fno_predict(params_raw, param_norm, field_norms, model, device,
                         NX_TRAIN, NT_TRAIN)
    t_fno_64 = time.perf_counter() - t0
    print(f'    Done in {t_fno_64*1000:.2f} ms')

    # ── 6. FNO at double resolution 128×128 ──────────────────────────────────
    print(f'\n[6] FNO inference at {NX_HI}×{NT_HI} (resolution generalisation)...')
    _ = fno_predict(params_raw, param_norm, field_norms, model, device,
                    NX_HI, NT_HI)               # warmup
    t0 = time.perf_counter()
    fno_128 = fno_predict(params_raw, param_norm, field_norms, model, device,
                          NX_HI, NT_HI)
    t_fno_128 = time.perf_counter() - t0
    print(f'    Done in {t_fno_128*1000:.2f} ms')

    # ── 7. Bilinear upsample 64→128 ──────────────────────────────────────────
    print(f'\n[7] Bilinear upsampling 64×64 → {NX_HI}×{NT_HI}...')
    fno_64_up = upsample_dict(fno_64, NX_HI, NT_HI)
    gt_64_up  = upsample_dict(gt_64,  NX_HI, NT_HI)

    # ── 8. Compute errors vs fine GT ─────────────────────────────────────────
    print(f'\n[8] Errors vs fine ground truth ({NX_HI}×{NT_HI}):')
    print(f'    {"Method":<25}  {"v_l":>8}  {"s_l":>8}  {"v_r":>8}  {"s_r":>8}')
    for method_name, pred in [
        ('Coarse Sim (64→128↑)', gt_64_up),
        ('FNO@64 (→128↑)',       fno_64_up),
        ('FNO@128',              fno_128),
    ]:
        errs = [rel_l2(phys(fn, pred[fn]), phys(fn, gt_128[fn])) * 100
                for fn in FIELD_NAMES]
        print(f'    {method_name:<25}  ' +
              '  '.join(f'{e:>7.3f}%' for e in errs))

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    print('\n[9] Generating plots...')
    plot_superres_fields(
        gt_128, fno_128, fno_64_up, gt_64_up,
        t_128, x_128,
        os.path.join(args.outdir, 'fields_superres.png'),
        params_raw
    )
    plot_error_comparison(
        gt_128, fno_128, fno_64_up, gt_64_up,
        t_128, x_128,
        os.path.join(args.outdir, 'error_comparison.png')
    )
    plot_waveform_convergence(
        gt_128, fno_128, fno_64_up, gt_64_up,
        t_128, x_128,
        os.path.join(args.outdir, 'waveform_convergence.png')
    )

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print('\n[10] Summary:')
    print_and_save_summary(
        gt_128, fno_128, fno_64_up, gt_64_up,
        t_fno_64, t_fno_128, t_sim_64,
        params_raw,
        os.path.join(args.outdir, 'summary_superres.txt')
    )

    # ── 11. Save .npz ─────────────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(args.outdir, 'results_superres.npz'),
        params_raw = params_raw,
        **{f'gt64_{k}':  gt_64[k]    for k in FIELD_NAMES},
        **{f'gt128_{k}': gt_128[k]   for k in FIELD_NAMES},
        **{f'fno64_{k}': fno_64[k]   for k in FIELD_NAMES},
        **{f'fno128_{k}':fno_128[k]  for k in FIELD_NAMES},
        t_64  = gt_64['time'],
        t_128 = t_128,
        x_64  = gt_64['x_l'],
        x_128 = x_128,
    )
    print(f'\n  → All outputs saved to: {args.outdir}/')


if __name__ == '__main__':
    main()
