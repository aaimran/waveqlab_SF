#!/usr/bin/env python3
"""
compare_run.py — Run the Numba simulator AND the trained FNO on the exact
parameters from an input file, then produce side-by-side comparison plots
and a printed/saved summary table.

Usage:
    python3 compare_run.py input/rupture_1d_SW.in \
            --checkpoint checkpoints/best.pt \
            --config     configs/fno_sw_baseline.yaml \
            --outdir     output/compare_SW

Outputs (in --outdir):
    fields_comparison.png   — 4×3 grid: v_l, s_l, v_r, s_r  (sim | FNO | diff)
    fault_timeseries.png    — slip, slip-rate, traction vs time
    error_profile.png       — per-field spatial & temporal L2 error profiles
    summary.txt             — plain-text metrics table
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, 'src')
sys.path.insert(0, _HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml required: pip install pyyaml")

from rupture_1d import parse_infile, build_params, validate, init_fields, build_friction_parameters
from data_gen.dataset import load_normalizers
from model.fno import FNO2d
from src.kernels import rk4_step

FIELD_NAMES   = ['v_l', 's_l', 'v_r', 's_r']

# ─── Inlined from generate_dataset.py (avoids slow scipy import) ────────────
PARAM_NAMES = ['Tau_0', 'alp_s', 'alp_d', 'D_c']
PARAM_BOUNDS = {
    'Tau_0': (78.0, 88.0),
    'alp_s': (0.62, 0.74),
    'alp_d': (0.45, 0.55),
    'D_c'  : (0.2,  0.8),
}
NX_OUT = 64
NT_OUT = 64


def _run_sim_inmem(p):
    """Run simulator in-memory, return downsampled fields."""
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
    SLIP   = np.zeros(n_stored, dtype=np.float32)
    SLIPR  = np.zeros(n_stored, dtype=np.float32)
    TRACT  = np.zeros(n_stored, dtype=np.float32)
    TIME   = np.zeros(n_stored, dtype=np.float32)

    idx = 0
    for n in range(nt):
        rk4_step(v_l, s_l, v_r, s_r, slip, psi,
                 rho, mu, nx, dx, order, r0_l, r1_r, dt, fp)
        # rk4_step updates v_l, s_l, v_r, s_r, slip, psi in-place
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
    xi = np.linspace(0, nx-1, NX_OUT, dtype=int)
    ti = np.linspace(0, n_stored_actual-1, NT_OUT, dtype=int)

    return dict(
        v_l      = VL[np.ix_(xi, ti)],
        s_l      = SL[np.ix_(xi, ti)],
        v_r      = VR[np.ix_(xi, ti)],
        s_r      = SR[np.ix_(xi, ti)],
        slip     = SLIP[ti],
        sliprate = SLIPR[ti],
        traction = TRACT[ti],
        time     = TIME[ti].astype(np.float32),
        x_l      = np.linspace(0, float(p['L']), NX_OUT, dtype=np.float32),
        x_r      = np.linspace(float(p['L']), 2*float(p['L']), NX_OUT, dtype=np.float32),
    )
FIELD_LABELS  = [r'$v_l$ (m/s)', r'$s_l$ (MPa)', r'$v_r$ (m/s)', r'$s_r$ (MPa)']
FIELD_CMAPS   = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> float:
    return float(np.sqrt(np.sum((pred - ref) ** 2)) /
                 (np.sqrt(np.sum(ref ** 2)) + eps))


def stress_to_mpa(arr: np.ndarray) -> np.ndarray:
    """Convert stress from Pa to MPa."""
    return arr / 1e6


def _symmetric_vlim(arr: np.ndarray, pct: float = 99.5) -> float:
    return float(np.percentile(np.abs(arr), pct))


# ─────────────────────────────────────────────────────────────────────────────
# FNO inference
# ─────────────────────────────────────────────────────────────────────────────

def fno_predict(params_raw: np.ndarray,
                param_norm, field_norms,
                model: FNO2d,
                device: torch.device,
                nx_out: int,
                nt_out: int) -> dict:
    """
    Forward pass through the FNO.

    params_raw : (4,) array  [Tau_0, alp_s, alp_d, D_c]

    Returns dict of (nx_out, nt_out) arrays for v_l, s_l, v_r, s_r.
    """
    # Normalise parameters
    params_norm = param_norm.transform(params_raw.reshape(1, -1)).squeeze(0)  # (4,)
    params_t    = torch.from_numpy(params_norm).float().unsqueeze(0)          # (1, 4)

    # Build (1, 2, NX, NT) coordinate grid
    x_grid = torch.linspace(0.0, 1.0, nx_out)
    t_grid = torch.linspace(0.0, 1.0, nt_out)
    x_2d   = x_grid.unsqueeze(1).expand(nx_out, nt_out)   # (NX, NT)
    t_2d   = t_grid.unsqueeze(0).expand(nx_out, nt_out)   # (NX, NT)
    coord_grid = torch.stack([x_2d, t_2d], dim=0).unsqueeze(0).to(device)  # (1, 2, NX, NT)

    params_t = params_t.to(device)

    model.eval()
    with torch.no_grad():
        pred_norm = model(params_t, coord_grid)   # (1, 4, NX, NT)

    pred_norm = pred_norm.squeeze(0).cpu().numpy()  # (4, NX, NT)

    # Denormalise each field
    out = {}
    for ci, fname in enumerate(FIELD_NAMES):
        if fname in field_norms:
            out[fname] = field_norms[fname].inverse_transform(pred_norm[ci])
        else:
            out[fname] = pred_norm[ci]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fields_comparison(sim: dict, fno: dict, t: np.ndarray, x: np.ndarray,
                           outpath: str, params_raw: np.ndarray):
    """4-row × 3-column: [Sim | FNO | Difference] for each field."""
    n_fields = len(FIELD_NAMES)
    fig = plt.figure(figsize=(16, 4.2 * n_fields))
    fig.suptitle(
        f'Simulator vs FNO Comparison\n'
        f'$\\tau_0$={params_raw[0]:.2f} MPa,  '
        f'$\\alpha_s$={params_raw[1]:.3f},  '
        f'$\\alpha_d$={params_raw[2]:.3f},  '
        f'$D_c$={params_raw[3]:.2f} m',
        fontsize=14, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(n_fields, 3, figure=fig, hspace=0.45, wspace=0.35)

    # stress fields: convert Pa → MPa
    def phys(fname, arr):
        if 's_' in fname:
            return arr / 1e6
        return arr

    def unit(fname):
        return 'MPa' if 's_' in fname else 'm/s'

    for row, fname in enumerate(FIELD_NAMES):
        s_data  = phys(fname, sim[fname])
        f_data  = phys(fname, fno[fname])
        diff    = f_data - s_data
        vmax    = _symmetric_vlim(s_data)
        vmax_d  = _symmetric_vlim(diff)

        for col, (data, title, cmap, vl, vh) in enumerate([
            (s_data, 'Simulator', 'RdBu_r', -vmax,  vmax),
            (f_data, 'FNO',       'RdBu_r', -vmax,  vmax),
            (diff,   'Difference','bwr',    -vmax_d, vmax_d),
        ]):
            ax = fig.add_subplot(gs[row, col])
            # extent: x in km, t in s
            extent = [t[0], t[-1], x[0], x[-1]]
            im = ax.imshow(data, origin='lower', aspect='auto',
                           extent=extent,
                           cmap=cmap, vmin=vl, vmax=vh)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(unit(fname), fontsize=8)
            cb.ax.tick_params(labelsize=7)

            rel_err = rel_l2(f_data, s_data) * 100 if col == 2 else None
            if col < 2:
                ax.set_title(f'{FIELD_LABELS[row]}  [{title}]', fontsize=9)
            else:
                ax.set_title(
                    f'{FIELD_LABELS[row]}  Diff  |rel L2={rel_err:.2f}%|',
                    fontsize=9, color='darkred')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('x (km)',   fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


def plot_fault_timeseries(sim: dict, fno: dict, t: np.ndarray, outpath: str):
    """Fault slip, slip-rate, and traction vs time."""
    # Reconstruct from fields at fault node (last node of left domain)
    # slip-rate ≈ |v_l[-1, :] - v_r[0, :]|
    sr_sim = np.abs(sim['v_l'][-1, :] - sim['v_r'][0, :])
    sr_fno = np.abs(fno['v_l'][-1, :] - fno['v_r'][0, :])

    # traction ≈ s_l[-1, :] (relative to initial, in MPa)
    tr_sim = sim['s_l'][-1, :] / 1e6
    tr_fno = fno['s_l'][-1, :] / 1e6

    # fault velocity discontinuity as proxy for slip rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(t, sr_sim, 'b-',  lw=2.0, label='Simulator')
    ax.plot(t, sr_fno, 'r--', lw=2.0, label='FNO')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Fault slip-rate |$v_l - v_r$| (m/s)', fontsize=11)
    ax.set_title('Fault Slip-Rate at $x = L$', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t, tr_sim, 'b-',  lw=2.0, label='Simulator')
    ax.plot(t, tr_fno, 'r--', lw=2.0, label='FNO')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Shear stress $s_l(x_f)$ (MPa)', fontsize=11)
    ax.set_title('Fault-Node Shear Stress', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    rel_sr = rel_l2(sr_fno, sr_sim) * 100
    rel_tr = rel_l2(tr_fno, tr_sim) * 100
    fig.suptitle(
        f'Fault Time Series  |  slip-rate rel-L2={rel_sr:.2f}%,  '
        f'traction rel-L2={rel_tr:.2f}%',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


def plot_error_profiles(sim: dict, fno: dict, t: np.ndarray, x: np.ndarray, outpath: str):
    """Spatial L2 profile (collapsed over t) and temporal L2 profile (collapsed over x)."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    fig.suptitle('Per-Field Error Profiles', fontsize=13, fontweight='bold')

    def phys(fn, arr):
        return arr / 1e6 if 's_' in fn else arr

    for ci, fname in enumerate(FIELD_NAMES):
        s = phys(fname, sim[fname])   # (NX, NT)
        f = phys(fname, fno[fname])
        diff = f - s

        # Spatial profile: RMS over time axis
        err_x = np.sqrt(np.mean(diff**2, axis=1))   # (NX,)
        sig_x = np.sqrt(np.mean(s**2, axis=1)) + 1e-15
        rel_x = err_x / sig_x * 100

        # Temporal profile: RMS over space axis
        err_t  = np.sqrt(np.mean(diff**2, axis=0))  # (NT,)
        sig_t  = np.sqrt(np.mean(s**2, axis=0)) + 1e-15
        rel_t  = err_t / sig_t * 100

        ax_x = axes[0, ci]
        ax_x.plot(x, rel_x, lw=2, color='steelblue')
        ax_x.set_xlabel('x (km)', fontsize=9)
        ax_x.set_ylabel('Rel. error (%)', fontsize=9)
        ax_x.set_title(f'{FIELD_LABELS[ci]}  spatial', fontsize=9)
        ax_x.grid(True, alpha=0.4)
        ax_x.fill_between(x, 0, rel_x, alpha=0.2, color='steelblue')

        ax_t = axes[1, ci]
        ax_t.plot(t, rel_t, lw=2, color='tomato')
        ax_t.set_xlabel('Time (s)', fontsize=9)
        ax_t.set_ylabel('Rel. error (%)', fontsize=9)
        ax_t.set_title(f'{FIELD_LABELS[ci]}  temporal', fontsize=9)
        ax_t.grid(True, alpha=0.4)
        ax_t.fill_between(t, 0, rel_t, alpha=0.2, color='tomato')

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


def plot_waveform_slices(sim: dict, fno: dict, t: np.ndarray, x: np.ndarray, outpath: str):
    """1-D slices through the space-time field at two fixed x positions and two fixed t."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    fig.suptitle('1-D Waveform Slices: Simulator vs FNO', fontsize=13, fontweight='bold')

    ix_mid  = len(x) // 4        # quarter-way through domain
    ix_far  = 3 * len(x) // 4   # three-quarter through
    it_early = len(t) // 4
    it_late  = 3 * len(t) // 4

    def phys(fn, arr):
        return arr / 1e6 if 's_' in fn else arr

    for ci, fname in enumerate(FIELD_NAMES):
        s = phys(fname, sim[fname])
        f = phys(fname, fno[fname])
        unit = 'MPa' if 's_' in fname else 'm/s'

        # Row 0: v vs time at two x-positions
        ax = axes[0, ci]
        ax.plot(t, s[ix_mid,  :], 'b-',  lw=1.8, label=f'Sim x={x[ix_mid]:.1f} km')
        ax.plot(t, f[ix_mid,  :], 'b--', lw=1.5)
        ax.plot(t, s[ix_far,  :], 'r-',  lw=1.8, label=f'Sim x={x[ix_far]:.1f} km')
        ax.plot(t, f[ix_far,  :], 'r--', lw=1.5, label='FNO (dashed)')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel(f'{fname} ({unit})', fontsize=8)
        ax.set_title(f'{FIELD_LABELS[ci]}: time trace', fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.4)

        # Row 1: snapshot at two times
        ax2 = axes[1, ci]
        ax2.plot(x, s[:, it_early], 'b-',  lw=1.8, label=f'Sim t={t[it_early]:.2f} s')
        ax2.plot(x, f[:, it_early], 'b--', lw=1.5)
        ax2.plot(x, s[:, it_late],  'r-',  lw=1.8, label=f'Sim t={t[it_late]:.2f} s')
        ax2.plot(x, f[:, it_late],  'r--', lw=1.5, label='FNO (dashed)')
        ax2.set_xlabel('x (km)', fontsize=8)
        ax2.set_ylabel(f'{fname} ({unit})', fontsize=8)
        ax2.set_title(f'{FIELD_LABELS[ci]}: snapshot', fontsize=9)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {outpath}')


# ─────────────────────────────────────────────────────────────────────────────
# Summary text
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(sim: dict, fno: dict,
                           t_sim: float, t_fno: float,
                           params_raw: np.ndarray,
                           t_arr: np.ndarray,
                           outpath: str):
    lines = []
    lines.append('=' * 65)
    lines.append('  SIMULATOR vs FNO COMPARISON SUMMARY')
    lines.append('=' * 65)
    lines.append(f'  Input parameters:')
    lines.append(f'    Tau_0  = {params_raw[0]:.4f} MPa')
    lines.append(f'    alp_s  = {params_raw[1]:.4f}')
    lines.append(f'    alp_d  = {params_raw[2]:.4f}')
    lines.append(f'    D_c    = {params_raw[3]:.4f} m')
    lines.append('-' * 65)
    lines.append(f'  Timing:')
    lines.append(f'    Simulator runtime : {t_sim:.4f} s')
    lines.append(f'    FNO inference     : {t_fno*1000:.4f} ms')
    lines.append(f'    Speedup           : {t_sim/t_fno:,.1f}×')
    lines.append('-' * 65)
    lines.append(f'  {"Field":<10} {"Rel L2 error":>14}  {"Max abs diff":>14}  {"Units":>6}')
    lines.append(f'  {"-"*10:<10} {"-"*14:>14}  {"-"*14:>14}  {"-"*6:>6}')

    for fname in FIELD_NAMES:
        s = sim[fname]
        f = fno[fname]
        if 's_' in fname:
            s = s / 1e6; f = f / 1e6
            unit = 'MPa'
        else:
            unit = 'm/s'
        err  = rel_l2(f, s) * 100
        mabs = float(np.max(np.abs(f - s)))
        lines.append(f'  {fname:<10} {err:>13.3f}%  {mabs:>14.4e}  {unit:>6}')

    # fault-node slip-rate comparison
    sr_sim = np.abs(sim['v_l'][-1, :] - sim['v_r'][0, :])
    sr_fno = np.abs(fno['v_l'][-1, :] - fno['v_r'][0, :])
    lines.append('-' * 65)
    lines.append(f'  Peak slip-rate (sim): {sr_sim.max():.4f} m/s  '
                 f'(FNO): {sr_fno.max():.4f} m/s  '
                 f'err: {rel_l2(sr_fno, sr_sim)*100:.2f}%')
    lines.append(f'  Peak slip-rate time (sim): {t_arr[sr_sim.argmax()]:.3f} s  '
                 f'(FNO): {t_arr[sr_fno.argmax()]:.3f} s')
    lines.append('=' * 65)

    text = '\n'.join(lines)
    print(text)
    with open(outpath, 'w') as fh:
        fh.write(text + '\n')
    print(f'  → Saved: {outpath}')
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare Numba simulator vs trained FNO on a single .in run')
    parser.add_argument('infile',      help='Path to .in parameter file')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--config',     default='configs/fno_sw_baseline.yaml')
    parser.add_argument('--outdir',     default='output/compare_SW')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── 1. Parse input file ───────────────────────────────────────────────────
    print('\n[1] Parsing input file:', args.infile)
    user   = parse_infile(args.infile)
    p      = build_params(user)
    validate(p)

    params_raw = np.array([
        float(p['Tau_0']),
        float(p['alp_s']),
        float(p['alp_d']),
        float(p['D_c']),
    ], dtype=np.float32)

    print(f'    Parameters: Tau_0={params_raw[0]:.4f} MPa, '
          f'alp_s={params_raw[1]:.4f}, alp_d={params_raw[2]:.4f}, '
          f'D_c={params_raw[3]:.4f} m')

    # Check parameter bounds (training range)
    for i, n in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[n]
        v = params_raw[i]
        flag = 'IN RANGE' if lo <= v <= hi else f'OUT OF RANGE [{lo}, {hi}]'
        print(f'    {n:8s} = {v:.4f}  [{lo}, {hi}]  -> {flag}')

    # ── 2. Run simulator ──────────────────────────────────────────────────────
    print('\n[2] Running Numba simulator (JIT warmup on first call)...')
    # Warmup (compile Numba kernels silently)
    _ = _run_sim_inmem(p)   # warmup run; result discarded

    t0 = time.perf_counter()
    sim_out = _run_sim_inmem(p)
    t_sim   = time.perf_counter() - t0
    print(f'    Simulator done in {t_sim:.4f} s')

    t_coords = sim_out['time']        # (NT_OUT,)
    x_coords = sim_out['x_l']         # (NX_OUT,)  — use left domain x-coords

    # ── 3. Load FNO ───────────────────────────────────────────────────────────
    print('\n[3] Loading FNO checkpoint:', args.checkpoint)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'    Device: {device}')

    m_cfg = cfg['model']
    model = FNO2d(
        n_params  = m_cfg['n_params'],
        modes_x   = m_cfg['modes_x'],
        modes_t   = m_cfg['modes_t'],
        width     = m_cfg['width'],
        n_layers  = m_cfg['n_layers'],
        c_out     = m_cfg['c_out'],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    print(f'    FNO loaded  ({sum(pp.numel() for pp in model.parameters()):,} parameters)')

    ckpt_dir  = os.path.dirname(args.checkpoint)
    norm_path = os.path.join(ckpt_dir, 'normalizers.pkl')
    param_norm, field_norms = load_normalizers(norm_path)
    print('    Normalizers loaded')

    nx_out = cfg['data']['nx_out']
    nt_out = cfg['data']['nt_out']

    # ── 4. FNO inference ──────────────────────────────────────────────────────
    print('\n[4] Running FNO inference...')
    # Warmup
    _ = fno_predict(params_raw, param_norm, field_norms, model, device, nx_out, nt_out)

    t0 = time.perf_counter()
    fno_out = fno_predict(params_raw, param_norm, field_norms, model, device, nx_out, nt_out)
    t_fno   = time.perf_counter() - t0
    print(f'    FNO inference done in {t_fno*1000:.4f} ms')

    # ── 5. Align grids ────────────────────────────────────────────────────────
    # FNO output is in normalised units; denormalise already done in fno_predict.
    # Simulator output: v in m/s, s in Pa — FNO should match these units.
    # Match time and space arrays:
    t = t_coords
    x_l = sim_out['x_l']
    x_r = sim_out['x_r']

    # Combine x dimensions for spatial plots (use left domain for profiles)
    x_plot = x_l

    # ── 6. Compute errors ────────────────────────────────────────────────────
    print('\n[5] Computing errors...')
    for fname in FIELD_NAMES:
        s = sim_out[fname]
        f = fno_out[fname]
        unit = 'MPa' if 's_' in fname else 'm/s'
        scale = 1e6 if 's_' in fname else 1.0
        err = rel_l2(f / scale, s / scale)
        print(f'    {fname}: rel L2 = {err*100:.3f}%')

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print('\n[6] Generating plots...')
    plot_fields_comparison(
        sim_out, fno_out, t, x_plot,
        os.path.join(args.outdir, 'fields_comparison.png'),
        params_raw
    )
    plot_fault_timeseries(
        sim_out, fno_out, t,
        os.path.join(args.outdir, 'fault_timeseries.png')
    )
    plot_error_profiles(
        sim_out, fno_out, t, x_plot,
        os.path.join(args.outdir, 'error_profiles.png')
    )
    plot_waveform_slices(
        sim_out, fno_out, t, x_plot,
        os.path.join(args.outdir, 'waveform_slices.png')
    )

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print('\n[7] Summary:')
    print_and_save_summary(
        sim_out, fno_out, t_sim, t_fno, params_raw, t,
        os.path.join(args.outdir, 'summary.txt')
    )

    # ── 9. Save combined .npz ────────────────────────────────────────────────
    np.savez_compressed(
        os.path.join(args.outdir, 'results.npz'),
        params_raw = params_raw,
        t          = t,
        x_l        = x_l,
        x_r        = x_r,
        **{f'sim_{k}': sim_out[k] for k in FIELD_NAMES + ['slip', 'sliprate', 'traction']},
        **{f'fno_{k}': fno_out[k] for k in FIELD_NAMES},
    )
    print(f'\n  → Results saved to: {args.outdir}/')


if __name__ == '__main__':
    main()
