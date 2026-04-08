#!/usr/bin/env python3
"""
Notebook-style plots from a 2D rupture .npz file
==================================================
Replicates the visual style of elasticwave2D_new.ipynb exactly:
  - Combined left+right wavefield stitched into one image  (seismic colormap)
  - On-fault slip / slip-rate / traction animation          (auto y-limits)
  - Space-time slip rate                                    (fault on x, time on y)
  - Domain snapshots at Nt evenly-spaced times              (static PNG grid)

Usage:
    python3 auxiliary/notebook_plots.py output/rupture_2d_SW_<id>.npz [options]

Options:
    --out-dir DIR     Save plots here (default: plots/ next to script)
    --no-anim         Skip GIF animations (PNG only)
    --frames N        Max animation frames [150]
    --vmax FLOAT      Colour scale cap for wavefield [2.0]
    --snapshots N     Number of domain snapshot times to show [4]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def _save_anim(ani, path, fps=15):
    try:
        import shutil
        if shutil.which('ffmpeg'):
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                            extra_args=['-vcodec', 'libx264',
                                                        '-pix_fmt', 'yuv420p'])
            ani.save(path, writer=writer)
            print(f"Saved: {path}")
            return
    except Exception:
        pass
    gif_path = path.replace('.mp4', '.gif')
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
    print(f"Saved: {gif_path}")


# ── 1. Combined wavefield animation ──────────────────────────────────────────
def plot_wavefield_combined(data, meta, out_dir, tag, max_frames=150, vmax=2.0):
    """Single stitched left+right domain image, matching notebook cell."""
    DO_l = data['DomainOutput_l']   # (nx, ny, n_snap, nf)
    DO_r = data['DomainOutput_r']
    snap_times = data['snap_times']
    Lx = float(meta['Lx'])
    Ly = float(meta['Ly'])
    mode = meta.get('mode', 'II')

    n_snap = DO_l.shape[2]
    stride = max(1, n_snap // max_frames)
    frames = list(range(0, n_snap, stride))

    fi = 1 if mode == 'II' else 0   # vy (mode II) or vz (mode III)

    def _stitch(idx):
        p_l = DO_l[:, :, idx, fi].T   # (ny, nx)
        p_r = DO_r[:, :, idx, fi].T
        return np.hstack([p_l, p_r])  # (ny, 2*nx)

    fig, ax = plt.subplots(figsize=(8, 5))
    img = _stitch(0)
    im = ax.imshow(img, aspect='auto', extent=[-Lx, Lx, Ly, 0],
                   cmap='seismic', vmin=-vmax, vmax=vmax, interpolation='none')
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    fig.colorbar(im, ax=ax, label=f'v{"y" if mode=="II" else "z"} [m/s]')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    title = ax.set_title(f'Wavefield (v{"y" if mode=="II" else "z"}) | fric={meta["fric_law"]} | t = {float(snap_times[0]):.2f} s')
    fig.tight_layout()

    def _update(k):
        im.set_data(_stitch(k))
        title.set_text(
            f'Wavefield (v{"y" if mode=="II" else "z"}) | fric={meta["fric_law"]} | t = {float(snap_times[k]):.2f} s')
        return [im, title]

    ani = animation.FuncAnimation(fig, _update, frames=frames, blit=True)
    _save_anim(ani, os.path.join(out_dir, f'{tag}_nb_wavefield.mp4'), fps=15)
    plt.close(fig)


# ── 2. Fault evolution animation ─────────────────────────────────────────────
def plot_fault_evolution(data, meta, out_dir, tag, max_frames=150):
    """Slip / slip-rate / traction evolution — matches notebook cell."""
    FO   = data['FaultOutput']   # (ny, nt, 6)
    time = data['time']
    y    = data['y_fault']
    Ly   = float(meta['Ly'])
    dt   = float(time[1] - time[0]) if len(time) > 1 else 1.0
    nt   = FO.shape[1]
    mode = meta.get('mode', 'II')

    stride = max(1, nt // max_frames)
    frames = list(range(0, nt, stride))

    slip_max  = max(float(FO[:, :, 4].max()), 1e-10)
    sr_max    = max(float(np.sqrt(FO[:,:,0]**2 + FO[:,:,1]**2).max()), 1e-10)
    tau_min   = float(FO[:, :, 3].min()) - 1
    tau_max   = float(FO[:, :, 3].max()) + 1

    fig, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax3.set_ylabel('Slip [m]')
    ax3.set_ylim([0, slip_max * 1.1])
    ax4.set_ylabel('Slip rate [m/s]')
    ax4.set_ylim([0, sr_max * 1.1])
    ax5.set_ylabel('Traction [MPa]')
    ax5.set_xlabel('fault [km]')
    ax5.set_ylim([tau_min, tau_max])
    for ax in (ax3, ax4, ax5):
        ax.set_xlim([0, Ly])
        ax.grid(True, alpha=0.3)

    line3, = ax3.plot([], [], 'g', lw=2, label='slip')
    line4, = ax4.plot([], [], 'r', lw=2, label='slip rate')
    line5, = ax5.plot([], [], 'b', lw=2, label='traction')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')
    ax5.legend(loc='upper right')
    sup = fig.suptitle('t = 0.00 s', y=1.005)
    fig.tight_layout()

    def _update(it):
        slip_     = FO[:, it, 4]
        sliprate_ = np.sqrt(FO[:, it, 0]**2 + FO[:, it, 1]**2)
        traction_ = FO[:, it, 3]
        line3.set_data(y, slip_)
        line4.set_data(y, sliprate_)
        line5.set_data(y, traction_)
        sup.set_text(f'On-fault evolution | fric={meta["fric_law"]} | t = {float(time[it]):.2f} s')
        return line3, line4, line5

    ani = animation.FuncAnimation(fig, _update, frames=frames, blit=False)
    _save_anim(ani, os.path.join(out_dir, f'{tag}_nb_fault_evolution.mp4'), fps=10)
    plt.close(fig)


# ── 3. Space-time slip rate (fault on x, time on y — notebook orientation) ──
def plot_sliprate_spacetime(data, meta, out_dir, tag):
    """Space-time slip rate with fault position on x-axis, time on y-axis."""
    FO   = data['FaultOutput']   # (ny, nt, 6)
    time = data['time']
    y    = data['y_fault']
    Ly   = float(meta['Ly'])

    VT = np.sqrt(FO[:, :, 0]**2 + FO[:, :, 1]**2)   # (ny, nt)
    vmax = min(float(VT.max()), 10.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(VT.T, aspect='auto',
                   extent=[float(y[0]), float(y[-1]),
                            float(time[-1]), float(time[0])],
                   cmap='viridis', vmin=0, vmax=vmax,
                   interpolation='bilinear')
    fig.colorbar(im, ax=ax, label='slip rate [m/s]')
    ax.set_xlabel('fault [km]')
    ax.set_ylabel('t [s]')
    ax.set_title(
        f'On-fault slip rate | {meta["fric_law"]} | run={meta["run_id"]}')
    fig.tight_layout()
    path = os.path.join(out_dir, f'{tag}_nb_sliprate_spacetime.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── 4. Domain snapshots grid ─────────────────────────────────────────────────
def plot_domain_snapshots(data, meta, out_dir, tag, n_times=4, vmax=2.0):
    """Grid of stitched wavefield snapshots at N evenly-spaced times."""
    DO_l = data['DomainOutput_l']
    DO_r = data['DomainOutput_r']
    snap_times = data['snap_times']
    Lx = float(meta['Lx'])
    Ly = float(meta['Ly'])
    mode = meta.get('mode', 'II')
    fi = 1 if mode == 'II' else 0

    n_snap = DO_l.shape[2]
    idxs = [int(i) for i in np.linspace(0, n_snap - 1, n_times)]

    ncols = min(n_times, 4)
    nrows = (n_times + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for k, idx in enumerate(idxs):
        p_l = DO_l[:, :, idx, fi].T
        p_r = DO_r[:, :, idx, fi].T
        img = np.hstack([p_l, p_r])
        im = axes[k].imshow(img, aspect='auto', extent=[-Lx, Lx, Ly, 0],
                            cmap='seismic', vmin=-vmax, vmax=vmax,
                            interpolation='bilinear')
        axes[k].axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
        axes[k].set_title(f't = {float(snap_times[idx]):.2f} s')
        axes[k].set_xlabel('x [km]')
        axes[k].set_ylabel('y [km]')
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    for k in range(n_times, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f'Domain snapshots | {meta["fric_law"]} | run={meta["run_id"]}', y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{tag}_nb_domain_snapshots.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


# ── 5. Final fault state ──────────────────────────────────────────────────────
def plot_fault_final(data, meta, out_dir, tag):
    """Final-time slip / slip-rate / traction with auto y-limits."""
    FO   = data['FaultOutput']
    time = data['time']
    y    = data['y_fault']

    slip_f = FO[:, -1, 4]
    sr_f   = np.sqrt(FO[:, -1, 0]**2 + FO[:, -1, 1]**2)
    tau_f  = FO[:, -1, 3]
    t_end  = float(time[-1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax1.plot(y, slip_f, 'g', lw=2)
    ax1.set_ylabel('Slip [m]')
    ax1.grid(True, alpha=0.3)

    ax2.plot(y, sr_f, 'r', lw=2)
    ax2.set_ylabel('Slip rate [m/s]')
    ax2.grid(True, alpha=0.3)

    ax3.plot(y, tau_f, 'b', lw=2)
    ax3.set_ylabel('Traction [MPa]')
    ax3.set_xlabel('fault position [km]')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f'Final snapshot (t = {t_end:.2f} s) | fric_law = {meta["fric_law"]} | run_id = {meta["run_id"]}')
    fig.tight_layout()
    path = os.path.join(out_dir, f'{tag}_nb_fault_final.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Notebook-style plots from a 2D rupture .npz file')
    parser.add_argument('npz', help='Path to .npz output file')
    parser.add_argument('--no-anim', action='store_true',
                        help='Skip animations (PNG only)')
    parser.add_argument('--frames', type=int, default=150,
                        help='Max animation frames [150]')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory [default: plots/ next to script]')
    parser.add_argument('--vmax', type=float, default=2.0,
                        help='Wavefield colour scale cap [2.0]')
    parser.add_argument('--snapshots', type=int, default=4,
                        help='Number of domain snapshot times [4]')
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: file not found: {args.npz}")
        sys.exit(1)

    data = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(data['metadata']))

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(_script_dir), 'plots')
    out_dir = args.out_dir if args.out_dir else default_out
    os.makedirs(out_dir, exist_ok=True)

    run_id = meta.get('run_id', 'unknown')
    fric   = meta.get('fric_law', 'XX')
    tag    = f"{fric}_{run_id}"

    print(f"run_id   : {run_id}")
    print(f"fric_law : {fric}   mode: {meta.get('mode','?')}")
    print(f"output   : {out_dir}/\n")

    # static plots
    plot_sliprate_spacetime(data, meta, out_dir, tag)
    plot_fault_final(data, meta, out_dir, tag)
    plot_domain_snapshots(data, meta, out_dir, tag,
                          n_times=args.snapshots, vmax=args.vmax)

    # animations
    if not args.no_anim:
        print("Generating combined wavefield animation…")
        plot_wavefield_combined(data, meta, out_dir, tag,
                                max_frames=args.frames, vmax=args.vmax)
        print("Generating fault evolution animation…")
        plot_fault_evolution(data, meta, out_dir, tag, max_frames=args.frames)

    print("\nDone.")


if __name__ == '__main__':
    main()
