#!/usr/bin/env python3
"""
2D Rupture Simulation — Output Plots
======================================
Usage:
    python3 auxiliary/output_2d.py output/rupture_2d_SW_<run_id>.npz

Produces (saved next to the .npz file):
    <prefix>_<run_id>_wavefield.mp4        — animated vy wavefield in both domains
    <prefix>_<run_id>_fault_evolution.mp4  — animated slip / slip-rate / traction along fault
    <prefix>_<run_id>_fault_final.png      — final-time fault profiles
    <prefix>_<run_id>_sliprate_spacetime.png — space-time slip-rate image
    <prefix>_<run_id>_friction_params.png  — friction parameter profiles along fault
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

def _ffmpeg_available():
    import shutil
    return shutil.which('ffmpeg') is not None


def _save_anim(ani, path, fps=20):
    """Save animation as .mp4 (ffmpeg) or fall back to .gif (pillow)."""
    if _ffmpeg_available():
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=['-vcodec', 'libx264',
                                                    '-pix_fmt', 'yuv420p'])
        ani.save(path, writer=writer)
        print(f"Saved: {path}")
    else:
        gif_path = path.replace('.mp4', '.gif')
        ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
        print(f"ffmpeg not found — saved GIF instead: {gif_path}")


# ── 1. Wavefield animation ─────────────────────────────────────────────────

def plot_wavefield(data, meta, out_dir, tag, max_frames=150):
    DO_l = data['DomainOutput_l']   # (nx, ny, n_snap, nf)
    DO_r = data['DomainOutput_r']
    snap_times = data['snap_times']
    x_l  = data['x_l']
    x_r  = data['x_r']
    y    = data['y_fault']
    Lx   = float(meta['Lx'])
    Ly   = float(meta['Ly'])

    n_snap = DO_l.shape[2]
    stride = max(1, n_snap // max_frames)
    frames = list(range(0, n_snap, stride))

    # field index 1 = vy (mode II) or 0 = vz (mode III)
    fi = 1 if meta.get('mode', 'II') == 'II' else 0

    vmax = max(
        np.abs(DO_l[:, :, :, fi]).max(),
        np.abs(DO_r[:, :, :, fi]).max(),
        1e-10)
    vmax = min(vmax, 5.0)    # cap colour scale

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    ims = []
    titles = []
    for ax, D, x, side in zip(axes,
                               [DO_l, DO_r],
                               [x_l,  x_r],
                               ['Left domain', 'Right domain']):
        im = ax.imshow(D[:, :, 0, fi].T, aspect='auto',
                       extent=[x.min(), x.max(), y.max(), y.min()],
                       cmap='seismic', vmin=-vmax, vmax=vmax,
                       interpolation='bilinear')
        fig.colorbar(im, ax=ax, label='vy [m/s]', shrink=0.8)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_title(side)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ims.append(im)

    sup = fig.suptitle('', y=1.01)

    def _update(idx):
        it = frames[idx]
        for im, D in zip(ims, [DO_l, DO_r]):
            im.set_data(D[:, :, it, fi].T)
        sup.set_text(
            f"Wavefield (vy)  |  fric={meta['fric_law']}  |  "
            f"t = {snap_times[it]:.2f} s")
        return ims + [sup]

    ani = animation.FuncAnimation(fig, _update, frames=len(frames),
                                  blit=False, interval=50)
    _save_anim(ani, os.path.join(out_dir, f'{tag}_wavefield.mp4'), fps=20)
    plt.close(fig)


# ── 2. Fault evolution animation ──────────────────────────────────────────

def plot_fault_evolution(data, meta, out_dir, tag, max_frames=150):
    FO   = data['FaultOutput']   # (ny, nt, 6)
    time = data['time']
    y    = data['y_fault']
    ny, nt, _ = FO.shape

    stride = max(1, nt // max_frames)
    frames = list(range(0, nt, stride))

    slip_max     = max(FO[:, :, 4].max(), 1.0)
    sliprate_max = max(np.sqrt(FO[:, :, 0]**2 + FO[:, :, 1]**2).max(), 0.1)
    trac_min     = FO[:, :, 3].min() - 1.0
    trac_max     = FO[:, :, 3].max() + 1.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    line1, = ax1.plot([], [], 'g',  lw=1.5, label='slip')
    line2, = ax2.plot([], [], 'r',  lw=1.5, label='slip rate')
    line3, = ax3.plot([], [], 'b',  lw=1.5, label='traction')

    ax1.set_xlim(y.min(), y.max());  ax1.set_ylim(-0.1, slip_max * 1.1)
    ax2.set_xlim(y.min(), y.max());  ax2.set_ylim(-0.05, sliprate_max * 1.1)
    ax3.set_xlim(y.min(), y.max());  ax3.set_ylim(trac_min, trac_max)

    ax1.set_ylabel('Slip [m]');      ax1.legend(loc='upper right')
    ax2.set_ylabel('Slip rate [m/s]'); ax2.legend(loc='upper right')
    ax3.set_ylabel('Traction [MPa]'); ax3.legend(loc='upper right')
    ax3.set_xlabel('fault position [km]')
    sup = fig.suptitle('', y=1.005)

    def _update(idx):
        it = frames[idx]
        sr = np.sqrt(FO[:, it, 0]**2 + FO[:, it, 1]**2)
        line1.set_data(y, FO[:, it, 4])
        line2.set_data(y, sr)
        line3.set_data(y, FO[:, it, 3])
        sup.set_text(
            f"On-fault evolution  |  fric={meta['fric_law']}  |  "
            f"t = {time[it]:.2f} s")
        return line1, line2, line3, sup

    ani = animation.FuncAnimation(fig, _update, frames=len(frames),
                                  blit=False, interval=50)
    _save_anim(ani, os.path.join(out_dir, f'{tag}_fault_evolution.mp4'), fps=20)
    plt.close(fig)


# ── 3. Final-time fault profiles ────────────────────────────────────────────

def plot_fault_final(data, meta, out_dir, tag):
    FO   = data['FaultOutput']
    time = data['time']
    y    = data['y_fault']

    slip     = FO[:, -1, 4]
    sliprate = np.sqrt(FO[:, -1, 0]**2 + FO[:, -1, 1]**2)
    traction = FO[:, -1, 3]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    axes[0].plot(y, slip,     'g',  lw=2);  axes[0].set_ylabel('Slip [m]')
    axes[1].plot(y, sliprate, 'r',  lw=2);  axes[1].set_ylabel('Slip rate [m/s]')
    axes[2].plot(y, traction, 'b',  lw=2);  axes[2].set_ylabel('Traction [MPa]')
    axes[2].set_xlabel('fault position [km]')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Final snapshot  (t = {float(time[-1]):.2f} s)  |  "
        f"fric_law = {meta['fric_law']}  |  run_id = {meta['run_id']}")
    fig.tight_layout()
    path = os.path.join(out_dir, f'{tag}_fault_final.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── 4. Space-time slip rate ──────────────────────────────────────────────────

def plot_sliprate_spacetime(data, meta, out_dir, tag):
    FO   = data['FaultOutput']   # (ny, nt, 6)
    time = data['time']
    y    = data['y_fault']

    VT = np.sqrt(FO[:, :, 0]**2 + FO[:, :, 1]**2)   # (ny, nt)
    vmax = min(VT.max(), 10.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(VT, aspect='auto',
                   extent=[float(time[0]), float(time[-1]),
                            float(y[-1]),  float(y[0])],
                   cmap='viridis', vmin=0, vmax=vmax,
                   interpolation='bilinear')
    fig.colorbar(im, ax=ax, label='slip rate [m/s]')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('fault position [km]')
    ax.set_title(
        f"Space-time slip rate  |  fric_law = {meta['fric_law']}  |  "
        f"run_id = {meta['run_id']}")
    fig.tight_layout()
    path = os.path.join(out_dir, f'{tag}_sliprate_spacetime.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── 5. Friction parameter profiles ──────────────────────────────────────────

def plot_friction_params(data, meta, out_dir, tag):
    fp = data['friction_parameters']   # (12, ny)
    y  = data['y_fault']

    labels = ['alpha_l', 'alpha_r', 'Tau_0', 'L0', 'f0',
              'a', 'b', 'V0', 'sigma_n', 'alp_s', 'alp_d', 'D_c']

    # skip rows that are spatially uniform and effectively dummy (alpha)
    skip = {0, 1}
    rows = [i for i in range(12) if i not in skip]

    ncols = 2
    nrows = (len(rows) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.5 * nrows), sharex=True)
    axes = axes.flatten()

    for k, i in enumerate(rows):
        axes[k].plot(y, fp[i], lw=1.5)
        axes[k].set_ylabel(labels[i])
        axes[k].grid(True, alpha=0.3)
        if k >= len(rows) - ncols:
            axes[k].set_xlabel('fault position [km]')

    for k in range(len(rows), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Friction parameters along fault  |  fric_law = {meta['fric_law']}  |  "
        f"run_id = {meta['run_id']}")
    fig.tight_layout()
    path = os.path.join(out_dir, f'{tag}_friction_params.png')
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Create PNG and MP4 plots from a 2D rupture .npz file')
    parser.add_argument('npz', help='Path to .npz output file')
    parser.add_argument('--no-anim', action='store_true',
                        help='Skip animations (faster, PNG only)')
    parser.add_argument('--frames', type=int, default=150,
                        help='Max animation frames [default: 150]')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save plots [default: same as .npz]')
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: file not found: {args.npz}")
        sys.exit(1)

    data = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(data['metadata']))

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)
    run_id  = meta.get('run_id', 'unknown')
    fric    = meta.get('fric_law', 'XX')
    tag     = f"{fric}_{run_id}"

    print(f"run_id   : {run_id}")
    print(f"fric_law : {fric}   mode: {meta.get('mode','?')}")
    print(f"output   : {out_dir}/\n")

    # static plots — always
    plot_fault_final(data, meta, out_dir, tag)
    plot_sliprate_spacetime(data, meta, out_dir, tag)
    if 'friction_parameters' in data.files:
        plot_friction_params(data, meta, out_dir, tag)
    else:
        print("Skipping friction_params plot (key not in .npz — run with updated rupture_2d.py)")

    # animations — optional
    if not args.no_anim:
        print("Generating wavefield animation…")
        plot_wavefield(data, meta, out_dir, tag, max_frames=args.frames)
        print("Generating fault evolution animation…")
        plot_fault_evolution(data, meta, out_dir, tag, max_frames=args.frames)

    print("\nDone.")


if __name__ == '__main__':
    main()
