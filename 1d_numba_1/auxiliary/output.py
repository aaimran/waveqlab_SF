#!/usr/bin/env python3
"""
Post-processing: generate plots from a saved .npz rupture simulation output.

Usage:
    python3 auxiliary/output.py output/rupture_SW_abc12345.npz
    python3 auxiliary/output.py output/rupture_SW_abc12345.npz --stride 2
    python3 auxiliary/output.py output/rupture_SW_abc12345.npz --plotdir /some/other/dir

Saves the following into plots/ (or --plotdir):
    {prefix}_{run_id}_timeseries.png       slip / slip-rate / traction vs time
    {prefix}_{run_id}_final_snapshot.png   velocity + stress across domain at t=tend
    {prefix}_{run_id}_domain_animation.mp4 animated domain fields (falls back to .gif)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation

# auxiliary/ lives one level below the project root
_AUX  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_AUX)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    metadata = json.loads(str(d['metadata'])) if 'metadata' in d else {}
    params   = metadata if metadata else {}

    # Tau_0 priority: explicit key > metadata > traction[0]
    if 'Tau_0' in d:
        Tau_0 = float(d['Tau_0'])
    elif 'Tau_0' in params:
        Tau_0 = float(params['Tau_0'])
    else:
        Tau_0 = float(d['traction'][0])

    # run_id from metadata or derive from filename
    stem   = os.path.splitext(os.path.basename(path))[0]   # e.g. rupture_SW_abc12345
    run_id = params.get('run_id', stem.split('_')[-1])
    prefix = params.get('output_prefix', '_'.join(stem.split('_')[:-1]))

    return d, params, Tau_0, run_id, prefix


# ---------------------------------------------------------------------------
# 1. Time-series PNG
# ---------------------------------------------------------------------------

def plot_timeseries(d, params, Tau_0, outpath):
    time     = d['time']
    slip     = d['slip']
    sliprate = d['sliprate']
    traction = d['traction']
    tend     = float(params.get('tend', time[-1]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"On-fault time series  |  fric_law = {params.get('fric_law','?')}  |  "
        f"run_id = {params.get('run_id','?')}",
        fontsize=13)

    axes[0].plot(time, slip[1:], '-g', lw=2)
    axes[0].set_ylabel('slip [m]', fontsize=13)
    axes[0].set_ylim([0, max(32.0, float(slip.max()) * 1.1)])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, sliprate, '-r', lw=2)
    axes[1].set_ylabel('slip rate [m/s]', fontsize=13)
    axes[1].set_ylim([-0.2, max(5.0, float(sliprate.max()) * 1.1)])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, traction, '-b', lw=2)
    axes[2].set_ylabel('traction [MPa]', fontsize=13)
    axes[2].set_xlabel('t [s]', fontsize=13)
    tmin = float(traction.min())
    tmax = float(traction.max())
    axes[2].set_ylim([min(50.0, tmin * 0.97), max(90.0, tmax * 1.03)])
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim([0, tend])

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# 2. Final snapshot PNG
# ---------------------------------------------------------------------------

def plot_final_snapshot(d, params, Tau_0, outpath):
    y_l = d['y_l']
    y_r = d['y_r']
    DL  = d['DomainOutput_l']
    DR  = d['DomainOutput_r']
    L   = float(params.get('L', y_l[-1]))
    tend = params.get('tend', '?')
    nt_stored = DL.shape[1]
    it = nt_stored - 2   # last filled time step

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        f"Final snapshot  (t = {tend} s)  |  fric_law = {params.get('fric_law','?')}  |  "
        f"run_id = {params.get('run_id','?')}",
        fontsize=13)

    ax1.plot(y_r, DR[:, it, 0], 'r', lw=2, label='right domain')
    ax1.plot(y_l, DL[:, it, 0], 'b', lw=2, label='left domain')
    ax1.set_ylabel('velocity [m/s]', fontsize=13)
    ax1.set_xlabel('x [km]', fontsize=13)
    ax1.set_xlim([0, 2 * L])
    ax1.set_ylim([-2.5, 2.5])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(y_r, DR[:, it, 1] + Tau_0, 'r', lw=2, label='right domain')
    ax2.plot(y_l, DL[:, it, 1] + Tau_0, 'b', lw=2, label='left domain')
    ax2.set_ylabel('stress [MPa]', fontsize=13)
    ax2.set_xlabel('x [km]', fontsize=13)
    ax2.set_xlim([0, 2 * L])
    ax2.set_ylim([50, 90])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# 3. Domain animation MP4 / GIF
# ---------------------------------------------------------------------------

def make_animation(d, params, Tau_0, outpath, stride=5):
    y_l  = d['y_l']
    y_r  = d['y_r']
    DL   = d['DomainOutput_l']
    DR   = d['DomainOutput_r']
    time = d['time']
    L    = float(params.get('L', y_l[-1]))

    n_filled = len(time)
    frames   = list(range(0, n_filled, stride))
    fps      = min(30, max(1, int(round(len(frames) / max(1.0, float(params.get('tend', 5.0)))))))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        f"Domain animation  |  fric_law = {params.get('fric_law','?')}  |  "
        f"run_id = {params.get('run_id','?')}",
        fontsize=13)

    line_vr, = ax1.plot([], [], 'r', lw=2, label='right')
    line_vl, = ax1.plot([], [], 'b', lw=2, label='left')
    ax1.set_xlim([0, 2 * L]); ax1.set_ylim([-2.5, 2.5])
    ax1.set_ylabel('velocity [m/s]', fontsize=13)
    ax1.set_xlabel('x [km]', fontsize=13)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    time_title = ax1.set_title('')

    line_sr, = ax2.plot([], [], 'r', lw=2)
    line_sl, = ax2.plot([], [], 'b', lw=2)
    ax2.set_xlim([0, 2 * L]); ax2.set_ylim([50, 90])
    ax2.set_ylabel('stress [MPa]', fontsize=13)
    ax2.set_xlabel('x [km]', fontsize=13)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    def init():
        for l in (line_vr, line_vl, line_sr, line_sl):
            l.set_data([], [])
        time_title.set_text('')
        return line_vr, line_vl, line_sr, line_sl, time_title

    def update(it):
        line_vr.set_data(y_r, DR[:, it, 0])
        line_vl.set_data(y_l, DL[:, it, 0])
        line_sr.set_data(y_r, DR[:, it, 1] + Tau_0)
        line_sl.set_data(y_l, DL[:, it, 1] + Tau_0)
        t_val = float(time[it]) if it < len(time) else float(time[-1])
        time_title.set_text(f't = {t_val:.3f} s')
        return line_vr, line_vl, line_sr, line_sl, time_title

    anim = mpl_animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=int(1000 / fps), blit=False)

    saved = False

    # Try system ffmpeg via matplotlib writer
    if mpl_animation.FFMpegWriter.isAvailable():
        try:
            writer = mpl_animation.FFMpegWriter(
                fps=fps, bitrate=2000,
                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            anim.save(outpath, writer=writer)
            print(f"Saved: {outpath}  (FFMpegWriter, {fps} fps)")
            saved = True
        except Exception as e:
            print(f"  FFMpegWriter failed: {e}")

    # Fall back: imageio-ffmpeg (bundled binary)
    if not saved:
        try:
            import imageio.v3 as iio
            frames_rgb = []
            for it in frames:
                update(it)
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                frames_rgb.append(buf.reshape(h, w, 4)[:, :, :3].copy())
            iio.imwrite(outpath, frames_rgb, fps=fps, codec='libx264',
                        output_params=['-pix_fmt', 'yuv420p'])
            print(f"Saved: {outpath}  (imageio-ffmpeg, {fps} fps)")
            saved = True
        except Exception as e:
            print(f"  imageio-ffmpeg failed: {e}")

    # Last resort: Pillow GIF
    if not saved:
        gif_path = outpath.replace('.mp4', '.gif')
        try:
            writer = mpl_animation.PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            print(f"  mp4 unavailable; saved GIF: {gif_path}")
        except Exception as e:
            print(f"  Could not save animation: {e}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate PNG + MP4 plots from a rupture_1d .npz output file')
    parser.add_argument('npzfile',
                        help='path to .npz file (e.g. output/rupture_SW_abc12345.npz)')
    parser.add_argument('--stride', type=int, default=5,
                        help='animate every N-th stored time step (default: 5)')
    parser.add_argument('--plotdir', default=None,
                        help='directory to write plots into (default: plots/ next to this script)')
    args = parser.parse_args()

    if not os.path.isfile(args.npzfile):
        print(f"ERROR: File not found: {args.npzfile}")
        sys.exit(1)

    plot_dir = args.plotdir if args.plotdir else os.path.join(_ROOT, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    d, params, Tau_0, run_id, prefix = load_npz(args.npzfile)
    base = os.path.join(plot_dir, f"{prefix}_{run_id}")

    nt = len(d['time'])
    nx = d['DomainOutput_l'].shape[0]
    print(f"Loaded  : {args.npzfile}")
    print(f"nx={nx}  nt={nt}  fric_law={params.get('fric_law','?')}  "
          f"Tau_0={Tau_0:.4f} MPa  run_id={run_id}")
    print(f"Plots → : {plot_dir}\n")

    plot_timeseries(d, params, Tau_0,
                    outpath=base + '_timeseries.png')
    plot_final_snapshot(d, params, Tau_0,
                        outpath=base + '_final_snapshot.png')
    make_animation(d, params, Tau_0,
                   outpath=base + '_domain_animation.mp4',
                   stride=args.stride)


if __name__ == '__main__':
    main()
