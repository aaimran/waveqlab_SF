"""
plot_withers.py  —  Withers et al. (2015) comparison figures

Reproduces the structure of Figures 3-6 from Withers et al. (2015) BSSA:
  Fig 3 → test-1 : elastic half-space
  Fig 4 → test-2 : constant-Q  half-space  (γ=0, Q_S=50)
  Fig 5 → test-3 : power-law-Q half-space  (γ=0.6, Q_S_0=50)
  Fig 6 → test-4 : layered model           (γ=0.6, Q_S=[20, 210])

Usage (from project root):
  python python/plot_withers.py [--outdir figures/] [--station-dist 5,10,15,20,25]

The "b" variants (absorbing outer BC) are used so no spurious reflections
pollute the waveforms before the fault-origin wave arrives.
"""

import argparse
import glob
import json
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def find_output(prefix: str, output_dir: str = "output") -> str:
    """Return path to newest correctly-formatted output file for *prefix*."""
    pattern = os.path.join(output_dir, f"{prefix}_*.npz")
    candidates = [f for f in glob.glob(pattern) if "timing" not in f]
    if not candidates:
        raise FileNotFoundError(f"No output file found for prefix '{prefix}'")
    # Prefer nx=751 (most recent params) over older nx=501 files
    best = None
    for f in candidates:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["metadata"]))
        if meta.get("nx", 0) == 751:
            if best is None:
                best = f
            else:
                # Take the more recently modified one
                if os.path.getmtime(f) > os.path.getmtime(best):
                    best = f
    if best is None:
        best = max(candidates, key=os.path.getmtime)
    return best


def load_velocity(fname: str):
    """Return (time, y_l, vel_l) arrays from an output .npz file."""
    d = np.load(fname, allow_pickle=True)
    time = d["time"]           # (nt,)
    y_l  = d["y_l"]            # (nx,)  — spatial coords of left domain [km]
    vel  = d["DomainOutput_l"][:, :, 0]  # (nx, nt)  — particle velocity [m/s]
    meta = json.loads(str(d["metadata"]))
    return time, y_l, vel, meta


def extract_trace(time, y_l, vel, dist_from_fault_km: float):
    """
    Return (time, trace) for a station at *dist_from_fault_km* km from the fault.

    The left domain has y_l[0]=0 (outer boundary) and y_l[-1]=L (fault).
    Distance from fault = L - y.
    """
    L = y_l[-1]
    target_x = L - dist_from_fault_km
    idx = int(np.round(target_x / (y_l[1] - y_l[0])))
    idx = np.clip(idx, 0, len(y_l) - 1)
    return time, vel[idx, :]


# ─── plotting ─────────────────────────────────────────────────────────────────

CASE_INFO = {
    "test-1b": dict(label="Elastic",          color="black",   fig="3"),
    "test-2b": dict(label="Const-Q  (γ=0)",   color="C0",      fig="4"),
    "test-3b": dict(label="Power-law Q (γ=0.6)", color="C1",   fig="5"),
    "test-4b": dict(label="Layered (γ=0.6)",  color="C2",      fig="6"),
}


def plot_stacked_traces(ax, time, y_l, vel, dists, scale=1.0,
                        color="black", label=None, t_max=None):
    """Plot stacked normalised velocity traces at multiple source distances."""
    first = True
    for k, dist in enumerate(dists):
        t, tr = extract_trace(time, y_l, vel, dist)
        if t_max is not None:
            mask = t <= t_max
            t, tr = t[mask], tr[mask]
        peak = np.max(np.abs(tr))
        if peak == 0:
            continue
        tr_norm = tr / peak
        offset = k * scale
        ax.plot(t, tr_norm + offset,
                color=color, lw=0.8,
                label=label if first else "_nolegend_")
        ax.text(-0.15, offset, f"{dist:.0f} km", ha="right", va="center",
                fontsize=7, transform=ax.get_yaxis_transform())
        first = False


def make_stacked_comparison(prefixes, output_dir, dists, t_max, outpath):
    """
    One figure with len(prefixes) columns (cases) and one row.
    Each column: stacked velocity traces at multiple station distances.
    """
    n = len(prefixes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, prefix in zip(axes, prefixes):
        info = CASE_INFO[prefix]
        fname = find_output(prefix, output_dir)
        time, y_l, vel, meta = load_velocity(fname)

        plot_stacked_traces(ax, time, y_l, vel, dists,
                            scale=2.0, color=info["color"], t_max=t_max)

        ax.set_xlim(0, t_max)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_title(f"Fig {info['fig']}: {info['label']}\n"
                     f"nx={meta['nx']}  tend={meta['tend']} s",
                     fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(-1.5, len(dists) * 2.0)
        ax.grid(axis="x", alpha=0.3)
        # annotate Q info
        if meta.get("response") == "anelastic":
            note = (f"c={meta['c']}  γ={meta['weight_exp']}\n"
                    f"Qs_file={bool(meta.get('Qs_inv_file'))}")
        else:
            note = "No attenuation"
        ax.annotate(note, xy=(0.98, 0.02), xycoords="axes fraction",
                    fontsize=7, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.7))

    fig.suptitle("Withers et al. (2015) — stacked velocity seismograms\n"
                 "(normalised per trace, absorbing outer BC)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved → {outpath}")
    plt.close(fig)


def make_overlay_comparison(prefixes, output_dir, dist_km, t_max, outpath):
    """
    Single panel: overlay elastic / const-Q / power-law-Q at one station.
    Shows amplitude decay and waveform distortion due to attenuation.
    """
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(8, 6))

    for prefix in prefixes:
        info = CASE_INFO[prefix]
        fname = find_output(prefix, output_dir)
        time, y_l, vel, meta = load_velocity(fname)
        t, tr = extract_trace(time, y_l, vel, dist_km)
        mask = t <= t_max
        t, tr = t[mask], tr[mask]

        # ── time-domain ──
        ax_t.plot(t, tr, label=info["label"], color=info["color"], lw=1.0)

        # ── Fourier amplitude ──
        dt = t[1] - t[0]
        N  = len(tr)
        freq = np.fft.rfftfreq(N, d=dt)
        amp  = np.abs(np.fft.rfft(tr)) / N
        fmask = freq > 0
        ax_f.semilogy(freq[fmask], amp[fmask],
                      label=info["label"], color=info["color"], lw=1.0)

    ax_t.set_xlabel("Time (s)", fontsize=9)
    ax_t.set_ylabel("Velocity (m/s)", fontsize=9)
    ax_t.set_xlim(0, t_max)
    ax_t.set_title(f"Velocity at {dist_km} km from fault", fontsize=9)
    ax_t.legend(fontsize=8)
    ax_t.grid(alpha=0.3)

    ax_f.set_xlabel("Frequency (Hz)", fontsize=9)
    ax_f.set_ylabel("Amplitude spectrum", fontsize=9)
    ax_f.set_xlim(0, 5)
    ax_f.set_title("Fourier amplitude spectrum", fontsize=9)
    ax_f.legend(fontsize=8)
    ax_f.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Withers et al. (2015) — elastic vs constant-Q vs power-law Q\n"
        f"Station at {dist_km} km from fault, absorbing outer BC",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved → {outpath}")
    plt.close(fig)


def make_layered_comparison(prefixes, output_dir, dist_km, t_max, outpath):
    """
    Two panels: uniform power-law Q vs layered model (Fig 6 analog).
    Time-domain on top, Fourier spectra on bottom.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    for col, prefix in enumerate(prefixes):
        info = CASE_INFO[prefix]
        fname = find_output(prefix, output_dir)
        time, y_l, vel, meta = load_velocity(fname)
        t, tr = extract_trace(time, y_l, vel, dist_km)
        mask = t <= t_max
        t, tr = t[mask], tr[mask]

        ax_t = axes[0, col]
        ax_f = axes[1, col]

        ax_t.plot(t, tr, color=info["color"], lw=1.0)
        ax_t.set_title(f"{info['label']}  (Fig {info['fig']})", fontsize=9)
        ax_t.set_xlabel("Time (s)", fontsize=8)
        ax_t.set_ylabel("Velocity (m/s)", fontsize=8)
        ax_t.set_xlim(0, t_max)
        ax_t.grid(alpha=0.3)

        dt = t[1] - t[0]
        N  = len(tr)
        freq = np.fft.rfftfreq(N, d=dt)
        amp  = np.abs(np.fft.rfft(tr)) / N
        fmask = (freq > 0) & (freq < 8)
        ax_f.semilogy(freq[fmask], amp[fmask], color=info["color"], lw=1.0)
        ax_f.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_f.set_ylabel("Amplitude", fontsize=8)
        ax_f.set_title("Fourier amplitude", fontsize=9)
        ax_f.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Withers et al. (2015) Fig 5 vs Fig 6 — effect of layering\n"
        f"Station at {dist_km} km from fault, absorbing outer BC",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved → {outpath}")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",  default="figures",
                        help="Directory for output figures")
    parser.add_argument("--output-dir", default="output",
                        help="Directory containing .npz simulation outputs")
    parser.add_argument("--station-dists", default="5,10,15,20,25",
                        help="Comma-separated station distances from fault (km)")
    parser.add_argument("--overlay-dist", type=float, default=15.0,
                        help="Station distance for overlay/spectrum plot (km)")
    parser.add_argument("--ext", default="pdf",
                        help="Output figure extension: pdf or png (default: pdf)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dists  = [float(x) for x in args.station_dists.split(",")]
    ext    = args.ext.lstrip(".")

    # ── Figure A: stacked traces for each of test-1..4 (Figs 3-6 analog)
    make_stacked_comparison(
        ["test-1b", "test-2b", "test-3b", "test-4b"],
        output_dir=args.output_dir,
        dists=dists,
        t_max=12.0,
        outpath=os.path.join(args.outdir, f"withers_stacked.{ext}"),
    )

    # ── Figure B: overlay elastic / const-Q / power-law Q at one station
    make_overlay_comparison(
        ["test-1b", "test-2b", "test-3b"],
        output_dir=args.output_dir,
        dist_km=args.overlay_dist,
        t_max=8.0,
        outpath=os.path.join(args.outdir, f"withers_overlay.{ext}"),
    )

    # ── Figure C: power-law Q uniform vs layered (Fig 5 vs 6)
    make_layered_comparison(
        ["test-3b", "test-4b"],
        output_dir=args.output_dir,
        dist_km=args.overlay_dist,
        t_max=12.0,
        outpath=os.path.join(args.outdir, f"withers_layered.{ext}"),
    )

    print("All figures saved.")


if __name__ == "__main__":
    main()
