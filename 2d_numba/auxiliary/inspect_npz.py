#!/usr/bin/env python3
"""
Inspect a 2D rupture .npz output file.

Usage:
    python3 inspect_npz.py output/rupture_2d_SW_<run_id>.npz [--plot]
"""

import argparse
import json
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Inspect 2D rupture .npz output')
    parser.add_argument('npz', help='Path to .npz file')
    parser.add_argument('--plot', action='store_true', help='Save diagnostic plots to PNG')
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    # ── Arrays ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"File : {args.npz}")
    print(f"{'='*60}")
    print(f"\n{'Array':<25} {'Shape':<30} {'Dtype':<10} {'Min':>12} {'Max':>12}")
    print('-' * 92)
    for k in sorted(data.files):
        if k == 'metadata':
            continue
        arr = data[k]
        print(f"{k:<25} {str(arr.shape):<30} {str(arr.dtype):<10} "
              f"{arr.min():>12.4g} {arr.max():>12.4g}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = json.loads(str(data['metadata']))
    print(f"\n{'─'*60}")
    print("Metadata:")
    skip = {'friction_parameters', 'dx', 'dy', 'dt', 'nt', 'nf', 'mu', 'Lambda'}
    for k, v in sorted(meta.items()):
        if k not in skip:
            print(f"  {k:<28} = {v}")

    # ── Summary ───────────────────────────────────────────────────────────────
    FO = data['FaultOutput']          # (ny, nt, 6)
    ny, nt, _ = FO.shape
    time = data['time']
    y    = data['y_fault']

    slip_final    = FO[:, -1, 4]
    sliprate_max  = np.sqrt(FO[:, :, 0]**2 + FO[:, :, 1]**2).max()
    traction_min  = FO[:, :, 3].min()
    traction_max  = FO[:, :, 3].max()

    print(f"\n{'─'*60}")
    print(f"FaultOutput summary  (ny={ny}, nt={nt}):")
    print(f"  max slip at t_end        = {slip_final.max():.4f} m")
    print(f"  max slip rate (all time) = {sliprate_max:.4f} m/s")
    print(f"  shear traction range     = [{traction_min:.2f}, {traction_max:.2f}] MPa")

    DO_l = data['DomainOutput_l']     # (nx, ny, n_snap, nf)
    nx, ny_d, n_snap, nf = DO_l.shape
    print(f"\nDomainOutput summary  (nx={nx}, ny={ny_d}, n_snap={n_snap}, nf={nf}):")
    print(f"  float32 size (L+R)       = "
          f"{2 * DO_l.nbytes / 1e6:.1f} MB")

    fp = data['friction_parameters']  # (12, ny)
    print(f"\nfriction_parameters  shape={fp.shape}:")
    labels = ['alpha_l','alpha_r','Tau_0','L0','f0','a','b','V0',
              'sigma_n','alp_s','alp_d','D_c']
    for i, lbl in enumerate(labels):
        row = fp[i]
        if row.min() == row.max():
            print(f"  [{i:2d}] {lbl:<10} = {row[0]:.6g}  (uniform)")
        else:
            print(f"  [{i:2d}] {lbl:<10}   min={row.min():.6g}  max={row.max():.6g}  (spatially varying)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        out_dir = os.path.dirname(args.npz)
        run_id  = meta.get('run_id', 'unknown')
        fric    = meta.get('fric_law', '')

        # 1. Space-time slip rate
        VT = np.sqrt(FO[:, :, 0]**2 + FO[:, :, 1]**2)   # (ny, nt)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(VT.T, aspect='auto',
                       extent=[y.min(), y.max(), time[-1], time[0]],
                       cmap='viridis', vmin=0, vmax=min(VT.max(), 10))
        fig.colorbar(im, ax=ax, label='slip rate [m/s]')
        ax.set_xlabel('fault position [km]')
        ax.set_ylabel('time [s]')
        ax.set_title(f'On-fault slip rate  |  {fric}  |  run={run_id}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'inspect_sliprate_{run_id}.png'), dpi=150)
        print(f"\nSaved: {os.path.join(out_dir, f'inspect_sliprate_{run_id}.png')}")
        plt.close(fig)

        # 2. Final-time fault profiles
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(y, FO[:, -1, 4]);  axes[0].set_ylabel('slip [m]')
        axes[1].plot(y, np.sqrt(FO[:, -1, 0]**2 + FO[:, -1, 1]**2))
        axes[1].set_ylabel('slip rate [m/s]')
        axes[2].plot(y, FO[:, -1, 3]);  axes[2].set_ylabel('traction [MPa]')
        axes[2].set_xlabel('fault position [km]')
        fig.suptitle(f'Final snapshot  |  {fric}  |  run={run_id}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'inspect_final_{run_id}.png'), dpi=150)
        print(f"Saved: {os.path.join(out_dir, f'inspect_final_{run_id}.png')}")
        plt.close(fig)

        # 3. Mid-domain wavefield snapshot (vy field, field index 1)
        snap_mid = n_snap // 2
        DO_r = data['DomainOutput_r']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        x_l = data['x_l'];  x_r = data['x_r']
        vmax = max(abs(DO_l[:, :, snap_mid, 1]).max(),
                   abs(DO_r[:, :, snap_mid, 1]).max(), 1e-10)
        for ax, D, x, side in zip(axes, [DO_l, DO_r], [x_l, x_r], ['left', 'right']):
            im = ax.imshow(D[:, :, snap_mid, 1].T, aspect='auto',
                           extent=[x.min(), x.max(), y.max(), y.min()],
                           cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.set_title(f'vy  {side}  t={data["snap_times"][snap_mid]:.2f}s')
            ax.set_xlabel('x [km]');  ax.set_ylabel('y [km]')
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'Domain snapshot  |  {fric}  |  run={run_id}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'inspect_domain_{run_id}.png'), dpi=150)
        print(f"Saved: {os.path.join(out_dir, f'inspect_domain_{run_id}.png')}")
        plt.close(fig)

    print()


if __name__ == '__main__':
    main()
