#!/usr/bin/env python3
"""
generate_dataset.py — Generate PINO-FNO training dataset by batch-running the
Numba rupture simulator with Latin Hypercube Sampling of physical parameters.

Usage:
    python3 data_gen/generate_dataset.py --n_train 200 --n_val 50 --n_test 50
    python3 data_gen/generate_dataset.py --n_train 200 --n_val 50 --n_test 50 --seed 42

Output:
    data/train/sample_{i:05d}.npz
    data/val/sample_{i:05d}.npz
    data/test/sample_{i:05d}.npz

Each .npz contains:
    params       : float32 (N_params,)       — normalised input parameters
    params_raw   : float32 (N_params,)       — raw physical parameter values
    v_l          : float32 (nx_out, nt_out)  — left domain velocity
    s_l          : float32 (nx_out, nt_out)  — left domain stress
    v_r          : float32 (nx_out, nt_out)  — right domain velocity
    s_r          : float32 (nx_out, nt_out)  — right domain stress
    slip         : float32 (nt_out,)         — fault slip time series
    sliprate     : float32 (nt_out,)         — fault slip rate
    traction     : float32 (nt_out,)         — fault traction
    x            : float32 (nx_out,)         — spatial coordinates (km)
    t            : float32 (nt_out,)         — time coordinates (s)
    param_names  : list[str]                 — parameter names (stored as JSON)
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Path to simulator
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from rupture_1d import (parse_infile, build_params, validate,
                        init_fields, build_friction_parameters, make_run_id)

import numba
from src.kernels import rk4_step, FRIC_SW
import timeit

# ---------------------------------------------------------------------------
# Parameter space definition
# POC: vary 4 SW parameters + initial background stress
# ---------------------------------------------------------------------------
PARAM_NAMES = ['Tau_0', 'alp_s', 'alp_d', 'D_c']

PARAM_BOUNDS = {
    'Tau_0': (78.0,  88.0),   # MPa  — must exceed peak strength to nucleate
    'alp_s': (0.62,  0.74),   # static friction coeff
    'alp_d': (0.45,  0.55),   # dynamic friction coeff  (must stay < alp_s)
    'D_c'  : (0.2,   0.8),    # m — critical slip distance
}

# Fixed baseline .in file (provides all other parameters)
BASELINE_IN = os.path.join(_ROOT, 'input', 'rupture_1d_SW.in')

# Output grid sub-sampling
NX_OUT = 64    # spatial points per domain (downsampled from nx=501)
NT_OUT = 64    # time points (downsampled from nt~577)


# ---------------------------------------------------------------------------
# Simulator runner (no-file-IO version)
# ---------------------------------------------------------------------------

def _run_sim_inmem(p):
    """Run simulator, return downsampled (v_l, s_l, v_r, s_r, slip, sliprate, traction)."""
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

    # Storage: every iplot-th step
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
        ds4, dp4 = rk4_step(v_l, s_l, v_r, s_r, slip, psi,
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
    # Downsample spatially and temporally to (NX_OUT, NT_OUT)
    xi = np.linspace(0, nx-1, NX_OUT, dtype=int)
    ti = np.linspace(0, n_stored_actual-1, NT_OUT, dtype=int)

    out = dict(
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
    return out


# ---------------------------------------------------------------------------
# LHS sampling with nucleation validity check
# ---------------------------------------------------------------------------

def _sample_params(n_samples, seed=0):
    """LHS sample N valid parameter sets.  Retries until nucleation condition holds."""
    rng = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    samples = rng.random(n=n_samples * 3)   # oversample to account for rejections

    results = []
    for row in samples:
        raw = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            raw[name] = lo + row[i] * (hi - lo)

        # Validity: alp_d < alp_s  (velocity-weakening)
        if raw['alp_d'] >= raw['alp_s']:
            continue

        # Nucleation check: Tau_0 > tau_peak = alp_s * sigma_n
        sigma_n = 120.0   # fixed
        tau_peak = raw['alp_s'] * sigma_n
        if raw['Tau_0'] <= tau_peak:
            continue

        results.append(raw)
        if len(results) >= n_samples:
            break

    if len(results) < n_samples:
        raise RuntimeError(
            f"Could only generate {len(results)}/{n_samples} valid samples. "
            "Widen parameter bounds or increase oversample factor.")

    return results[:n_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_split(split_name, param_list, out_dir, p_base):
    """Simulate each parameter set and save to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(param_list)
    print(f"\n--- {split_name}: {n} samples → {out_dir} ---")

    for i, raw_params in enumerate(param_list):
        # Build params by overriding baseline
        p = dict(p_base)
        p.update(raw_params)
        p['nx']  = int(p['nx'])
        p['dx']  = float(p['L']) / (p['nx'] - 1)
        if 'dt' not in p or p.get('dt') is None:
            p['dt'] = (float(p['cfl']) / float(p['cs'])) * p['dx']
        p['dt']  = float(p['dt'])
        p['nt']  = int(round(float(p['tend']) / p['dt']))

        t0 = timeit.default_timer()
        sim = _run_sim_inmem(p)
        wall = timeit.default_timer() - t0

        # Parameter vector (raw float values in PARAM_NAMES order)
        params_raw = np.array([raw_params[k] for k in PARAM_NAMES], dtype=np.float32)

        out_file = os.path.join(out_dir, f'sample_{i:05d}.npz')
        np.savez_compressed(
            out_file,
            params_raw   = params_raw,
            param_names  = json.dumps(PARAM_NAMES),
            param_bounds = json.dumps(PARAM_BOUNDS),
            v_l          = sim['v_l'],
            s_l          = sim['s_l'],
            v_r          = sim['v_r'],
            s_r          = sim['s_r'],
            slip         = sim['slip'],
            sliprate     = sim['sliprate'],
            traction     = sim['traction'],
            time         = sim['time'],
            x_l          = sim['x_l'],
            x_r          = sim['x_r'],
        )
        print(f"  [{split_name}] {i+1:>4}/{n}  "
              f"Tau_0={raw_params['Tau_0']:.2f}  alp_s={raw_params['alp_s']:.3f}  "
              f"alp_d={raw_params['alp_d']:.3f}  D_c={raw_params['D_c']:.2f}  "
              f"→ {wall:.2f}s")

    print(f"  {split_name} done.")


def main():
    parser = argparse.ArgumentParser(description='Generate PINO-FNO rupture dataset')
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_val',   type=int, default=50)
    parser.add_argument('--n_test',  type=int, default=50)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--out_dir', default=os.path.join(_ROOT, 'data'))
    args = parser.parse_args()

    # Load fixed baseline parameters
    p_base = build_params(parse_infile(BASELINE_IN))

    # JIT warmup (single call so compilation doesn't count in generation time)
    print("JIT warmup ...")
    numba.set_num_threads(1)
    _ = _run_sim_inmem(p_base)
    print("  done.\n")

    n_total = args.n_train + args.n_val + args.n_test
    print(f"Sampling {n_total} parameter sets (LHS, seed={args.seed}) ...")
    all_params = _sample_params(n_total, seed=args.seed)

    train_params = all_params[:args.n_train]
    val_params   = all_params[args.n_train:args.n_train + args.n_val]
    test_params  = all_params[args.n_train + args.n_val:]

    generate_split('train', train_params, os.path.join(args.out_dir, 'train'), p_base)
    generate_split('val',   val_params,   os.path.join(args.out_dir, 'val'),   p_base)
    generate_split('test',  test_params,  os.path.join(args.out_dir, 'test'),  p_base)

    print(f"\nDataset complete: {args.n_train} train / {args.n_val} val / {args.n_test} test")
    print(f"Output: {args.out_dir}")


if __name__ == '__main__':
    main()
