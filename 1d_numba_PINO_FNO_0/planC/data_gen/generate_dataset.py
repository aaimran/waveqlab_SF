#!/usr/bin/env python3
"""
planC/data_gen/generate_dataset.py
===================================
Plan-C data generator for SW rupture.

Conditions:
  bc_both0 : r0_l=0, r1_r=0 (both absorbing)
  bc_both1 : r0_l=1, r1_r=1 (both free-surface)

Resolutions stored per sample:
  _r80  : nx=376  → dx=80 m   (30000/375)
  _r40  : nx=751  → dx=40 m
  _r20  : nx=1501 → dx=20 m

All fields are downsampled from the full SBP-SAT native resolution (nx=3001,
dx=10 m) so every resolution sees the same physics.  The GT for
super-resolution evaluation is the raw 10 m run (NOT stored in training .npz).

Extended SW parameter sweep (7 independent params  +  cs, rho):
  Tau_0    [78, 88]   MPa
  alp_s    [0.62, 0.74]
  alp_d    [0.40, 0.56]   (constrained < alp_s, sampled then filtered)
  D_c      [0.15, 0.90]   m
  sigma_n  [100, 140]     MPa  (varies — Tau_0 nucleation check uses this)
  cs       [3.0, 4.0]     km/s
  rho      [2.4, 3.0]     g/cm³

Usage:
    cd /scratch/aimran/FNO/waveqlab_SF/1d_numba_PINO_FNO_0
    source /work/aimran/wql1d/env.sh

    python planC/data_gen/generate_dataset.py \\
        --bc both0 \\
        --n_train 400 --n_val 100 --n_test 100 \\
        --n_workers 48 \\
        --data_dir planC/data/bc_both0

    python planC/data_gen/generate_dataset.py \\
        --bc both1 \\
        --n_train 400 --n_val 100 --n_test 100 \\
        --n_workers 48 \\
        --data_dir planC/data/bc_both1
"""

import argparse
import json
import os
import sys
import timeit
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Locate simulator root
_HERE  = os.path.dirname(os.path.abspath(__file__))
_PLANC = os.path.dirname(_HERE)
_ROOT  = os.path.dirname(_PLANC)          # 1d_numba_PINO_FNO_0/
sys.path.insert(0, _ROOT)

from rupture_1d import (parse_infile, build_params,
                        init_fields, build_friction_parameters)
import numba
from src.kernels import rk4_step, FRIC_SW

# ---------------------------------------------------------------------------
# Resolutions: (label, nx_out)
# Native simulation always at NX_NATIVE=3001 (dx=10 m) then downsampled.
# ---------------------------------------------------------------------------
NX_NATIVE = 3001                        # 30 km / 10 m  = 3000 intervals
NT_OUT    = 128                         # time-axis output points (all resolutions)

RESOLUTIONS = {
    'r80': 376,    # dx = 30000/(376-1) = 80.0 m
    'r40': 751,    # dx = 40.0 m
    'r20': 1501,   # dx = 20.0 m
}

# ---------------------------------------------------------------------------
# BC presets
# ---------------------------------------------------------------------------
BC_PRESETS = {
    'both0': dict(r0_l=0, r1_l=1, r0_r=1, r1_r=0),
    'both1': dict(r0_l=1, r1_l=1, r0_r=1, r1_r=1),
}

# ---------------------------------------------------------------------------
# Extended SW parameter sweep
# ---------------------------------------------------------------------------
PARAM_NAMES = ['Tau_0', 'alp_s', 'alp_d', 'D_c', 'sigma_n', 'cs', 'rho']

PARAM_BOUNDS = {
    'Tau_0'  : (78.0,  88.0),
    'alp_s'  : (0.62,  0.74),
    'alp_d'  : (0.40,  0.56),   # post-filter: must be < alp_s
    'D_c'    : (0.15,  0.90),
    'sigma_n': (100.0, 140.0),
    'cs'     : (3.0,   4.0),    # km/s
    'rho'    : (2.4,   3.0),    # g/cm³
}

BASELINE_IN = os.path.join(_ROOT, 'planC', 'input', 'sw_both0.in')

# ---------------------------------------------------------------------------
# Simulator helpers
# ---------------------------------------------------------------------------

def _downsample(arr_full, n_out):
    """Nearest-neighbour downsample 1-D or 2-D array along first axis to n_out."""
    n_in = arr_full.shape[0]
    idx  = np.round(np.linspace(0, n_in - 1, n_out)).astype(int)
    return arr_full[idx]


def _run_sim_inmem(p_dict):
    """Run full-resolution SBP-SAT simulation; return snapshots.

    Note: L, dx, dt are in km / (km/s) → seconds.  All distances in km.
    """
    nx    = int(p_dict['nx'])
    nt    = int(p_dict['nt'])
    dx    = float(p_dict['dx'])    # km
    dt    = float(p_dict['dt'])    # s
    order = int(p_dict['order'])
    r0_l  = float(p_dict['r0_l'])
    r1_r  = float(p_dict['r1_r'])
    rho   = float(p_dict['rho'])   # g/cm³  → scale rho→kg/m³ if needed by kernels
    cs    = float(p_dict['cs'])    # km/s
    mu    = rho * cs ** 2          # GPa-like (consistent units)
    Tau_0 = float(p_dict['Tau_0'])
    iplot = int(p_dict['iplot'])

    y_l, y_r, v_l, s_l, v_r, s_r, slip, psi = init_fields(p_dict)
    fp = build_friction_parameters(p_dict)

    n_stored = nt // iplot + 1
    VL = np.zeros((nx, n_stored), dtype=np.float64)
    SL = np.zeros((nx, n_stored), dtype=np.float64)
    VR = np.zeros((nx, n_stored), dtype=np.float64)
    SR = np.zeros((nx, n_stored), dtype=np.float64)
    SLIP  = np.zeros(n_stored, dtype=np.float32)
    SLIPR = np.zeros(n_stored, dtype=np.float32)
    TRACT = np.zeros(n_stored, dtype=np.float32)
    TIME  = np.zeros(n_stored, dtype=np.float32)

    idx = 0
    for n in range(nt):
        rk4_step(v_l, s_l, v_r, s_r, slip, psi,
                 rho, mu, nx, dx, order, r0_l, r1_r, dt, fp)
        if n % iplot == 0 and idx < n_stored:
            VL[:, idx] = v_l
            SL[:, idx] = s_l
            VR[:, idx] = v_r
            SR[:, idx] = s_r
            SLIP[idx]  = float(slip[0])
            SLIPR[idx] = abs(float(v_l[nx-1]) - float(v_r[0]))
            TRACT[idx] = float(Tau_0) + float(s_l[nx-1])
            TIME[idx]  = n * dt
            idx += 1

    actual = idx
    VL, SL, VR, SR = VL[:, :actual], SL[:, :actual], VR[:, :actual], SR[:, :actual]
    SLIP, SLIPR, TRACT, TIME = SLIP[:actual], SLIPR[:actual], TRACT[:actual], TIME[:actual]

    return dict(VL=VL, SL=SL, VR=VR, SR=SR,
                SLIP=SLIP, SLIPR=SLIPR, TRACT=TRACT, TIME=TIME,
                nx=nx, nt_actual=actual,
                dx=dx,         # km
                dt=dt,         # s
                L=float(p_dict['L']))   # km


def _write_sample(path, sim, raw_params, bc_label):
    """Downsample sim results to all three resolutions + metadata."""
    nx_full  = sim['nx']
    nt_full  = sim['nt_actual']
    dx_km    = sim['dx']   # km
    dt       = sim['dt']   # s
    L        = sim['L']    # km

    # Temporal downsample indices (same for all resolutions)
    ti  = np.round(np.linspace(0, nt_full - 1, NT_OUT)).astype(int)
    t_axis = sim['TIME'][ti]

    save = {
        'params_raw' : np.array([raw_params[k] for k in PARAM_NAMES],
                                 dtype=np.float32),
        'param_names': json.dumps(PARAM_NAMES),
        'param_bounds': json.dumps(PARAM_BOUNDS),
        'slip'       : sim['SLIP'][ti],
        'sliprate'   : sim['SLIPR'][ti],
        'traction'   : sim['TRACT'][ti],
        'time'       : t_axis.astype(np.float32),
        'dx_native_km' : np.float32(dx_km),
        'dt_native_s'  : np.float32(dt),
        'L_km'         : np.float32(L),
        'bc_label'   : bc_label,
        'r0_l'       : np.int8(int(raw_params['r0_l'])),
        'r1_r'       : np.int8(int(raw_params['r1_r'])),
    }

    for label, nx_out in RESOLUTIONS.items():
        xi = np.round(np.linspace(0, nx_full - 1, nx_out)).astype(int)
        dx_out_km = L / (nx_out - 1)   # km
        save[f'v_l_{label}'] = sim['VL'][np.ix_(xi, ti)].astype(np.float32)
        save[f's_l_{label}'] = sim['SL'][np.ix_(xi, ti)].astype(np.float32)
        save[f'v_r_{label}'] = sim['VR'][np.ix_(xi, ti)].astype(np.float32)
        save[f's_r_{label}'] = sim['SR'][np.ix_(xi, ti)].astype(np.float32)
        save[f'dx_km_{label}'] = np.float32(dx_out_km)
        x_arr = np.linspace(0.0, L, nx_out, dtype=np.float32)   # km
        save[f'x_{label}']   = x_arr

    np.savez_compressed(path, **save)


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker(args):
    orig_idx, raw_params, tmp_path, p_base = args
    p = dict(p_base)
    p.update({k: raw_params[k] for k in ['Tau_0', 'alp_s', 'alp_d',
                                          'D_c', 'sigma_n', 'cs', 'rho',
                                          'r0_l', 'r1_r']})
    p['nx']  = NX_NATIVE
    # L is in km; dx in km; cs in km/s → dt in seconds
    p['dx']  = float(p['L']) / (NX_NATIVE - 1)
    p['mu']  = float(p['rho']) * float(p['cs']) ** 2
    p['dt']  = (float(p['cfl']) / float(p['cs'])) * p['dx']
    p['nt']  = int(round(float(p['tend']) / p['dt']))

    try:
        sim = _run_sim_inmem(p)
        _write_sample(tmp_path, sim, raw_params, raw_params['bc_label'])
        return orig_idx, True, tmp_path
    except Exception as e:
        return orig_idx, False, str(e)


# ---------------------------------------------------------------------------
# LHS sampling with nucleation + stability checks
# ---------------------------------------------------------------------------

def _sample_params(n_samples, bc_preset, seed):
    rng     = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    samples = rng.random(n=n_samples * 4)

    r0_l = bc_preset['r0_l']
    r1_r = bc_preset['r1_r']
    results = []
    for row in samples:
        raw = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            raw[name] = lo + row[i] * (hi - lo)

        # Validity checks
        if raw['alp_d'] >= raw['alp_s']:
            continue
        sigma_n = raw['sigma_n']
        tau_peak = raw['alp_s'] * sigma_n
        if raw['Tau_0'] <= tau_peak:
            continue
        # Nucleation: background stress must exceed peak strength
        # Allow small over-stress (at least 0.1 MPa)
        if raw['Tau_0'] - tau_peak < 0.1:
            continue

        raw['r0_l']    = r0_l
        raw['r1_r']    = r1_r
        raw['bc_label'] = f'r0l{r0_l}_r1r{r1_r}'
        results.append(raw)
        if len(results) >= n_samples:
            break

    if len(results) < n_samples:
        raise RuntimeError(
            f"Only {len(results)}/{n_samples} valid samples. Widen bounds or "
            "increase oversample factor.")
    return results[:n_samples]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(split_name, param_list, out_dir, p_base, n_workers):
    os.makedirs(out_dir, exist_ok=True)
    n = len(param_list)
    print(f'\n[{split_name}] generating {n} samples → {out_dir}')

    job_args = [
        (i, param_list[i],
         os.path.join(out_dir, f'tmp_{i:05d}.npz'),
         p_base)
        for i in range(n)
    ]

    results = []
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_worker, job_args):
                results.append(res)
                done = sum(1 for r in results if r[1])
                print(f'  [{split_name}] {done}/{n} ok', end='\r', flush=True)
    else:
        for args in job_args:
            results.append(_worker(args))

    print()

    # Sort by original index; rename tmp → sample_NNNNN.npz
    results.sort(key=lambda r: r[0])
    n_saved = 0
    for orig_idx, ok, tmp_or_err in results:
        tmp_path = os.path.join(out_dir, f'tmp_{orig_idx:05d}.npz')
        if ok and os.path.exists(tmp_path):
            final = os.path.join(out_dir, f'sample_{n_saved:05d}.npz')
            os.rename(tmp_path, final)
            n_saved += 1
        else:
            print(f'  SKIP {orig_idx}: {tmp_or_err}')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f'[{split_name}] saved {n_saved}/{n}.\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plan-C SW dataset generator')
    parser.add_argument('--bc',       choices=['both0', 'both1'], required=True)
    parser.add_argument('--n_train',  type=int, default=400)
    parser.add_argument('--n_val',    type=int, default=100)
    parser.add_argument('--n_test',   type=int, default=100)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--n_workers',type=int, default=min(48, cpu_count()))
    parser.add_argument('--data_dir', default=None)
    args = parser.parse_args()

    bc_preset = BC_PRESETS[args.bc]
    bc_label  = f'r0l{bc_preset["r0_l"]}_r1r{bc_preset["r1_r"]}'
    if args.data_dir is None:
        args.data_dir = os.path.join(_PLANC, 'data', f'bc_{args.bc}')

    # Load baseline parameters
    p_base = build_params(parse_infile(BASELINE_IN))
    p_base.update(bc_preset)

    print(f'Plan-C dataset generation')
    print(f'  BC: {args.bc}  ({bc_label})')
    print(f'  Resolutions: {list(RESOLUTIONS.items())}')
    print(f'  Native grid: {NX_NATIVE} pts (dx=10 m)')
    print(f'  Workers: {args.n_workers}')
    print(f'  Output: {args.data_dir}')

    # JIT warm-up
    print('\nJIT warm-up …')
    numba.set_num_threads(1)
    p_warm = dict(p_base)
    p_warm['nx']  = NX_NATIVE
    p_warm['dx']  = float(p_warm['L']) / (NX_NATIVE - 1)   # km
    p_warm['dt']  = (float(p_warm['cfl']) / float(p_warm['cs'])) * p_warm['dx']
    p_warm['nt']  = int(round(float(p_warm['tend']) / p_warm['dt']))
    _run_sim_inmem(p_warm)
    print('  JIT done.\n')

    total = args.n_train + args.n_val + args.n_test
    print(f'Sampling {total} parameter sets (LHS seed={args.seed}) ...')
    all_params = _sample_params(total, bc_preset, args.seed)

    train_p = all_params[:args.n_train]
    val_p   = all_params[args.n_train:args.n_train + args.n_val]
    test_p  = all_params[args.n_train + args.n_val:]

    for split, plist in [('train', train_p), ('val', val_p), ('test', test_p)]:
        generate(split, plist,
                 os.path.join(args.data_dir, split),
                 p_base, args.n_workers)

    print('All done.')


if __name__ == '__main__':
    main()
