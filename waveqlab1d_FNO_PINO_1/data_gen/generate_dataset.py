#!/usr/bin/env python3
"""
generate_dataset.py — Generate PINO-FNO training data using waveqlab1d simulator
==================================================================================

Runs the waveqlab1d rupture simulator in-memory for many parameter combinations
(Latin Hypercube Sampling) and saves each sample as a .npz file.

Supports both Plan A (unified, all conditions) and Plan B (one model at a time).

Usage:
    # Plan A — all conditions mixed
    python data_gen/generate_dataset.py --plan A --n_train 2400 --n_val 300 --n_test 300

    # Plan B — one model at a time
    python data_gen/generate_dataset.py --plan B --model sw_absorbing --n_train 800

    # Multi-resolution storage (HR + LR both saved)
    python data_gen/generate_dataset.py --plan A --n_train 2400 --nx_hr 256 --nt_hr 256

Output:
    data/{plan_A|model_key}/train/sample_{i:05d}.npz
    data/{plan_A|model_key}/val/sample_{i:05d}.npz
    data/{plan_A|model_key}/test/sample_{i:05d}.npz

Each .npz contains:
    # --- inputs (FNO channels) ---
    cs_arr      : (NX_SIM,)       shear wave velocity profile [km/s]
    rho_arr     : (NX_SIM,)       density [g/cm³]
    mu_arr      : (NX_SIM,)       shear modulus [GPa]
    Qs_inv_arr  : (NX_SIM,)       1/Q_S profile (0 if elastic)
    d_l         : (NX_SIM,)       PML damping left [1/s] (0 if no PML)
    d_r         : (NX_SIM,)       PML damping right [1/s]
    # --- outputs at training resolution ---
    v_l_lr      : (NX_LR, NT_LR)  left velocity [km/s]
    s_l_lr      : (NX_LR, NT_LR)  left stress [MPa]
    v_r_lr      : (NX_LR, NT_LR)
    s_r_lr      : (NX_LR, NT_LR)
    # --- at high resolution (if nx_hr > 0) ---
    v_l_hr      : (NX_HR, NT_HR)  (omitted if --nx_hr 0)
    s_l_hr      : ...
    v_r_hr, s_r_hr
    # --- fault time series ---
    slip        : (NT_SIM_stored,)
    sliprate    : (NT_SIM_stored,)
    traction    : (NT_SIM_stored,)
    time        : (NT_SIM_stored,)
    # --- metadata ---
    params      : JSON string of all physical parameters
    fric_law    : str ('SW' | 'RS')
    bc_mode     : str ('free' | 'absorbing' | 'pml')
    response    : str ('elastic' | 'anelastic')
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import zoom

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
_SIM    = os.path.join(_ROOT, '..', 'waveqlab1d')   # path to waveqlab1d project
_SIM    = os.path.normpath(_SIM)

# Add simulator root and src to path
sys.path.insert(0, _SIM)
sys.path.insert(0, os.path.join(_SIM, 'src'))

from rupture_1d import (build_params, validate,
                        init_fields, build_friction_parameters)
from kernels import rk4_step, rk4_step_anelastic, FRIC_SW, FRIC_RS
from anelastic import init_anelastic
from pml import init_pml

# Local
sys.path.insert(0, _ROOT)
from data_gen.param_space import (
    sample_sw, sample_rs, sample_unified, sample_for_model, PLAN_B_MODELS
)


# ─── Simulator constants ──────────────────────────────────────────────────────

SIM_DEFAULTS = dict(
    L        = 30.0,    # km
    nx       = 301,     # spatial points per domain (hi-fidelity)
    tend     = 10.0,    # s
    cfl      = 0.5,
    order    = 6,
    r1_l     = 1,       # inner (fault-side) BCs: always free-surface
    r0_r     = 1,
    fref     = 1.0,
    # SW defaults (overridden by sampled params)
    slip_init = 0.0,
    psi_init  = 0.4367,
)


# ─── In-memory simulator ─────────────────────────────────────────────────────

def run_sim_inmem(p: dict) -> dict | None:
    """
    Run waveqlab1d simulator in memory for parameter dict p.
    Returns dict of output arrays, or None if simulation diverged.

    p must include all fields from SIM_DEFAULTS + sampled variable params.
    """
    # Build complete parameter set
    full = dict(SIM_DEFAULTS)
    full.update(p)
    full['fric_law'] = p.get('fric_law', 'SW')

    # Derive nx, dx, dt, nt from L / cfl / cs
    nx  = int(full['nx'])
    dx  = float(full['L']) / (nx - 1)
    cs  = float(full.get('cs', 3.464))
    rho = float(full.get('rho', 2.67))
    dt  = (float(full['cfl']) / cs) * dx
    nt  = int(round(float(full['tend']) / dt))

    # Build uniform material arrays (spatially variable cs/rho not yet supported
    # in data-gen; profiles can be added as a later extension)
    cs_arr  = np.full(nx, cs,  dtype=np.float64)
    rho_arr = np.full(nx, rho, dtype=np.float64)
    mu_arr  = rho_arr * cs_arr ** 2

    # PML damping
    use_pml = bool(full.get('pml', False))
    if use_pml:
        npml      = int(full.get('npml', 20))
        pml_alpha = float(full.get('pml_alpha', 10.0))
        d_l, d_r  = init_pml(nx, npml, cs, dx, pml_alpha)
    else:
        d_l = np.zeros(nx, dtype=np.float64)
        d_r = np.zeros(nx, dtype=np.float64)

    # Anelastic initialisation
    response = full.get('response', 'elastic')
    Qs_inv_arr = np.zeros(nx, dtype=np.float64)
    if response == 'anelastic' and float(full.get('c', 0.0)) > 0:
        anel = init_anelastic(
            nx, mu_arr, rho_arr,
            c          = float(full['c']),
            weight_exp = float(full.get('weight_exp', 0.0)),
            fref       = float(full.get('fref', 1.0)),
        )
        mu_arr_run = anel['mu_unrelax']
        Qs_inv_arr = anel['Qs_inv']
        tau_mech   = anel['tau']
        weight     = anel['weight']
        eta_l      = np.zeros((nx, 4), dtype=np.float64)
        eta_r      = np.zeros((nx, 4), dtype=np.float64)
    else:
        mu_arr_run = mu_arr
        tau_mech = weight = eta_l = eta_r = None
        response = 'elastic'   # force elastic if c=0

    # Build friction params array
    full['nx']  = nx
    full['dx']  = dx
    full['dt']  = dt
    full['nt']  = nt
    full['Tau_0'] = full.get('Tau_0', 81.6)
    # build_friction_parameters expects ALL 12 params regardless of fric_law;
    # supply safe defaults for the law that is NOT being used.
    _fp_defaults = {
        # RS defaults (used when fric_law=='SW')
        'L0': 0.02, 'f0': 0.6, 'a': 0.010, 'b': 0.020, 'V0': 1e-6,
        # SW defaults (used when fric_law=='RS')
        'alp_s': 0.68, 'alp_d': 0.50, 'D_c': 0.4,
        'sigma_n': 120.0,
    }
    for k, v in _fp_defaults.items():
        full.setdefault(k, v)
    fp = build_friction_parameters(full)

    # Initial fields
    y_l, y_r, v_l, s_l, v_r, s_r, slip, psi = init_fields(full)

    order = int(full['order'])
    r0_l  = float(full.get('r0_l', 1))
    r1_r  = float(full.get('r1_r', 1))
    Tau_0 = float(full['Tau_0'])

    # Storage (every step — will downsample after)
    VL = np.zeros((nx, nt), dtype=np.float32)
    SL = np.zeros((nx, nt), dtype=np.float32)
    VR = np.zeros((nx, nt), dtype=np.float32)
    SR = np.zeros((nx, nt), dtype=np.float32)
    SLIP   = np.zeros(nt, dtype=np.float32)
    SRATE  = np.zeros(nt, dtype=np.float32)
    TRACT  = np.zeros(nt, dtype=np.float32)
    TIME   = np.zeros(nt, dtype=np.float32)

    try:
        for n in range(nt):
            if response == 'anelastic':
                rk4_step_anelastic(
                    v_l, s_l, eta_l, v_r, s_r, eta_r, slip, psi,
                    rho_arr, mu_arr_run, Qs_inv_arr, tau_mech, weight,
                    d_l, d_r, nx, dx, order, r0_l, r1_r, dt, fp)
            else:
                rk4_step(
                    v_l, s_l, v_r, s_r, slip, psi,
                    float(rho), float(mu_arr[0]), nx, dx, order, r0_l, r1_r,
                    d_l, d_r, dt, fp)

            VL[:, n] = v_l.astype(np.float32)
            SL[:, n] = s_l.astype(np.float32)
            VR[:, n] = v_r.astype(np.float32)
            SR[:, n] = s_r.astype(np.float32)
            SLIP[n]  = float(slip[0])
            SRATE[n] = abs(float(v_l[nx-1]) - float(v_r[0]))
            TRACT[n] = float(Tau_0) + float(s_l[nx-1])
            TIME[n]  = n * dt

            # Divergence check
            if not np.isfinite(v_l[nx-1]):
                return None

    except Exception:
        return None

    return dict(
        cs_arr    = cs_arr.astype(np.float32),
        rho_arr   = rho_arr.astype(np.float32),
        mu_arr    = mu_arr.astype(np.float32),
        Qs_inv    = Qs_inv_arr.astype(np.float32),
        d_l       = d_l.astype(np.float32),
        d_r       = d_r.astype(np.float32),
        v_l       = VL, s_l = SL,
        v_r       = VR, s_r = SR,
        slip      = SLIP, sliprate = SRATE,
        traction  = TRACT, time = TIME,
        dx        = np.float32(dx),
        dt        = np.float32(dt),
        nx_sim    = np.int32(nx),
        nt_sim    = np.int32(nt),
        response  = response,
    )


# ─── Downsampling ─────────────────────────────────────────────────────────────

def downsample_field(field: np.ndarray, nx_out: int, nt_out: int) -> np.ndarray:
    """
    Bilinear downsample a (NX_SIM, NT_SIM) field to (nx_out, nt_out).
    Uses scipy.ndimage.zoom with order=1 (bilinear).
    """
    nx_in, nt_in = field.shape
    zx = nx_out / nx_in
    zt = nt_out / nt_in
    return zoom(field, (zx, zt), order=1).astype(np.float32)


# ─── Sample writer ────────────────────────────────────────────────────────────

def write_sample(out_path: str, raw: dict, params: dict,
                 nx_lr: int, nt_lr: int,
                 nx_hr: int, nt_hr: int):
    """
    Downsample raw simulation output to LR (and optionally HR) and save as .npz.
    """
    save = dict(
        cs_arr   = raw['cs_arr'],
        rho_arr  = raw['rho_arr'],
        mu_arr   = raw['mu_arr'],
        Qs_inv   = raw['Qs_inv'],
        d_l      = raw['d_l'],
        d_r      = raw['d_r'],
        slip     = raw['slip'],
        sliprate = raw['sliprate'],
        traction = raw['traction'],
        time     = raw['time'],
        dx       = raw['dx'],
        dt       = raw['dt'],
        params   = np.array(json.dumps(params)),
    )

    for name, field in [('v_l', raw['v_l']), ('s_l', raw['s_l']),
                         ('v_r', raw['v_r']), ('s_r', raw['s_r'])]:
        save[f'{name}_lr'] = downsample_field(field, nx_lr, nt_lr)
        if nx_hr > 0 and nt_hr > 0:
            save[f'{name}_hr'] = downsample_field(field, nx_hr, nt_hr)

    np.savez_compressed(out_path, **save)


# ─── Main loop ────────────────────────────────────────────────────────────────

def _worker(args):
    """Worker function for parallel generation (top-level for pickling).
    Returns (orig_idx, ok, tmp_path).
    """
    orig_idx, p, tmp_path, nx_lr, nt_lr, nx_hr, nt_hr = args
    raw = run_sim_inmem(p)
    if raw is None:
        return orig_idx, False, tmp_path
    write_sample(tmp_path, raw, p, nx_lr, nt_lr, nx_hr, nt_hr)
    return orig_idx, True, tmp_path


def generate(
    param_list   : list[dict],
    out_dir      : str,
    nx_lr        : int,
    nt_lr        : int,
    nx_hr        : int,
    nt_hr        : int,
    max_failures : int = 50,
    n_workers    : int = 1,
):
    """Generate and save all samples, skipping diverged runs."""
    os.makedirs(out_dir, exist_ok=True)
    n_target = len(param_list)
    n_saved  = 0
    n_failed = 0

    # Pre-assign temp output paths using original index (unique, no collisions)
    work_items = [
        (i, p, os.path.join(out_dir, f'tmp_{i:05d}.npz'), nx_lr, nt_lr, nx_hr, nt_hr)
        for i, p in enumerate(param_list)
    ]

    if n_workers > 1:
        # Parallel: write tmp_XXXXX.npz, then rename successful files to
        # sequential sample_XXXXX.npz in original-index order.
        print(f"  (parallel: {n_workers} workers)")
        results = {}  # orig_idx -> (ok, tmp_path)
        with Pool(processes=n_workers) as pool:
            for orig_idx, ok, tmp_path in pool.imap_unordered(_worker, work_items):
                results[orig_idx] = (ok, tmp_path)
                done = len(results)
                if done % 20 == 0 or done == n_target:
                    print(f"  {done}/{n_target} completed", flush=True)

        # Rename in original order → sequential sample indices
        for orig_idx in sorted(results):
            ok, tmp_path = results[orig_idx]
            if ok and os.path.exists(tmp_path):
                final = os.path.join(out_dir, f'sample_{n_saved:05d}.npz')
                os.rename(tmp_path, final)
                n_saved += 1
            else:
                n_failed += 1
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        # Serial generation for reproducibility / debugging
        for i, (p, out_path, nx_lr_, nt_lr_, nx_hr_, nt_hr_) in enumerate(work_items):
            print(f"  [{i+1}/{n_target}] {p.get('fric_law','?')} "
                  f"{p.get('bc_mode','?')} cs={p.get('cs',3.464):.2f} "
                  f"resp={p.get('response','elastic')} ... ", end='', flush=True)
            raw = run_sim_inmem(p)
            if raw is None:
                n_failed += 1
                print(f"FAILED (total failures: {n_failed})")
                if n_failed >= max_failures:
                    print(f"Too many failures ({n_failed}), aborting.")
                    break
                continue
            final_path = os.path.join(out_dir, f'sample_{n_saved:05d}.npz')
            write_sample(final_path, raw, p, nx_lr, nt_lr, nx_hr, nt_hr)
            n_saved += 1
            print(f"OK → sample_{n_saved-1:05d}.npz")

    print(f"\nSaved {n_saved}/{n_target} samples ({n_failed} failures)")
    return n_saved


def main():
    parser = argparse.ArgumentParser(
        description='Generate waveqlab1d PINO-FNO dataset')
    parser.add_argument('--plan',    choices=['A', 'B'], default='A')
    parser.add_argument('--model',   default='sw_absorbing',
                        help='Plan B model key (e.g. sw_free, rs_pml)')
    parser.add_argument('--n_train', type=int, default=2400)
    parser.add_argument('--n_val',   type=int, default=300)
    parser.add_argument('--n_test',  type=int, default=300)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--nx_lr',   type=int, default=128,
                        help='Spatial points at training resolution')
    parser.add_argument('--nt_lr',   type=int, default=128,
                        help='Time steps at training resolution')
    parser.add_argument('--nx_hr',   type=int, default=256,
                        help='Spatial points at SR target (0 = skip HR)')
    parser.add_argument('--nt_hr',   type=int, default=256,
                        help='Time points at SR target')
    parser.add_argument('--data_dir', default=None,
                        help='Output root directory (default: data/<plan_key>)')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Parallel workers for simulation (default: 1)')
    args = parser.parse_args()

    if args.plan == 'A':
        plan_key = 'plan_A'
        train_params = sample_unified(args.n_train, seed=args.seed)
        val_params   = sample_unified(args.n_val,   seed=args.seed + 10000)
        test_params  = sample_unified(args.n_test,  seed=args.seed + 20000)
    else:
        plan_key   = args.model
        fric, _    = args.model.upper().split('_', 1)
        if fric == 'SW':
            fn = lambda n, s: sample_for_model(args.model, n, seed=s)
        else:
            fn = lambda n, s: sample_for_model(args.model, n, seed=s)
        train_params = fn(args.n_train, args.seed)
        val_params   = fn(args.n_val,   args.seed + 10000)
        test_params  = fn(args.n_test,  args.seed + 20000)

    root = args.data_dir or os.path.join(_ROOT, 'data', plan_key)
    print(f"\nPlan {args.plan} | key={plan_key} | "
          f"train={len(train_params)} val={len(val_params)} test={len(test_params)}")
    print(f"LR resolution: ({args.nx_lr}, {args.nt_lr})")
    if args.nx_hr > 0:
        print(f"HR resolution: ({args.nx_hr}, {args.nt_hr})")
    print(f"Output root:   {root}\n")

    for split, params in [('train', train_params),
                           ('val',   val_params),
                           ('test',  test_params)]:
        print(f"─── {split} ({len(params)} samples) ───")
        generate(params, os.path.join(root, split),
                 args.nx_lr, args.nt_lr, args.nx_hr, args.nt_hr,
                 n_workers=args.n_workers)

    print('\nDone.')


if __name__ == '__main__':
    main()
