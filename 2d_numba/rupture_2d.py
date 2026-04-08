#!/usr/bin/env python3
"""
2D Dynamic Rupture Simulation Runner — Numba edition
======================================================
Usage:
    python3 rupture_2d.py input/rupture_2d_SW.in [-np 4]

Reads all parameters from the .in file (key = value, # comments).
Saves results to:
    output/{output_prefix}_{run_id}.npz

Output format is identical to 2d_serial/rupture_2d.py, enabling direct
comparison and use with the same auxiliary/plotting scripts.

Key differences vs. 2d_serial:
  - All inner loops replaced by @njit(parallel=True, cache=True) Numba kernels
  - slip, psi are flat float64[ny] arrays (not (ny,1) column vectors)
  - FaultOutput columns 4-5 store the CURRENT slip/psi at each step
    (serial has a bug: it stores the initial values throughout)
  - JIT warmup step on a tiny 4×4 grid before the main loop
  - Thread count controlled by -np argument (default: all cores)
"""

import argparse
import hashlib
import json
import os
import sys
import timeit
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Simulation modules live in src/ next to this script
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numba
import kernels_2d as K

# ---------------------------------------------------------------------------
# Default parameter values (identical to 2d_serial)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Output
    output_prefix  = 'rupture_2d',
    iplot          = 5,
    # Domain
    Lx             = 10.0,
    Ly             = 20.0,
    nx             = 26,
    ny             = 51,
    # Material
    cs             = 3.464,
    cp             = 6.0,
    rho            = 2.6702,
    # Time stepping
    cfl            = 0.5,
    tend           = 10.0,
    # Scheme
    order          = 6,
    # Fracture mode: 'II' (in-plane) or 'III' (anti-plane)
    mode           = 'II',
    # Friction law
    fric_law       = 'SW',
    # Nucleation patch center along fault (km)
    Y0             = 10.0,
    # Point source (set M0=0 to disable)
    x0             = -15.0,
    y0             = 7.5,
    t0             = 0.0,
    T              = 0.1,
    M0             = 0.0,
    source_type    = 'Gaussian',
    # Boundary reflection coefficients [x_outer, x_fault, y_bottom, y_top]
    r_l_0          = 0.,   r_l_1 = 0.,   r_l_2 = 1.,   r_l_3 = 0.,
    r_r_0          = 0.,   r_r_1 = 0.,   r_r_2 = 1.,   r_r_3 = 0.,
    # SW parameters
    Tau_0_SW       = 70.0,
    alp_s          = 0.677,
    alp_d          = 0.525,
    D_c            = 0.4,
    sigma_n        = 120.0,
    # RS parameters
    Tau_0_RS       = 75.0,
    f0             = 0.6,
    a              = 0.008,
    b              = 0.012,
    V0             = 1.0e-6,
    L0             = 0.02,
    psi_init       = 0.4367,
    sigma_n_RS     = 120.0,
    Vin            = 2.0e-12,
)

VALID_FRIC_LAWS = ('SW', 'RS')
VALID_ORDERS    = (2, 4, 6)
VALID_MODES     = ('II', 'III')


# ---------------------------------------------------------------------------
# Parsing (identical to 2d_serial)
# ---------------------------------------------------------------------------

def parse_infile(path):
    """Parse key = value input file. Returns dict with typed values."""
    raw = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or '=' not in line:
                continue
            k, v = [s.strip() for s in line.split('=', 1)]
            if not k:
                continue
            raw[k] = v

    params = {}
    for k, v in raw.items():
        if k in ('fric_law', 'mode', 'output_prefix', 'source_type'):
            params[k] = v.strip()
            continue
        for conv in (int, float):
            try:
                params[k] = conv(v)
                break
            except ValueError:
                pass
        else:
            params[k] = v
    return params


def build_params(user, num_threads):
    """Merge user values over defaults; derive grid/time quantities."""
    p = dict(DEFAULTS)
    p.update(user)

    p['nx']  = int(p['nx'])
    p['ny']  = int(p['ny'])
    p['fric_law'] = p['fric_law'].upper()
    p['mode']     = p['mode'].upper()
    p['num_threads'] = int(num_threads)

    p['dx'] = float(p['Lx']) / (p['nx'] - 1)
    p['dy'] = float(p['Ly']) / (p['ny'] - 1)

    if 'dt' not in p:
        p['dt'] = float(p['cfl']) / np.sqrt(float(p['cp'])**2 + float(p['cs'])**2) * p['dx']
    p['dt'] = float(p['dt'])
    p['nt'] = int(round(float(p['tend']) / p['dt']))

    # nf depends on mode
    p['nf'] = 5 if p['mode'] == 'II' else 3

    # Lame parameters
    mu     = float(p['rho']) * float(p['cs'])**2
    Lambda = float(p['rho']) * float(p['cp'])**2 - 2.0 * mu
    p['mu']        = mu
    p['Lambda']    = Lambda
    p['twomulam']  = 2.0 * mu + Lambda   # convenience for kernels

    return p


# ---------------------------------------------------------------------------
# Validation (identical to 2d_serial)
# ---------------------------------------------------------------------------

def validate(p):
    errors = []

    if p['fric_law'] not in VALID_FRIC_LAWS:
        errors.append(f"fric_law='{p['fric_law']}' must be one of {VALID_FRIC_LAWS}")
    if p['mode'] not in VALID_MODES:
        errors.append(f"mode='{p['mode']}' must be one of {VALID_MODES}")
    if int(p['order']) not in VALID_ORDERS:
        errors.append(f"order={p['order']} must be one of {VALID_ORDERS}")
    if not (0 < float(p['cfl']) <= 1.0):
        errors.append(f"cfl={p['cfl']} must be in (0, 1]")
    if p['nx'] < 4:
        errors.append(f"nx={p['nx']} is too small")
    if p['ny'] < 4:
        errors.append(f"ny={p['ny']} is too small")
    if float(p['tend']) <= 0:
        errors.append(f"tend={p['tend']} must be positive")

    if p['fric_law'] == 'SW':
        if float(p['alp_d']) >= float(p['alp_s']):
            errors.append(
                f"SW requires alp_d ({p['alp_d']}) < alp_s ({p['alp_s']})")
        if float(p['D_c']) <= 0:
            errors.append(f"D_c={p['D_c']} must be positive")
        if float(p['sigma_n']) <= 0:
            errors.append(f"sigma_n={p['sigma_n']} must be positive")

    if p['fric_law'] == 'RS':
        if float(p['a']) >= float(p['b']):
            errors.append(
                f"RS velocity-weakening requires b ({p['b']}) > a ({p['a']})")
        if float(p['V0']) <= 0:
            errors.append(f"V0={p['V0']} must be positive")
        if float(p['L0']) <= 0:
            errors.append(f"L0={p['L0']} must be positive")

    if errors:
        print("PARAMETER ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Domain initialisation
# ---------------------------------------------------------------------------

def init_domains(p):
    """Build coordinate grids."""
    nx  = p['nx']
    ny  = p['ny']
    dx  = p['dx']
    dy  = p['dy']
    Lx  = float(p['Lx'])

    X_l = np.zeros((nx, ny))
    Y_l = np.zeros((nx, ny))
    X_r = np.zeros((nx, ny))
    Y_r = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            X_l[i, j] = -Lx + i * dx
            Y_l[i, j] = j * dy
            X_r[i, j] = i * dx
            Y_r[i, j] = j * dy

    # nucleation patch indicator along fault
    Y0 = float(p['Y0'])
    Y_fault = np.zeros((ny, 1))
    for j in range(ny):
        if abs(j * dy - Y0) <= 1.5:
            Y_fault[j, 0] = 1.0

    return X_l, Y_l, X_r, Y_r, Y_fault


def init_fault_state(p, Y_fault):
    """
    Initialise slip, psi (flat 1D) and friction_parameters arrays.

    Key difference from 2d_serial: slip/psi are float64[ny] (not (ny,1))
    for direct compatibility with @njit kernels.
    """
    ny       = p['ny']
    fric_law = p['fric_law']
    alpha    = 1e308   # effectively infinite initial friction

    slip = np.zeros(ny, dtype=np.float64)
    psi  = np.zeros(ny, dtype=np.float64)

    if fric_law == 'SW':
        Tau_0   = np.ones((ny, 1)) * (float(p['Tau_0_SW']) + 11.6 * Y_fault)
        alp_s   = np.full((ny, 1), float(p['alp_s']))
        alp_d   = np.full((ny, 1), float(p['alp_d']))
        D_c     = np.full((ny, 1), float(p['D_c']))
        sigma_n = -np.full((ny, 1), float(p['sigma_n']))
        L0      = np.ones((ny, 1))
        f0      = np.ones((ny, 1))
        a       = np.ones((ny, 1))
        b       = np.ones((ny, 1))
        V0      = np.ones((ny, 1))

    else:  # RS
        sigma_n = -np.full((ny, 1), float(p['sigma_n_RS']))
        Tau_0   = np.full((ny, 1), float(p['Tau_0_RS']))
        L0      = np.full((ny, 1), float(p['L0']))
        f0      = np.full((ny, 1), float(p['f0']))
        a       = np.full((ny, 1), float(p['a']))
        b       = np.full((ny, 1), float(p['b']))
        V0      = np.full((ny, 1), float(p['V0']))
        Vin     = np.full((ny, 1), float(p['Vin']))
        theta   = (L0 / V0 * np.exp(
            (a * np.log(2.0 * np.sinh(float(p['Tau_0_RS']) / (a * float(p['sigma_n_RS']))))
             - f0 - a * np.log(Vin / V0)) / b))
        psi[:] = (f0[:, 0] + b[:, 0] * np.log(V0[:, 0] / L0[:, 0] * theta[:, 0]))
        alp_s   = np.ones((ny, 1))
        alp_d   = np.ones((ny, 1))
        D_c     = np.ones((ny, 1))

    # Pack friction_parameters: shape (12, ny) float64
    # rows: [alpha, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]
    friction_parameters = np.zeros((12, ny), dtype=np.float64)
    friction_parameters[0,  :] = alpha
    friction_parameters[1,  :] = alpha
    friction_parameters[2,  :] = Tau_0[:, 0]
    friction_parameters[3,  :] = L0[:, 0]
    friction_parameters[4,  :] = f0[:, 0]
    friction_parameters[5,  :] = a[:, 0]
    friction_parameters[6,  :] = b[:, 0]
    friction_parameters[7,  :] = V0[:, 0]
    friction_parameters[8,  :] = sigma_n[:, 0]
    friction_parameters[9,  :] = alp_s[:, 0]
    friction_parameters[10, :] = alp_d[:, 0]
    friction_parameters[11, :] = D_c[:, 0]

    # fault_output0: shape (ny, 6) — columns 0-3 from kernel, 4-5 from slip/psi
    fault_output0 = np.zeros((ny, 6), dtype=np.float64)
    fault_output0[:, 2] = sigma_n[:, 0]
    fault_output0[:, 3] = Tau_0[:, 0]
    fault_output0[:, 4] = slip
    fault_output0[:, 5] = psi

    return slip, psi, friction_parameters, fault_output0


# ---------------------------------------------------------------------------
# Run ID (identical to 2d_serial)
# ---------------------------------------------------------------------------

def make_run_id(p):
    skip = {'dx', 'dy', 'dt', 'nt', 'nf', 'mu', 'Lambda', 'twomulam',
            'output_prefix', 'iplot', 'num_threads'}
    core = {k: v for k, v in sorted(p.items()) if k not in skip}
    blob = json.dumps(core, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def jit_warmup(p):
    """
    Run one RK4 step on a tiny grid to trigger JIT compilation
    before the timed main loop.

    Minimum grid size for SBP order 6: 17 pts each direction.
    Order-6 left-boundary row 7 accesses u[:,10,:] (needs ny>=11),
    and the no-overlap condition requires ny>=17.
    Using 17x17 is safe for all supported orders.
    """
    w_nx  = 17
    w_ny  = 17
    w_nf  = p['nf']
    w_dx  = p['dx']
    w_dy  = p['dy']
    w_dt  = p['dt']
    order = int(p['order'])

    F_l_w = np.zeros((w_nx, w_ny, w_nf))
    F_r_w = np.zeros((w_nx, w_ny, w_nf))
    slip_w   = np.zeros(w_ny)
    psi_w    = np.zeros(w_ny)
    fp_w     = np.zeros((12, w_ny))
    fp_w[2, :]  = float(p['Tau_0_SW']) if p['fric_law'] == 'SW' else float(p['Tau_0_RS'])
    fp_w[3, :]  = float(p['L0']) if p['fric_law'] == 'RS' else 1.0
    fp_w[4, :]  = float(p['f0']) if p['fric_law'] == 'RS' else 1.0
    fp_w[5, :]  = float(p['a'])  if p['fric_law'] == 'RS' else 1.0
    fp_w[6, :]  = float(p['b'])  if p['fric_law'] == 'RS' else 1.0
    fp_w[7, :]  = float(p['V0']) if p['fric_law'] == 'RS' else 1.0
    fp_w[8, :]  = -float(p['sigma_n'] if p['fric_law'] == 'SW' else p['sigma_n_RS'])
    fp_w[9, :]  = float(p['alp_s']) if p['fric_law'] == 'SW' else 1.0
    fp_w[10, :] = float(p['alp_d']) if p['fric_law'] == 'SW' else 1.0
    fp_w[11, :] = float(p['D_c'])   if p['fric_law'] == 'SW' else 1.0

    Y_arr_w = np.linspace(0.0, float(p['Ly']), w_ny)
    r_l_w   = np.array([0., 0., 1., 0.])
    r_r_w   = np.array([0., 0., 1., 0.])
    fric_flag = K.FRIC_SW if p['fric_law'] == 'SW' else K.FRIC_RS
    mode_int  = K.MODE_II if p['mode'] == 'II' else K.MODE_III
    fo_w = np.zeros((w_ny, 6))

    source_moment = np.zeros(w_nf)
    if w_nf == 5:
        source_moment[2] = 1.0; source_moment[3] = 1.0

    K.rk4_step_2d(
        F_l_w, F_r_w, slip_w, psi_w,
        fp_w, Y_arr_w, float(p['Y0']), 0.0,
        w_nx, w_ny, w_nf, w_dx, w_dy, w_dt, order,
        r_l_w, r_r_w,
        float(p['rho']), float(p['twomulam']), float(p['mu']),
        mode_int, fric_flag,
        0.0, 0.0, 0.0, 1.0, 0.0, source_moment,
        fo_w)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_sim(infile, num_threads):
    user_params = parse_infile(infile)
    p = build_params(user_params, num_threads)
    validate(p)

    # Set Numba thread count
    numba.set_num_threads(p['num_threads'])
    print(f"Numba threads : {numba.get_num_threads()}")

    run_id  = make_run_id(p)
    out_dir = os.path.join(_ROOT, 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{p['output_prefix']}_{run_id}.npz")

    print(f"Run ID        : {run_id}")
    print(f"fric_law      : {p['fric_law']}   mode: {p['mode']}")
    print(f"nx={p['nx']}  ny={p['ny']}  nt={p['nt']}  "
          f"dt={p['dt']:.6f} s  tend={p['tend']} s")
    print(f"dx={p['dx']:.4f} km  dy={p['dy']:.4f} km")
    print(f"order={p['order']}  cfl={p['cfl']}  Lx={p['Lx']} km  Ly={p['Ly']} km")
    print(f"Output        : {out_file}\n")

    nx       = p['nx']
    ny       = p['ny']
    nf       = p['nf']
    nt       = p['nt']
    dx       = p['dx']
    dy       = p['dy']
    dt       = p['dt']
    order    = int(p['order'])
    fric_law = p['fric_law']
    Y0       = float(p['Y0'])
    iplot    = int(p['iplot'])

    # Numba-compatible scalars and arrays
    rho      = float(p['rho'])
    mu       = float(p['mu'])
    twomulam = float(p['twomulam'])

    fric_flag = K.FRIC_SW if fric_law == 'SW' else K.FRIC_RS
    mode_int  = K.MODE_II if p['mode'] == 'II' else K.MODE_III

    # Boundary reflection coefficient arrays [x_outer, x_fault, y_bottom, y_top]
    r_l = np.array([p['r_l_0'], p['r_l_1'], p['r_l_2'], p['r_l_3']], dtype=np.float64)
    r_r = np.array([p['r_r_0'], p['r_r_1'], p['r_r_2'], p['r_r_3']], dtype=np.float64)

    # Source parameters as individual scalars + moment tensor array
    x0_src = float(p['x0'])
    y0_src = float(p['y0'])
    t0_src = float(p['t0'])
    T_src  = float(p['T'])
    M0_src = float(p['M0'])
    source_moment = np.zeros(nf, dtype=np.float64)
    if nf == 5:       # Mode II: sxx and syy components
        source_moment[2] = 1.0
        source_moment[3] = 1.0
    else:             # Mode III: sxz component (index 1)
        source_moment[1] = 1.0

    # Coordinate grids
    X_l, Y_l, X_r, Y_r, Y_fault = init_domains(p)
    Y_arr = Y_r[0, :].copy().astype(np.float64)   # y-coords along fault (ny,)

    # Fields: C-contiguous float64[nx, ny, nf]
    F_l = np.zeros((nx, ny, nf), dtype=np.float64)
    F_r = np.zeros((nx, ny, nf), dtype=np.float64)

    # Fault state (flat 1D arrays for Numba)
    slip, psi, friction_parameters, fault_output0 = init_fault_state(p, Y_fault)

    # Ensure all Numba arrays are C-contiguous float64
    friction_parameters = np.ascontiguousarray(friction_parameters, dtype=np.float64)
    Y_arr = np.ascontiguousarray(Y_arr, dtype=np.float64)
    r_l   = np.ascontiguousarray(r_l,   dtype=np.float64)
    r_r   = np.ascontiguousarray(r_r,   dtype=np.float64)

    # Output arrays
    n_snap = (nt + iplot - 1) // iplot
    snap_times = np.empty(n_snap, dtype=np.float64)

    FaultOutput    = np.zeros((ny, nt, 6), dtype=np.float32)
    DomainOutput_l = np.zeros((nx, ny, n_snap, nf), dtype=np.float32)
    DomainOutput_r = np.zeros((nx, ny, n_snap, nf), dtype=np.float32)

    step_time_compute = np.empty(nt, dtype=np.float64)
    step_time_total   = np.empty(nt, dtype=np.float64)

    # JIT warmup — compiles all kernels before the main timed loop
    print("JIT warmup ... ", end='', flush=True)
    t_warmup = timeit.default_timer()
    jit_warmup(p)
    print(f"done ({timeit.default_timer() - t_warmup:.1f} s)")

    snap_idx = 0
    t_start_dt = datetime.now()
    start = timeit.default_timer()
    print(f"Start time  : {t_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    for it in range(nt):
        t = it * dt

        t0_compute = timeit.default_timer()

        K.rk4_step_2d(
            F_l, F_r, slip, psi,
            friction_parameters, Y_arr, Y0, t,
            nx, ny, nf, dx, dy, dt, order,
            r_l, r_r,
            rho, twomulam, mu,
            mode_int, fric_flag,
            x0_src, y0_src, t0_src, T_src, M0_src, source_moment,
            fault_output0)

        step_time_compute[it] = timeit.default_timer() - t0_compute

        # Record fault time series
        # Columns 0-3: from kernel (slip rates, tractions)
        # Columns 4-5: current slip and psi (correct; serial has a bug here)
        FaultOutput[:, it, 0:4] = fault_output0[:, 0:4].astype(np.float32)
        FaultOutput[:, it, 4]   = slip.astype(np.float32)
        FaultOutput[:, it, 5]   = psi.astype(np.float32)

        if it % iplot == 0:
            DomainOutput_l[:, :, snap_idx, :] = F_l.astype(np.float32)
            DomainOutput_r[:, :, snap_idx, :] = F_r.astype(np.float32)
            snap_times[snap_idx] = (it + 1) * dt
            snap_idx += 1

        step_time_total[it] = timeit.default_timer() - t0_compute

        sliprate = float(np.sqrt(
            fault_output0[:, 0]**2 + fault_output0[:, 1]**2).max())
        print(f"  step {it+1:>6d}/{nt}  t={t+dt:.4f}s  "
              f"step_ms={step_time_compute[it]*1e3:.2f}  "
              f"max_sliprate={sliprate:.4f}m/s", flush=True)

    # Trim snap arrays if last block was partial
    DomainOutput_l = DomainOutput_l[:, :, :snap_idx, :]
    DomainOutput_r = DomainOutput_r[:, :, :snap_idx, :]
    snap_times     = snap_times[:snap_idx]

    wall = timeit.default_timer() - start
    t_end_dt = datetime.now()
    print(f"\nEnd time    : {t_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"total simulation time  = {wall:.2f} s")
    print(f"mean step (compute)    = {step_time_compute.mean()*1e3:.3f} ms")
    print(f"mean step (total)      = {step_time_total.mean()*1e3:.3f} ms")
    print(f"spatial order          = {order}")
    print(f"grid                   = {nx} x {ny}")
    print(f"threads                = {numba.get_num_threads()}")

    # ------------------------------------------------------------------
    # Save (identical format to 2d_serial)
    # ------------------------------------------------------------------
    metadata = {k: v for k, v in p.items()}
    metadata['run_id']               = run_id
    metadata['wall_time_s']          = wall
    metadata['mean_step_compute_ms'] = float(step_time_compute.mean() * 1e3)
    metadata['mean_step_total_ms']   = float(step_time_total.mean()   * 1e3)
    metadata['numba_threads']        = numba.get_num_threads()

    y_fault = Y_r[0, :]
    x_l     = X_l[:, 0]
    x_r     = X_r[:, 0]

    np.savez_compressed(
        out_file,
        DomainOutput_l      = DomainOutput_l,
        DomainOutput_r      = DomainOutput_r,
        snap_times          = snap_times.astype(np.float32),
        FaultOutput         = FaultOutput,
        friction_parameters = friction_parameters.astype(np.float32),
        x_l                 = x_l.astype(np.float32),
        x_r                 = x_r.astype(np.float32),
        y_fault             = y_fault.astype(np.float32),
        time                = (np.arange(1, nt + 1) * dt).astype(np.float32),
        metadata            = json.dumps(metadata, default=str),
    )

    timing_file = os.path.join(out_dir, f"{p['output_prefix']}_{run_id}_timing.npz")
    np.savez_compressed(
        timing_file,
        step_time_compute  = step_time_compute,
        step_time_total    = step_time_total,
        step_time_overhead = step_time_total - step_time_compute,
        metadata           = json.dumps(dict(
            run_id               = run_id,
            output_prefix        = p['output_prefix'],
            fric_law             = p['fric_law'],
            nx = nx, ny = ny, nt = nt, dt = dt,
            wall_time_s          = wall,
            mean_step_compute_ms = float(step_time_compute.mean() * 1e3),
            mean_step_total_ms   = float(step_time_total.mean()   * 1e3),
            numba_threads        = numba.get_num_threads(),
        ), default=str),
    )

    print(f"Saved: {out_file}")
    print(f"Saved: {timing_file}")
    return out_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='2D dynamic rupture simulation (SBP-SAT, RK4, Numba)')
    parser.add_argument('infile', help='Path to .in parameter file')
    parser.add_argument('-np', '--np', type=int, default=numba.config.NUMBA_NUM_THREADS,
                        help='Number of Numba threads (default: all available)')
    args = parser.parse_args()

    if not os.path.isfile(args.infile):
        print(f"ERROR: input file not found: {args.infile}")
        sys.exit(1)

    run_sim(args.infile, args.np)


if __name__ == '__main__':
    main()
