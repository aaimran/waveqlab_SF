#!/usr/bin/env python3
"""
2D Dynamic Rupture Simulation Runner
=====================================
Usage:
    python3 rupture_2d.py input/rupture_2d_SW.in

Reads all parameters from the .in file (key = value, # comments).
Saves results to:
    output/{output_prefix}_{run_id}.npz

Where run_id is a short hash of the full parameter set, ensuring unique
filenames and full reproducibility for PINO training data generation.
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

try:
    import RK4_2D
except Exception:
    print("ERROR: Could not import RK4_2D from src/.")
    raise

# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Output
    output_prefix  = 'rupture_2d',
    iplot          = 5,            # snapshot stride (store every iplot steps)
    # Domain
    Lx             = 10.0,         # x half-domain length (km)
    Ly             = 20.0,         # y domain length along fault (km)
    nx             = 26,           # grid points in x per domain
    ny             = 51,           # grid points in y
    # Material
    cs             = 3.464,        # shear wave speed (km/s)
    cp             = 6.0,          # P-wave speed (km/s)
    rho            = 2.6702,       # density (g/cm^3)
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
    # 0=absorbing, 1=free-surface, -1=clamped
    r_l_0          = 0.,   r_l_1 = 0.,   r_l_2 = 1.,   r_l_3 = 0.,
    r_r_0          = 0.,   r_r_1 = 0.,   r_r_2 = 1.,   r_r_3 = 0.,
    # SW parameters
    Tau_0_SW       = 70.0,         # background traction; patch gets +11.6 added
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
# Parsing
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


def build_params(user):
    """Merge user values over defaults; derive grid/time quantities."""
    p = dict(DEFAULTS)
    p.update(user)

    p['nx']  = int(p['nx'])
    p['ny']  = int(p['ny'])
    p['fric_law'] = p['fric_law'].upper()
    p['mode']     = p['mode'].upper()

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
    p['mu']     = mu
    p['Lambda'] = Lambda

    return p


# ---------------------------------------------------------------------------
# Validation
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
    """Build material matrices, coordinate grids."""
    nx  = p['nx']
    ny  = p['ny']
    dx  = p['dx']
    dy  = p['dy']
    Lx  = float(p['Lx'])
    rho = float(p['rho'])
    mu  = p['mu']
    L   = p['Lambda']

    Mat_l = np.zeros((nx, ny, 3))
    Mat_r = np.zeros((nx, ny, 3))
    Mat_l[:, :, 0] = rho;  Mat_l[:, :, 1] = L;  Mat_l[:, :, 2] = mu
    Mat_r[:, :, 0] = rho;  Mat_r[:, :, 1] = L;  Mat_r[:, :, 2] = mu

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

    return Mat_l, Mat_r, X_l, Y_l, X_r, Y_r, Y_fault


def init_fault_state(p, Y_fault):
    """Initialise slip, psi and friction_parameters arrays."""
    ny       = p['ny']
    fric_law = p['fric_law']
    alpha    = 1e308          # effectively infinite initial friction

    slip     = np.zeros((ny, 1))
    slip_new = np.zeros((ny, 1))
    psi      = np.zeros((ny, 1))
    psi_new  = np.zeros((ny, 1))

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
        psi[:, 0] = f0[:, 0] + b[:, 0] * np.log(V0[:, 0] / L0[:, 0] * theta[:, 0])
        alp_s   = np.ones((ny, 1))
        alp_d   = np.ones((ny, 1))
        D_c     = np.ones((ny, 1))

    # Pack friction_parameters: shape (12, ny)
    # [alpha, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]
    friction_parameters = np.zeros((12, ny))
    for j in range(ny):
        friction_parameters[0,  j] = alpha
        friction_parameters[1,  j] = alpha
        friction_parameters[2,  j] = Tau_0[j, 0]
        friction_parameters[3,  j] = L0[j, 0]
        friction_parameters[4,  j] = f0[j, 0]
        friction_parameters[5,  j] = a[j, 0]
        friction_parameters[6,  j] = b[j, 0]
        friction_parameters[7,  j] = V0[j, 0]
        friction_parameters[8,  j] = sigma_n[j, 0]
        friction_parameters[9,  j] = alp_s[j, 0]
        friction_parameters[10, j] = alp_d[j, 0]
        friction_parameters[11, j] = D_c[j, 0]

    FaultOutput0 = np.zeros((ny, 6))
    FaultOutput0[:, 2] = sigma_n[:, 0]
    FaultOutput0[:, 3] = Tau_0[:, 0]
    FaultOutput0[:, 4] = slip[:, 0]
    FaultOutput0[:, 5] = psi[:, 0]

    return slip, slip_new, psi, psi_new, friction_parameters, FaultOutput0


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def make_run_id(p):
    skip = {'dx', 'dy', 'dt', 'nt', 'nf', 'mu', 'Lambda',
            'output_prefix', 'iplot'}
    core = {k: v for k, v in sorted(p.items()) if k not in skip}
    blob = json.dumps(core, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_sim(infile):
    user_params = parse_infile(infile)
    p = build_params(user_params)
    validate(p)

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

    # source parameters list (matches notebook)
    M = [0, 0, 1., 1., 0]
    source_parameter = [p['x0'], p['y0'], p['t0'], p['T'], p['M0'],
                        p['source_type'], M]

    # boundary reflection coefficient arrays [x_outer, x_fault, y_bottom, y_top]
    r_l = np.array([p['r_l_0'], p['r_l_1'], p['r_l_2'], p['r_l_3']])
    r_r = np.array([p['r_r_0'], p['r_r_1'], p['r_r_2'], p['r_r_3']])

    # initialise domains
    Mat_l, Mat_r, X_l, Y_l, X_r, Y_r, Y_fault = init_domains(p)

    # initialise fields
    F_l    = np.zeros((nx, ny, nf))
    Fnew_l = np.zeros((nx, ny, nf))
    F_r    = np.zeros((nx, ny, nf))
    Fnew_r = np.zeros((nx, ny, nf))

    # initialise fault state and friction
    slip, slip_new, psi, psi_new, friction_parameters, FaultOutput0 = \
        init_fault_state(p, Y_fault)

    # output arrays
    n_snap = (nt + iplot - 1) // iplot   # number of stored domain snapshots
    snap_times = np.empty(n_snap, dtype=np.float64)

    # Fault time series — every time step (small: ny × nt × 6)
    FaultOutput = np.zeros((ny, nt, 6), dtype=np.float32)

    # Domain snapshots (float32 to keep memory manageable)
    DomainOutput_l = np.zeros((nx, ny, n_snap, nf), dtype=np.float32)
    DomainOutput_r = np.zeros((nx, ny, n_snap, nf), dtype=np.float32)

    step_time_compute = np.empty(nt, dtype=np.float64)
    step_time_total   = np.empty(nt, dtype=np.float64)

    snap_idx = 0
    t_start_dt = datetime.now()
    start = timeit.default_timer()
    print(f"Start time  : {t_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    for it in range(nt):
        t = it * dt

        t0_compute = timeit.default_timer()
        RK4_2D.elastic_RK4_2D(
            Fnew_l, F_l, Mat_l, X_l, Y_l, t, nf, nx, ny, dx, dy, dt, order,
            r_l, source_parameter,
            Fnew_r, F_r, Mat_r, X_r, Y_r, r_r,
            friction_parameters, slip, psi, slip_new, psi_new,
            fric_law, FaultOutput0, Y0)
        step_time_compute[it] = timeit.default_timer() - t0_compute

        # update fields in-place
        F_l[:] = Fnew_l
        F_r[:] = Fnew_r
        slip[:] = slip_new
        psi[:]  = psi_new

        FaultOutput[:, it, :] = FaultOutput0

        if it % iplot == 0:
            DomainOutput_l[:, :, snap_idx, :] = F_l
            DomainOutput_r[:, :, snap_idx, :] = F_r
            snap_times[snap_idx] = (it + 1) * dt
            snap_idx += 1

        step_time_total[it] = timeit.default_timer() - t0_compute

        sliprate = np.sqrt(FaultOutput0[:, 0]**2 + FaultOutput0[:, 1]**2).max()
        print(f"  step {it+1:>6d}/{nt}  t={t+dt:.4f}s  "
              f"step_ms={step_time_compute[it]*1e3:.2f}  "
              f"max_sliprate={sliprate:.4f}m/s", flush=True)

    # trim snap arrays if last block was partial
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

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    metadata = dict(p)
    metadata['run_id']               = run_id
    metadata['wall_time_s']          = wall
    metadata['mean_step_compute_ms'] = float(step_time_compute.mean() * 1e3)
    metadata['mean_step_total_ms']   = float(step_time_total.mean()   * 1e3)

    # coordinate arrays
    y_fault = Y_r[0, :]    # y coords at fault (x=0), shape (ny,)
    x_l     = X_l[:, 0]    # x coords of left domain, shape (nx,)
    x_r     = X_r[:, 0]    # x coords of right domain, shape (nx,)

    np.savez_compressed(
        out_file,
        # Domain snapshots: (nx, ny, n_snap, nf) float32
        DomainOutput_l  = DomainOutput_l,
        DomainOutput_r  = DomainOutput_r,
        snap_times      = snap_times.astype(np.float32),
        # Fault time series: (ny, nt, 6) float32
        # axis-2: [vx_slip_rate, vy_slip_rate, normal_traction, shear_traction, slip, psi]
        FaultOutput     = FaultOutput,
        # FNO input field: friction parameters along fault (12, ny) float32
        # rows: [alpha, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]
        friction_parameters = friction_parameters.astype(np.float32),
        # Coordinates
        x_l             = x_l.astype(np.float32),
        x_r             = x_r.astype(np.float32),
        y_fault         = y_fault.astype(np.float32),
        time            = (np.arange(1, nt + 1) * dt).astype(np.float32),
        # Metadata
        metadata        = json.dumps(metadata, default=str),
    )

    # timing file
    timing_file = os.path.join(out_dir, f"{p['output_prefix']}_{run_id}_timing.npz")
    np.savez_compressed(
        timing_file,
        step_time_compute = step_time_compute,
        step_time_total   = step_time_total,
        step_time_overhead = step_time_total - step_time_compute,
        metadata = json.dumps(dict(
            run_id               = run_id,
            output_prefix        = p['output_prefix'],
            fric_law             = p['fric_law'],
            nx = nx, ny = ny, nt = nt, dt = dt,
            wall_time_s          = wall,
            mean_step_compute_ms = float(step_time_compute.mean() * 1e3),
            mean_step_total_ms   = float(step_time_total.mean()   * 1e3),
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
        description='2D dynamic rupture simulation (SBP-SAT, RK4)')
    parser.add_argument('infile', help='Path to .in parameter file')
    args = parser.parse_args()

    if not os.path.isfile(args.infile):
        print(f"ERROR: input file not found: {args.infile}")
        sys.exit(1)

    run_sim(args.infile)


if __name__ == '__main__':
    main()
