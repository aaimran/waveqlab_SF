#!/usr/bin/env python3
"""
1D Dynamic Rupture Simulation Runner
=====================================
Usage:
    python3 rupture_1d.py input/rupture_1d_SW.in

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

import numpy as np

# ---------------------------------------------------------------------------
# Simulation modules live in src/ next to this script
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import time_integrator
    import utils
except Exception:
    print("ERROR: Could not import simulation modules from src/.")
    print("       Make sure src/ contains time_integrator.py, rate.py, utils.py, etc.")
    raise

# ---------------------------------------------------------------------------
# Default parameter values (used when key is absent from .in file)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Output
    output_prefix  = 'rupture_run',
    iplot          = 5,
    # Domain
    L              = 30.0,
    nx             = 501,
    # Material
    cs             = 3.464,
    rho            = 2.67,
    # Time stepping
    cfl            = 0.5,
    tend           = 5.0,
    # Scheme
    order          = 6,
    # Boundary conditions (1=free-surface, 0=absorbing, -1=clamped)
    r0_l           = 1,    r1_l = 1,
    r0_r           = 1,    r1_r = 1,
    # SAT penalty weights
    tau_11_l       = 1,    tau_12_l = 1,
    tau_21_l       = 1,    tau_22_l = 1,
    tau_11_r       = 1,    tau_12_r = 1,
    tau_21_r       = 1,    tau_22_r = 1,
    # Friction
    fric_law       = 'SW',
    # Fault initial conditions
    Tau_0          = None,   # set per fric_law below if not provided
    slip_init      = 0.0,
    # SW parameters
    alp_s          = 0.677,
    alp_d          = 0.525,
    D_c            = 0.4,
    sigma_n        = 120.0,
    # RS parameters
    f0             = 0.6,
    a              = 0.008,
    b              = 0.012,
    V0             = 1.0e-6,
    L0             = 0.02,
    psi_init       = 0.4367,
)

VALID_FRIC_LAWS = ('SW', 'RS')
VALID_ORDERS    = (2, 4, 6)
VALID_BC_VALS   = (-1, 0, 1)


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
        if k == 'fric_law':
            params[k] = v.upper()
            continue
        if k == 'output_prefix':
            params[k] = v
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
    """Merge user values over defaults; derive dx, dt, nt; set Tau_0 if absent."""
    p = dict(DEFAULTS)
    p.update(user)

    p['nx']  = int(p['nx'])
    p['dx']  = float(p['L']) / (p['nx'] - 1)
    if 'dt' not in p:
        p['dt'] = (float(p['cfl']) / float(p['cs'])) * p['dx']
    p['dt']  = float(p['dt'])
    p['nt']  = int(round(float(p['tend']) / p['dt']))

    if p['Tau_0'] is None:
        p['Tau_0'] = 81.24 + 1.0 * 0.36 if p['fric_law'] == 'SW' else 81.24 + 0.1 * 0.36

    return p


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(p):
    errors = []

    if p['fric_law'] not in VALID_FRIC_LAWS:
        errors.append(f"fric_law='{p['fric_law']}' is not valid. Choose from: {VALID_FRIC_LAWS}")

    if int(p['order']) not in VALID_ORDERS:
        errors.append(f"order={p['order']} is not valid. Choose from: {VALID_ORDERS}")

    if not (0 < float(p['cfl']) <= 1.0):
        errors.append(f"cfl={p['cfl']} must be in (0, 1]")

    if p['nx'] < 10:
        errors.append(f"nx={p['nx']} is too small (minimum 10)")

    if float(p['tend']) <= 0:
        errors.append(f"tend={p['tend']} must be positive")

    for bc_key in ('r0_l', 'r1_l', 'r0_r', 'r1_r'):
        if int(p[bc_key]) not in VALID_BC_VALS:
            errors.append(f"{bc_key}={p[bc_key]} must be one of {VALID_BC_VALS}")

    if p['fric_law'] == 'SW':
        if float(p['alp_d']) >= float(p['alp_s']):
            errors.append(
                f"SW requires alp_d ({p['alp_d']}) < alp_s ({p['alp_s']}) for velocity-weakening")
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
# Initialisation
# ---------------------------------------------------------------------------

def init_fields(p):
    nx = p['nx']
    dx = p['dx']
    L  = float(p['L'])

    y_l = np.array([j * dx      for j in range(nx)], dtype=float).reshape(nx, 1)
    y_r = np.array([L + j * dx  for j in range(nx)], dtype=float).reshape(nx, 1)

    v_l  = np.zeros((nx, 1))
    s_l  = np.zeros((nx, 1))
    v_r  = np.zeros((nx, 1))
    s_r  = np.zeros((nx, 1))
    slip = np.full((1, 1), float(p['slip_init']))
    psi  = np.zeros((1, 1))

    if p['fric_law'] == 'RS':
        psi[:, :] = float(p['psi_init'])

    return y_l, y_r, v_l, s_l, v_r, s_r, slip, psi


def build_friction_parameters(p):
    fric_law = p['fric_law']
    alpha = 1e308   # effectively infinite initial friction coefficient
    if fric_law == 'SW':
        return [fric_law, alpha, p['Tau_0'],
                p['L0'], p['f0'], p['a'], p['b'], p['V0'],
                p['sigma_n'], p['alp_s'], p['alp_d'], p['D_c']]
    else:  # RS
        return [fric_law, alpha, p['Tau_0'],
                p['L0'], p['f0'], p['a'], p['b'], p['V0'],
                p['sigma_n'], 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Run ID: short hash of all parameters for unique, reproducible filenames
# ---------------------------------------------------------------------------

def make_run_id(p):
    skip = {'dx', 'dt', 'nt', 'output_prefix', 'iplot'}
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
    print(f"fric_law      : {p['fric_law']}")
    print(f"nx={p['nx']}  nt={p['nt']}  dt={p['dt']:.6f} s  tend={p['tend']} s")
    print(f"order={p['order']}  cfl={p['cfl']}  L={p['L']} km")
    print(f"Output        : {out_file}\n")

    mu  = float(p['rho']) * float(p['cs']) ** 2
    rho = float(p['rho'])

    y_l, y_r, v_l, s_l, v_r, s_r, slip, psi = init_fields(p)
    friction_parameters = build_friction_parameters(p)
    Tau_0 = float(p['Tau_0'])

    nx = p['nx']
    nt = p['nt']
    dx = p['dx']
    dt = p['dt']

    bc = {k: int(p[k]) for k in
          ('r0_l','r1_l','tau_11_l','tau_12_l','tau_21_l','tau_22_l',
           'r0_r','r1_r','tau_11_r','tau_12_r','tau_21_r','tau_22_r')}

    # Pre-allocate as float32 to halve memory for large PINO datasets
    DomainOutput_l = np.zeros((nx, nt + 1, 2), dtype=np.float32)
    DomainOutput_r = np.zeros((nx, nt + 1, 2), dtype=np.float32)

    traction_vector = np.empty(nt,      dtype=np.float64)
    sliprate_vector = np.empty(nt,      dtype=np.float64)
    time_vector     = np.empty(nt,      dtype=np.float64)
    slip_vector     = np.empty(nt + 1,  dtype=np.float64)
    slip_vector[0]  = float(p['slip_init'])

    # Per-step timing
    step_time_compute = np.empty(nt, dtype=np.float64)  # RK4 only
    step_time_total   = np.empty(nt, dtype=np.float64)  # RK4 + output writing

    start = timeit.default_timer()

    for n in range(nt):
        t = n * dt

        t0_compute = timeit.default_timer()
        time_integrator.elastic_RK4(
            v_l, s_l, v_l, s_l, rho, mu, nx, dx, p['order'], y_l, t, dt,
            bc['r0_l'], bc['r1_l'], bc['tau_11_l'], bc['tau_21_l'],
            bc['tau_12_l'], bc['tau_22_l'],
            v_r, s_r, v_r, s_r, rho, mu, nx, dx, p['order'], y_r, t, dt,
            bc['r0_r'], bc['r1_r'], bc['tau_11_r'], bc['tau_21_r'],
            bc['tau_12_r'], bc['tau_22_r'],
            slip, psi, slip, psi, friction_parameters
        )
        step_time_compute[n] = timeit.default_timer() - t0_compute

        traction_vector[n]  = Tau_0 + s_l[-1, 0]
        slip_vector[n + 1]  = slip[0, 0]
        time_vector[n]      = (n + 1) * dt
        sliprate_vector[n]  = np.abs(v_r[0, 0] - v_l[-1, 0])

        DomainOutput_l[:, n, 0] = v_l[:, 0]
        DomainOutput_l[:, n, 1] = s_l[:, 0]
        DomainOutput_r[:, n, 0] = v_r[:, 0]
        DomainOutput_r[:, n, 1] = s_r[:, 0]

        step_time_total[n] = timeit.default_timer() - t0_compute

        if (n + 1) % max(1, nt // 10) == 0 or (n + 1) == nt:
            print(f"  step {n+1:>6d}/{nt}  t={time_vector[n]:.4f}s  "
                  f"slip={slip_vector[n+1]:.4f}m  sliprate={sliprate_vector[n]:.4f}m/s")

    wall = timeit.default_timer() - start
    print(f"\ntotal simulation time  = {wall:.2f} s")
    print(f"mean step (compute)    = {step_time_compute.mean()*1e3:.3f} ms")
    print(f"mean step (total)      = {step_time_total.mean()*1e3:.3f} ms")
    print(f"spatial order          = {p['order']}")
    print(f"number of grid points  = {nx}")

    # ------------------------------------------------------------------
    # PINO metadata: everything needed to reconstruct or label this run
    # ------------------------------------------------------------------
    metadata = dict(p)
    metadata['run_id']              = run_id
    metadata['wall_time_s']         = wall
    metadata['mean_step_compute_ms'] = float(step_time_compute.mean() * 1e3)
    metadata['mean_step_total_ms']   = float(step_time_total.mean()   * 1e3)
    metadata['Tau_0']               = Tau_0
    metadata['friction_parameters'] = friction_parameters

    np.savez_compressed(
        out_file,
        # Full spatio-temporal fields (nx, nt+1, 2): axis-2 = [velocity, stress]
        DomainOutput_l  = DomainOutput_l,
        DomainOutput_r  = DomainOutput_r,
        # Spatial coordinates (nx,)
        y_l             = y_l[:, 0],
        y_r             = y_r[:, 0],
        # On-fault time series
        time            = time_vector,
        slip            = slip_vector,
        sliprate        = sliprate_vector,
        traction        = traction_vector,
        Tau_0           = np.float64(Tau_0),
        # Full run metadata as JSON string
        metadata        = json.dumps(metadata, default=str),
    )

    # ------------------------------------------------------------------
    # Timing file: saved separately so it can be loaded without the
    # large domain fields
    # ------------------------------------------------------------------
    timing_file = os.path.join(out_dir, f"{p['output_prefix']}_{run_id}_timing.npz")
    timing_metadata = dict(
        run_id              = run_id,
        output_prefix       = p['output_prefix'],
        fric_law            = p['fric_law'],
        nx                  = nx,
        nt                  = nt,
        dt                  = dt,
        wall_time_s         = wall,
        mean_step_compute_ms = float(step_time_compute.mean() * 1e3),
        mean_step_total_ms   = float(step_time_total.mean()   * 1e3),
        overhead_per_step_ms = float((step_time_total - step_time_compute).mean() * 1e3),
    )
    np.savez_compressed(
        timing_file,
        # Per-step timings (nt,)  in seconds
        step_time_compute   = step_time_compute,
        step_time_total     = step_time_total,
        # Derived: output-writing overhead per step
        step_time_overhead  = step_time_total - step_time_compute,
        metadata            = json.dumps(timing_metadata, default=str),
    )

    print(f"Saved: {out_file}")
    print(f"Saved: {timing_file}")
    return out_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='1D rupture simulation — reads .in file, saves .npz to output/')
    parser.add_argument('infile',
                        help='path to .in parameter file (e.g. input/rupture_1d_SW.in)')
    args = parser.parse_args()

    if not os.path.isfile(args.infile):
        print(f"ERROR: Input file not found: {args.infile}")
        sys.exit(1)

    run_sim(args.infile)


if __name__ == '__main__':
    main()
