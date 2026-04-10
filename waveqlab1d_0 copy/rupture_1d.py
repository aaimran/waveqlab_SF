#!/usr/bin/env python3
"""
1D Dynamic Rupture Simulation Runner — Numba version
=====================================================
Usage:
    python3 rupture_1d.py input/rupture_1d_SW.in          # serial (1 thread)
    python3 rupture_1d.py input/rupture_1d_SW.in -np 4    # 4 threads

Reads all parameters from the .in file (key = value, # comments).
Saves results to:
    output/{output_prefix}_{run_id}.npz
    output/{output_prefix}_{run_id}_timing.npz

Numba differences vs 1d_serial:
  - All state arrays are flat (nx,) — not (nx, 1) column vectors
  - friction_params is np.float64[12] — not a Python list with string flag
  - The full RK4 step is compiled by numba on the first call (~1-2 s warmup)
  - -np N controls the numba thread count for interior stencil parallelism
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
    import kernels
    from kernels import (rk4_step, rk4_step_anelastic, FRIC_SW, FRIC_RS,
                          rk4_step_pml_dk,
                          compute_energy, compute_energy_variable)
except Exception:
    print("ERROR: Could not import kernels from src/kernels.py.")
    raise

# ---------------------------------------------------------------------------
# Default parameter values (used when key is absent from .in file)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Output
    output_prefix  = 'rupture_run',
    w_stride          = 5,
    # Domain
    L              = 30.0,
    nx             = 501,
    # Material
    cs             = 3.464,
    rho            = 2.67,
    # Heterogeneous material profiles (optional file paths)
    rho_file       = None,
    cs_file        = None,
    # Spatially variable Q profile (overrides c*V_S when provided)
    Qs_inv_file    = None,
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
    # Response: 'elastic' or 'anelastic'
    response       = 'elastic',
    # GSLS anelastic parameters (used when response='anelastic')
    c              = 1.0,
    weight_exp     = 0.0,
    fref           = 1.0,
    # PML absorbing boundaries
    pml            = False,
    pml_dk         = False,   # True → Duru–Kreiss provably stable PML
    npml           = 20,
    pml_alpha      = 10.0,
    # Anelastic snapshot stride (0 = skip saving eta arrays)
    iplot_eta      = 0,
)

VALID_FRIC_LAWS = ('SW', 'RS')
VALID_ORDERS    = (2, 4, 6)
VALID_BC_VALS   = (-1, 0, 1)

# Caveat 3: per-order CFL stability limits for classical RK4 + SBP operators.
# Derived from the RK4 imaginary-axis stability bound |lambda*dt| <= 2*sqrt(2)
# and the spectral radius rho(D) ~ c / (h11 * dx).
_CFL_MAX = {2: 1.00, 4: 0.68, 6: 0.45}


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

    # Keys that stay as strings
    _str_keys = {'fric_law', 'output_prefix', 'response',
                 'rho_file', 'cs_file', 'Qs_inv_file'}
    # Keys that are boolean flags
    _bool_keys = {'pml', 'pml_dk'}

    params = {}
    for k, v in raw.items():
        if k in _str_keys:
            params[k] = v.upper() if k == 'fric_law' else v
            continue
        if k in _bool_keys:
            params[k] = v.strip().lower() in ('true', '1', 'yes')
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

    p['nx']       = int(p['nx'])
    p['dx']       = float(p['L']) / (p['nx'] - 1)
    if 'dt' not in p:
        p['dt']   = (float(p['cfl']) / float(p['cs'])) * p['dx']
    p['dt']       = float(p['dt'])
    p['nt']       = int(round(float(p['tend']) / p['dt']))
    p['response'] = str(p.get('response', 'elastic')).lower()
    p['pml']      = bool(p.get('pml', False))
    p['pml_dk']   = bool(p.get('pml_dk', False))
    p['npml']     = int(p.get('npml', 20))
    p['iplot_eta']= int(p.get('iplot_eta', 0))

    if p['Tau_0'] is None:
        p['Tau_0'] = 81.24 + 1.0 * 0.36 if p['fric_law'] == 'SW' else 81.24 + 0.1 * 0.36

    return p


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(p):
    errors = []

    if p['response'] not in ('elastic', 'anelastic'):
        errors.append(f"response='{p['response']}' is not valid. Choose 'elastic' or 'anelastic'")

    if p['fric_law'] not in VALID_FRIC_LAWS:
        errors.append(f"fric_law='{p['fric_law']}' is not valid. Choose from: {VALID_FRIC_LAWS}")

    if int(p['order']) not in VALID_ORDERS:
        errors.append(f"order={p['order']} is not valid. Choose from: {VALID_ORDERS}")

    # Caveat 3: enforce order-dependent CFL stability limit
    if not (0 < float(p['cfl']) <= 1.0):
        errors.append(f"cfl={p['cfl']} must be in (0, 1]")
    else:
        cfl_max = _CFL_MAX.get(int(p.get('order', 6)), 1.0)
        if float(p['cfl']) > cfl_max:
            errors.append(
                f"cfl={p['cfl']} exceeds the RK4+SBP stability limit "
                f"{cfl_max} for order={p['order']}. "
                f"Reduce cfl or use a lower-order scheme.")

    # Caveat 2: warn if legacy PML selected without Duru–Kreiss flag
    if p.get('pml') and not p.get('pml_dk'):
        import warnings
        warnings.warn(
            "pml=True uses the legacy volume-damping PML which lies outside "
            "the SBP energy-stability framework.  Set pml_dk=True to use the "
            "provably stable Duru–Kreiss formulation, or use SAT absorbing BCs "
            "(r0_l=0, r1_r=0) for certified stability.",
            stacklevel=3)

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

    # Flat (nx,) arrays — no column-vector axis
    y_l = np.array([j * dx     for j in range(nx)], dtype=np.float64)
    y_r = np.array([L + j * dx for j in range(nx)], dtype=np.float64)

    v_l  = np.zeros(nx, dtype=np.float64)
    s_l  = np.zeros(nx, dtype=np.float64)
    v_r  = np.zeros(nx, dtype=np.float64)
    s_r  = np.zeros(nx, dtype=np.float64)
    slip = np.array([float(p['slip_init'])], dtype=np.float64)   # shape (1,)
    psi  = np.zeros(1, dtype=np.float64)

    if p['fric_law'] == 'RS':
        psi[0] = float(p['psi_init'])

    return y_l, y_r, v_l, s_l, v_r, s_r, slip, psi


def build_friction_parameters(p):
    """
    Build the float64[12] friction_params array consumed by numba kernels.

    Index layout (matches kernels.py docstring):
      [0]  fric_flag   0.0 = SW, 1.0 = RS
      [1]  alpha
      [2]  Tau_0
      [3]  L0       [4] f0     [5] a      [6] b
      [7]  V0       [8] sigma_n
      [9]  alp_s    [10] alp_d [11] D_c
    """
    fric_flag = FRIC_SW if p['fric_law'] == 'SW' else FRIC_RS
    alpha     = np.float64(1e308)   # effectively locked at t=0
    return np.array([
        fric_flag, alpha, p['Tau_0'],
        p['L0'],   p['f0'], p['a'],   p['b'],
        p['V0'],   p['sigma_n'],
        p['alp_s'], p['alp_d'], p['D_c'],
    ], dtype=np.float64)



# ---------------------------------------------------------------------------
# Run ID: short hash of all parameters for unique, reproducible filenames
# ---------------------------------------------------------------------------

def make_run_id(p):
    """Return '{param_hash}_{NNN}' where param_hash is stable across runs with
    identical physics parameters and NNN is a per-hash run counter (001, 002, …)
    stored in output/.run_counter_{param_hash}.txt."""
    skip = {'dx', 'dt', 'nt', 'output_prefix', 'w_stride'}
    core = {k: v for k, v in sorted(p.items()) if k not in skip}
    blob = json.dumps(core, sort_keys=True, default=str).encode()
    param_hash = hashlib.sha256(blob).hexdigest()[:8]

    out_dir = os.path.join(_ROOT, 'output')
    os.makedirs(out_dir, exist_ok=True)
    counter_file = os.path.join(out_dir, f'.run_counter_{param_hash}.txt')

    if os.path.isfile(counter_file):
        with open(counter_file) as f:
            count = int(f.read().strip()) + 1
    else:
        count = 1
    with open(counter_file, 'w') as f:
        f.write(str(count))

    return f"{param_hash}_{count:03d}"


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_sim(infile, nthreads=1):
    import numba
    numba.set_num_threads(nthreads)

    user_params = parse_infile(infile)
    p = build_params(user_params)
    validate(p)

    run_id  = make_run_id(p)
    out_dir = os.path.join(_ROOT, 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{p['output_prefix']}_{run_id}.npz")

    print(f"Run ID        : {run_id}")
    print(f"fric_law      : {p['fric_law']}")
    print(f"response      : {p['response']}")
    print(f"nx={p['nx']}  nt={p['nt']}  dt={p['dt']:.6f} s  tend={p['tend']} s")
    print(f"order={p['order']}  cfl={p['cfl']}  L={p['L']} km")
    print(f"Numba threads : {nthreads}")
    print(f"Output        : {out_file}\n")

    mu  = float(p['rho']) * float(p['cs']) ** 2
    rho = float(p['rho'])
    nx  = p['nx']
    dx  = p['dx']

    # Build material arrays — uniform by default, loaded from file if provided
    rho_arr = np.loadtxt(p['rho_file']).ravel() if p.get('rho_file') else np.full(nx, rho)
    if p.get('cs_file'):
        cs_arr = np.loadtxt(p['cs_file']).ravel()
        mu_arr = rho_arr * cs_arr ** 2
    else:
        mu_arr = np.full(nx, mu)

    # Anelastic (GSLS) initialisation
    if p['response'] == 'anelastic':
        import sys as _sys
        _sys.path.insert(0, os.path.join(_ROOT, 'src'))
        from anelastic import init_anelastic
        print("Initialising GSLS anelastic properties ...")
        Qs_inv_arr = None
        if p.get('Qs_inv_file'):
            Qs_inv_arr = np.loadtxt(p['Qs_inv_file']).ravel()
            if len(Qs_inv_arr) != nx:
                raise ValueError(f"Qs_inv_file must have {nx} values, got {len(Qs_inv_arr)}")
            print(f"  Qs_inv loaded from {p['Qs_inv_file']}")
        anel   = init_anelastic(nx, mu_arr, rho_arr,
                                c=float(p['c']),
                                weight_exp=float(p['weight_exp']),
                                fref=float(p['fref']),
                                Qs_inv_arr=Qs_inv_arr)
        mu_arr = anel['mu_unrelax']   # replace with dispersed unrelaxed modulus
        Qs_inv = anel['Qs_inv']
        tau    = anel['tau']
        weight = anel['weight']
        eta_l  = np.zeros((nx, 4), dtype=np.float64)   # GSLS memory variables
        eta_r  = np.zeros((nx, 4), dtype=np.float64)
        print(f"  Q_S (median)  = {1.0/float(np.median(Qs_inv)):.1f}")
        print(f"  tau           = {tau}")
        print(f"  weight        = {weight}")
        print()
    else:
        Qs_inv = eta_l = eta_r = tau = weight = None

    # PML absorbing boundaries (applies to both elastic and anelastic)
    if p['pml'] or p['pml_dk']:
        import sys as _sys
        _sys.path.insert(0, os.path.join(_ROOT, 'src'))
        cs_pml = float(np.sqrt(mu_arr[0] / rho_arr[0]))
        if p['pml_dk']:
            from pml import init_pml_dk as _init_pml_fn
            print(f"PML active (Duru–Kreiss): npml={p['npml']}  pml_alpha={p['pml_alpha']}")
        else:
            from pml import init_pml as _init_pml_fn
            print(f"PML active (legacy): npml={p['npml']}  pml_alpha={p['pml_alpha']}  "
                  f"R={float(np.exp(-2.0*(p['pml_alpha']/3.0))):.4f}")
        d_l, d_r = _init_pml_fn(nx, int(p['npml']), cs_pml, dx, float(p['pml_alpha']))
    else:
        d_l = np.zeros(nx, dtype=np.float64)
        d_r = np.zeros(nx, dtype=np.float64)

    # Duru–Kreiss auxiliary variables s_tilde (zero initial condition)
    st_l = np.zeros(nx, dtype=np.float64)
    st_r = np.zeros(nx, dtype=np.float64)

    y_l, y_r, v_l, s_l, v_r, s_r, slip, psi = init_fields(p)
    friction_params = build_friction_parameters(p)
    Tau_0 = float(p['Tau_0'])

    nt    = p['nt']
    dt    = p['dt']
    order = int(p['order'])
    r0_l  = float(p['r0_l'])
    r1_r  = float(p['r1_r'])

    # Phase 5: JIT warm-up — compile before the timed loop
    print("Compiling numba kernels (one-time JIT warmup) ...")
    t_warmup = timeit.default_timer()
    _vl = v_l.copy(); _sl = s_l.copy()
    _vr = v_r.copy(); _sr = s_r.copy()
    _slip = slip.copy(); _psi = psi.copy()
    if p['response'] == 'anelastic':
        _el = eta_l.copy(); _er = eta_r.copy()
        rk4_step_anelastic(_vl, _sl, _el, _vr, _sr, _er, _slip, _psi,
                           rho_arr, mu_arr, Qs_inv, tau, weight,
                           d_l, d_r, nx, dx, order, r0_l, r1_r, dt, friction_params)
    else:
        rk4_step(_vl, _sl, _vr, _sr, _slip, _psi,
                 rho, mu, nx, dx, order, r0_l, r1_r,
                 d_l, d_r, dt, friction_params)
    warmup_s = timeit.default_timer() - t_warmup
    print(f"  Warmup complete in {warmup_s:.2f} s\n")

    # Pre-allocate output storage (float32 to halve memory for PINO datasets)
    w_stride  = max(1, int(p['w_stride']))
    nsnaps = (nt + w_stride - 1) // w_stride          # number of stored snapshots
    DomainOutput_l = np.zeros((nx, nsnaps, 2), dtype=np.float32)
    DomainOutput_r = np.zeros((nx, nsnaps, 2), dtype=np.float32)
    snap_times     = np.empty(nsnaps, dtype=np.float64)
    isnap          = 0

    # Energy monitor storage (one value per snapshot, Caveat 1 fix)
    energy_snaps = np.empty(nsnaps, dtype=np.float64)

    traction_vector = np.empty(nt,     dtype=np.float64)
    sliprate_vector = np.empty(nt,     dtype=np.float64)
    time_vector     = np.empty(nt,     dtype=np.float64)
    slip_vector     = np.empty(nt + 1, dtype=np.float64)
    slip_vector[0]  = float(p['slip_init'])

    # Per-step timing
    step_time_compute = np.empty(nt, dtype=np.float64)  # rk4_step kernel only
    step_time_total   = np.empty(nt, dtype=np.float64)  # kernel + output writing

    start = timeit.default_timer()

    for n in range(nt):
        t0_compute = timeit.default_timer()

        if p['response'] == 'anelastic':
            rk4_step_anelastic(v_l, s_l, eta_l, v_r, s_r, eta_r, slip, psi,
                               rho_arr, mu_arr, Qs_inv, tau, weight,
                               d_l, d_r, nx, dx, order, r0_l, r1_r, dt, friction_params)
        elif p['pml_dk']:
            rk4_step_pml_dk(v_l, s_l, st_l, v_r, s_r, st_r, slip, psi,
                            rho, mu, nx, dx, order, r0_l, r1_r,
                            d_l, d_r, dt, friction_params)
        else:
            rk4_step(v_l, s_l, v_r, s_r, slip, psi,
                     rho, mu, nx, dx, order, r0_l, r1_r,
                     d_l, d_r, dt, friction_params)

        step_time_compute[n] = timeit.default_timer() - t0_compute

        traction_vector[n]  = Tau_0 + s_l[nx - 1]
        slip_vector[n + 1]  = slip[0]
        time_vector[n]      = (n + 1) * dt
        sliprate_vector[n]  = abs(v_r[0] - v_l[nx - 1])

        if (n + 1) % w_stride == 0:
            DomainOutput_l[:, isnap, 0] = v_l
            DomainOutput_l[:, isnap, 1] = s_l
            DomainOutput_r[:, isnap, 0] = v_r
            DomainOutput_r[:, isnap, 1] = s_r
            snap_times[isnap] = (n + 1) * dt
            # Caveat 1: compute and store augmented energy
            if p['response'] == 'anelastic':
                _, _, E = compute_energy_variable(
                    v_l, s_l, v_r, s_r, rho_arr, mu_arr, nx, dx, order,
                    eta_l=eta_l, eta_r=eta_r, Qs_inv=Qs_inv, tau=tau, weight=weight)
            else:
                _, _, E = compute_energy(
                    v_l, s_l, v_r, s_r, rho, mu, nx, dx, order)
            energy_snaps[isnap] = E
            isnap += 1

        step_time_total[n] = timeit.default_timer() - t0_compute

        if (n + 1) % max(1, nt // 10) == 0 or (n + 1) == nt:
            print(f"  step {n+1:>6d}/{nt}  t={time_vector[n]:.4f}s  "
                  f"slip={slip_vector[n+1]:.4f}m  sliprate={sliprate_vector[n]:.4f}m/s")

    wall = timeit.default_timer() - start
    print(f"\ntotal simulation time  = {wall:.2f} s  (excl. {warmup_s:.2f} s JIT warmup)")
    print(f"mean step (compute)    = {step_time_compute.mean()*1e3:.3f} ms")
    print(f"mean step (total)      = {step_time_total.mean()*1e3:.3f} ms")
    print(f"spatial order          = {order}")
    print(f"number of grid points  = {nx}")
    print(f"numba threads          = {nthreads}")

    # ------------------------------------------------------------------
    # PINO metadata
    # ------------------------------------------------------------------
    metadata = dict(p)
    metadata['run_id']               = run_id
    metadata['wall_time_s']          = wall
    metadata['warmup_time_s']        = warmup_s
    metadata['nthreads']             = nthreads
    metadata['mean_step_compute_ms'] = float(step_time_compute.mean() * 1e3)
    metadata['mean_step_total_ms']   = float(step_time_total.mean()   * 1e3)
    metadata['Tau_0']                = Tau_0
    metadata['friction_params']      = friction_params.tolist()

    # Core fields always saved
    save_fields = dict(
        DomainOutput_l  = DomainOutput_l,
        DomainOutput_r  = DomainOutput_r,
        y_l             = y_l,
        y_r             = y_r,
        time            = time_vector,
        slip            = slip_vector,
        sliprate        = sliprate_vector,
        traction        = traction_vector,
        Tau_0           = np.float64(Tau_0),
        energy          = energy_snaps,   # augmented SBP energy at each snapshot
        snap_times      = snap_times,
        metadata        = json.dumps(metadata, default=str),
    )

    # Extra fields for anelastic run
    if p['response'] == 'anelastic':
        save_fields['Qs_inv']       = Qs_inv
        save_fields['tau']          = tau
        save_fields['weight']       = weight
        save_fields['eta_l_final']  = eta_l
        save_fields['eta_r_final']  = eta_r

    # Extra fields for Duru–Kreiss PML run
    if p['pml_dk']:
        save_fields['st_l_final'] = st_l
        save_fields['st_r_final'] = st_r

    np.savez_compressed(out_file, **save_fields)

    timing_file = os.path.join(out_dir, f"{p['output_prefix']}_{run_id}_timing.npz")
    timing_metadata = dict(
        run_id               = run_id,
        output_prefix        = p['output_prefix'],
        fric_law             = p['fric_law'],
        nx                   = nx,
        nt                   = nt,
        dt                   = dt,
        nthreads             = nthreads,
        wall_time_s          = wall,
        warmup_time_s        = warmup_s,
        mean_step_compute_ms = float(step_time_compute.mean() * 1e3),
        mean_step_total_ms   = float(step_time_total.mean()   * 1e3),
        overhead_per_step_ms = float((step_time_total - step_time_compute).mean() * 1e3),
    )
    np.savez_compressed(
        timing_file,
        step_time_compute  = step_time_compute,
        step_time_total    = step_time_total,
        step_time_overhead = step_time_total - step_time_compute,
        metadata           = json.dumps(timing_metadata, default=str),
    )

    print(f"Saved: {out_file}")
    print(f"Saved: {timing_file}")
    return out_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='1D rupture simulation (Numba) — reads .in file, saves .npz to output/')
    parser.add_argument('infile',
                        help='path to .in parameter file (e.g. input/rupture_1d_SW.in)')
    parser.add_argument('-np', '--nthreads', type=int, default=1,
                        help='number of threads for numba parallel kernels (default: 1)')
    args = parser.parse_args()

    if not os.path.isfile(args.infile):
        print(f"ERROR: Input file not found: {args.infile}")
        sys.exit(1)

    run_sim(args.infile, nthreads=args.nthreads)


if __name__ == '__main__':
    main()
