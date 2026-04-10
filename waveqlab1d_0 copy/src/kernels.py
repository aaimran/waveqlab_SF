"""
kernels.py — Numba-JIT kernels for 1D SBP-SAT elastic/anelastic + friction
============================================================================
All state arrays are flat (nx,) or (nx, N_MECH) — no (nx, 1) column vectors.

friction_params: np.float64[12]
  [0]  fric_flag   0.0 = FRIC_SW (slip-weakening)
                   1.0 = FRIC_RS (rate-and-state)
  [1]  alpha       initial penalty coefficient (large number → locked IC)
  [2]  Tau_0       background shear stress [MPa]
  [3]  L0          RS: characteristic slip distance [m]
  [4]  f0          RS: reference friction coefficient
  [5]  a           RS: direct effect parameter
  [6]  b           RS: evolution effect parameter
  [7]  V0          RS: reference slip velocity [m/s]
  [8]  sigma_n     normal stress [MPa]
  [9]  alp_s       SW: static friction coefficient
  [10] alp_d       SW: dynamic friction coefficient
  [11] D_c         SW: critical slip distance [m]

Elastic path  (rk4_step):
  rho, mu       — scalars; uniform material
  Parallelism   — sbp_dx uses prange; other kernels are serial @njit

Anelastic path  (rk4_step_anelastic):
  rho_arr, mu_arr — (nx,) arrays; supports spatially variable material
  eta_l, eta_r    — (nx, 4) memory variables for GSLS (N=4 mechanisms)
  d_l, d_r        — (nx,) PML damping profiles (zeros if PML inactive)
  _anelastic_rhs  — parallel @njit(parallel=True) kernel for memory variables
"""

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Friction law integer flags (stored as float64 in friction_params[0])
# ---------------------------------------------------------------------------
FRIC_SW = np.float64(0.0)
FRIC_RS = np.float64(1.0)


# ============================================================================
# Phase 1 — SBP finite-difference derivative (vectorised + prange interior)
# ============================================================================

@njit(parallel=True, cache=True)
def sbp_dx(ux, u, nx, dx, order):
    """
    In-place SBP first derivative: ux = d(u)/dx.
    Flat arrays (nx,), no column-vector axis.
    Interior loop uses prange → distributes across numba thread pool.
    """
    m = nx - 1
    inv_dx = 1.0 / dx

    if order == 2:
        ux[0] = (u[1] - u[0]) * inv_dx
        ux[m] = (u[m] - u[m - 1]) * inv_dx
        for i in prange(1, m):
            ux[i] = 0.5 * (u[i + 1] - u[i - 1]) * inv_dx

    elif order == 4:
        ux[0] = (-24.0/17.0*u[0] + 59.0/34.0*u[1]
                 - 4.0/17.0*u[2] - 3.0/34.0*u[3]) * inv_dx
        ux[1] = (-0.5*u[0] + 0.5*u[2]) * inv_dx
        ux[2] = (4.0/43.0*u[0] - 59.0/86.0*u[1]
                 + 59.0/86.0*u[3] - 4.0/43.0*u[4]) * inv_dx
        ux[3] = (3.0/98.0*u[0] - 59.0/98.0*u[2]
                 + 32.0/49.0*u[4] - 4.0/49.0*u[5]) * inv_dx

        ux[m]     = (24.0/17.0*u[m] - 59.0/34.0*u[m-1]
                     + 4.0/17.0*u[m-2] + 3.0/34.0*u[m-3]) * inv_dx
        ux[m - 1] = (0.5*u[m] - 0.5*u[m-2]) * inv_dx
        ux[m - 2] = (-4.0/43.0*u[m] + 59.0/86.0*u[m-1]
                     - 59.0/86.0*u[m-3] + 4.0/43.0*u[m-4]) * inv_dx
        ux[m - 3] = (-3.0/98.0*u[m] + 59.0/98.0*u[m-2]
                     - 32.0/49.0*u[m-4] + 4.0/49.0*u[m-5]) * inv_dx

        for i in prange(4, m - 3):
            ux[i] = (0.083333333333333*u[i-2] - 0.666666666666667*u[i-1]
                     + 0.666666666666667*u[i+1] - 0.083333333333333*u[i+2]) * inv_dx

    else:  # order == 6
        ux[0] = inv_dx * (-1.694834962162858*u[0]  + 2.245634824947698*u[1]
                           - 0.055649692295628*u[2] - 0.670383570370653*u[3]
                           - 0.188774952148393*u[4] + 0.552135032829910*u[5]
                           - 0.188126680800077*u[6])
        ux[1] = inv_dx * (-0.434411786832708*u[0]  + 0.107043134706685*u[2]
                           + 0.420172642668695*u[3] + 0.119957288069806*u[4]
                           - 0.328691543801578*u[5] + 0.122487487014485*u[6]
                           - 0.006557221825386*u[7])
        ux[2] = inv_dx * ( 0.063307644169533*u[0]  - 0.629491308812471*u[1]
                           + 0.809935419586724*u[3] - 0.699016381364484*u[4]
                           + 0.850345731199969*u[5] - 0.509589652965290*u[6]
                           + 0.114508548186019*u[7])
        ux[3] = inv_dx * ( 0.110198643174386*u[0]  - 0.357041083340051*u[1]
                           - 0.117033418681039*u[2] + 0.120870009174558*u[4]
                           + 0.349168902725368*u[5] - 0.104924741749615*u[6]
                           - 0.001238311303608*u[7])
        ux[4] = inv_dx * ( 0.133544619364965*u[0]  - 0.438678347579289*u[1]
                           + 0.434686341173840*u[2] - 0.520172867814934*u[3]
                           + 0.049912002176267*u[5] + 0.504693510958978*u[6]
                           - 0.163985258279827*u[7])
        ux[5] = inv_dx * (-0.127754693486067*u[0]  + 0.393149407857401*u[1]
                           - 0.172955234680916*u[2] - 0.491489487857764*u[3]
                           - 0.016325050231672*u[4] + 0.428167552785852*u[6]
                           - 0.025864364383975*u[7] + 0.013071869997141*u[8])
        ux[6] = inv_dx * ( 0.060008241515128*u[0]  - 0.201971348965594*u[1]
                           + 0.142885356631256*u[2] + 0.203603636754774*u[3]
                           - 0.227565385120003*u[4] - 0.590259111130048*u[5]
                           + 0.757462553894374*u[7] - 0.162184436527372*u[8]
                           + 0.018020492947486*u[9])
        ux[7] = inv_dx * ( 0.009910488565285*u[1]  - 0.029429452176588*u[2]
                           + 0.002202493355677*u[3] + 0.067773581604826*u[4]
                           + 0.032681945726690*u[5] - 0.694285851935105*u[6]
                           + 0.743286642396343*u[8] - 0.148657328479269*u[9]
                           + 0.016517480942141*u[10])

        ux[m-7] = inv_dx * (-0.016517480942141*u[m-10] + 0.148657328479269*u[m-9]
                              - 0.743286642396343*u[m-8]  + 0.694285851935105*u[m-6]
                              - 0.032681945726690*u[m-5]  - 0.067773581604826*u[m-4]
                              - 0.002202493355677*u[m-3]  + 0.029429452176588*u[m-2]
                              - 0.009910488565285*u[m-1])
        ux[m-6] = inv_dx * (-0.018020492947486*u[m-9]  + 0.162184436527372*u[m-8]
                              - 0.757462553894374*u[m-7]  + 0.590259111130048*u[m-5]
                              + 0.227565385120003*u[m-4]  - 0.203603636754774*u[m-3]
                              - 0.142885356631256*u[m-2]  + 0.201971348965594*u[m-1]
                              - 0.060008241515128*u[m])
        ux[m-5] = inv_dx * (-0.013071869997141*u[m-8]  + 0.025864364383975*u[m-7]
                              - 0.428167552785852*u[m-6]  + 0.016325050231672*u[m-4]
                              + 0.491489487857764*u[m-3]  + 0.172955234680916*u[m-2]
                              - 0.393149407857401*u[m-1]  + 0.127754693486067*u[m])
        ux[m-4] = inv_dx * ( 0.163985258279827*u[m-7]  - 0.504693510958978*u[m-6]
                              - 0.049912002176267*u[m-5]  + 0.520172867814934*u[m-3]
                              - 0.434686341173840*u[m-2]  + 0.438678347579289*u[m-1]
                              - 0.133544619364965*u[m])
        ux[m-3] = inv_dx * ( 0.001238311303608*u[m-7]  + 0.104924741749615*u[m-6]
                              - 0.349168902725368*u[m-5]  - 0.120870009174558*u[m-4]
                              + 0.117033418681039*u[m-2]  + 0.357041083340051*u[m-1]
                              - 0.110198643174386*u[m])
        ux[m-2] = inv_dx * (-0.114508548186019*u[m-7]  + 0.509589652965290*u[m-6]
                              - 0.850345731199969*u[m-5]  + 0.699016381364484*u[m-4]
                              - 0.809935419586724*u[m-3]  + 0.629491308812471*u[m-1]
                              - 0.063307644169533*u[m])
        ux[m-1] = inv_dx * ( 0.006557221825386*u[m-7]  - 0.122487487014485*u[m-6]
                              + 0.328691543801578*u[m-5]  - 0.119957288069806*u[m-4]
                              - 0.420172642668695*u[m-3]  - 0.107043134706685*u[m-2]
                              + 0.434411786832708*u[m])
        ux[m]   = inv_dx * ( 0.188126680800077*u[m-6]  - 0.552135032829910*u[m-5]
                              + 0.188774952148393*u[m-4]  + 0.670383570370653*u[m-3]
                              + 0.055649692295628*u[m-2]  - 2.245634824947698*u[m-1]
                              + 1.694834962162858*u[m])

        for i in prange(8, m - 7):
            ux[i] = inv_dx * (-0.016666666666667*u[i-3] + 0.15*u[i-2]
                               - 0.75*u[i-1]             + 0.75*u[i+1]
                               - 0.15*u[i+2]             + 0.016666666666667*u[i+3])


# ============================================================================
# Phase 2 — SAT penalty weights
# ============================================================================

@njit(cache=True)
def _penalty_weight(order, dx):
    """h11 * dx: the SBP boundary norm weight for the given order."""
    if order == 2:
        return 0.5 * dx
    elif order == 4:
        return (17.0 / 48.0) * dx
    else:
        return (13649.0 / 43200.0) * dx


# ============================================================================
# Phase 2 — SAT boundary conditions (left and right physical boundaries)
#
# Only applied to:
#   left domain  — left physical boundary  (index 0,   flag r0_l)
#   right domain — right physical boundary (index nx-1, flag r1_r)
# The inner fault-facing boundaries are handled by the interface kernel.
# tau_1 = tau_2 = 1.0 (standard penalty, fixed here).
# ============================================================================

@njit(cache=True)
def _bc_left(hv, hs, v, s, rho, mu, r0, h11):
    """SAT left boundary (index 0).  bcm in the serial code."""
    cs   = math.sqrt(mu / rho)
    zs   = rho * cs
    p    = 0.5 * (zs * v[0] - s[0])   # outgoing minus characteristic
    q    = 0.5 * (zs * v[0] + s[0])   # incoming plus  characteristic
    sat  = p - r0 * q                  # residual (data = 0 for all BC types)
    # penalties: tau_1 = tau_2 = 1.0
    hv[0] -= sat / (rho * h11)
    hs[0] += cs * sat / h11


@njit(cache=True)
def _bc_right(hv, hs, v, s, rho, mu, r1, h11, nx):
    """SAT right boundary (index nx-1).  bcp in the serial code."""
    cs   = math.sqrt(mu / rho)
    zs   = rho * cs
    m    = nx - 1
    p    = 0.5 * (zs * v[m] + s[m])   # outgoing plus  characteristic
    q    = 0.5 * (zs * v[m] - s[m])   # incoming minus characteristic
    sat  = p - r1 * q
    hv[m] -= sat / (rho * h11)
    hs[m] -= cs * sat / h11


# ============================================================================
# Phase 3 — Friction: Slip-Weakening
# ============================================================================

@njit(cache=True)
def _tau_strength_sw(slip_val, friction_params):
    """Fault strength for slip-weakening law."""
    alp_s   = friction_params[9]
    alp_d   = friction_params[10]
    sigma_n = friction_params[8]
    D_c     = friction_params[11]
    s_capped = slip_val if slip_val < D_c else D_c
    coeff = alp_s - (alp_s - alp_d) * s_capped / D_c
    return coeff * sigma_n


@njit(cache=True)
def _slip_weakening_solve(phi, Tau_0, T_str, eta_s):
    """Solve for slip rate and traction given fault strength."""
    Tau_lock = phi + Tau_0
    vv    = (Tau_lock - T_str) / eta_s
    Tau_h = Tau_lock - eta_s * vv   # == T_str
    return Tau_h, vv


# ============================================================================
# Phase 3 — Friction: Rate-and-State (Regula Falsi solver)
# ============================================================================

@njit(cache=True)
def _regula_falsi(V, Phi, eta, sigma_n, psi, V0, a):
    """
    Solve: V + (a*sigma_n/eta) * asinh(V / (2*V0) * exp(psi/a)) = Phi/eta
    using the Regula-Falsi (Illinois variant) method.
    """
    tol   = 1e-12
    Vl    = 0.0
    Vr    = Phi / eta
    maxit = 5000

    coeff       = a * sigma_n / eta
    rhs         = Phi / eta
    exp_psi_a   = math.exp(psi / a)
    inv_2V0     = 0.5 / V0

    fv = V  + coeff * math.asinh(inv_2V0 * V  * exp_psi_a) - rhs
    fl = Vl + coeff * math.asinh(inv_2V0 * Vl * exp_psi_a) - rhs
    fr = Vr + coeff * math.asinh(inv_2V0 * Vr * exp_psi_a) - rhs

    if math.fabs(fv) <= tol:
        return V
    if math.fabs(fl) <= tol:
        return Vl
    if math.fabs(fr) <= tol:
        return Vr

    err = 1.0
    k   = 1
    while k <= maxit and err >= tol:
        V1 = V
        if fv * fl > tol:
            Vl = V
            V  = Vr - (Vr - Vl) * fr / (fr - fl)
            fl = fv
        elif fv * fr > tol:
            Vr = V
            V  = Vr - (Vr - Vl) * fr / (fr - fl)
            fr = fv
        fv  = V + coeff * math.asinh(inv_2V0 * V * exp_psi_a) - rhs
        err = math.fabs(V - V1)
        k  += 1

    return V


@njit(cache=True)
def _rate_and_state_solve(V_init, phi, Tau_0, psi, eta_s, friction_params):
    """Solve for slip rate and traction using rate-and-state friction."""
    sigma_n = friction_params[8]
    V0      = friction_params[7]
    a       = friction_params[5]
    PHI     = phi + Tau_0
    vv      = _regula_falsi(V_init, PHI, eta_s, sigma_n, psi, V0, a)
    Tau_h   = PHI - eta_s * vv
    return Tau_h, vv


# ============================================================================
# Phase 3 — Fault interface coupling + friction + state evolution
# Returns (dslip, dpsi) and updates hv_l, hs_l, hv_r, hs_r in place.
# ============================================================================

@njit(cache=True)
def _interface_kernel(hv_l, hs_l, hv_r, hs_r,
                      v_l, s_l, v_r, s_r,
                      slip, psi,
                      nx, h11, rho, mu,
                      friction_params):
    """
    Fault interface coupling between left domain [nx-1] and right domain [0].
    Uniform material (same rho, mu on both sides).
    Returns (dslip, dpsi) — scalar float64 values.
    """
    cs  = math.sqrt(mu / rho)
    zs  = rho * cs

    # Extract fault-face values
    v0  = v_l[nx - 1];  s0 = s_l[nx - 1]   # left  domain, right face
    v1  = v_r[0];       s1 = s_r[0]          # right domain, left face

    # Outgoing Riemann characteristics
    q_m = zs * v0 - s0   # left outgoing minus
    p_p = zs * v1 + s1   # right outgoing plus

    eta_s = 0.5 * zs                              # half harmonic mean (symmetric)
    Phi   = eta_s * ((p_p / zs) - (q_m / zs))    # stress transfer functional

    fric_flag = friction_params[0]
    Tau_0     = friction_params[2]

    if fric_flag == FRIC_SW:
        Tau_lock = Phi + Tau_0
        T_str    = _tau_strength_sw(slip[0], friction_params)

        if Tau_lock >= T_str:        # slipping
            Tau_h, vv = _slip_weakening_solve(Phi, Tau_0, T_str, eta_s)
            T_m = Tau_h - Tau_0
            T_p = Tau_h - Tau_0
            V_p = (q_m + T_m) / zs + vv
            V_m = (p_p - T_p) / zs - vv
        else:                        # locked
            vv  = 0.0
            T_m = Phi
            T_p = Phi
            V_p = (q_m + Phi) / zs
            V_m = (p_p - Phi) / zs

    else:  # FRIC_RS
        V_init = math.fabs(v1 - v0)
        if V_init > math.fabs(Phi):
            V_init = 0.5 * math.fabs(Phi) / eta_s

        Tau_h, vv = _rate_and_state_solve(V_init, Phi, Tau_0, psi[0], eta_s,
                                          friction_params)
        T_m = Tau_h - Tau_0
        T_p = Tau_h - Tau_0
        V_p = (q_m + T_m) / zs + vv
        V_m = (p_p - T_p) / zs - vv

    # Slip-rate and state variable evolution (aging law — applies to both laws)
    vv_abs  = math.fabs(V_p - V_m)
    dslip   = vv_abs
    b  = friction_params[6]
    V0 = friction_params[7]
    L0 = friction_params[3]
    f0 = friction_params[4]
    dpsi = b * V0 / L0 * math.exp(-(psi[0] - f0) / b) - vv_abs * b / L0

    # SAT interface flux penalties
    inv_h11 = 1.0 / h11

    p0 = 0.5 * (zs * v0 + s0)
    p1 = 0.5 * (zs * V_m + T_m)
    q1 = 0.5 * (zs * v1 - s1)
    q0 = 0.5 * (zs * V_p - T_p)

    hv_l[nx - 1] -= 2.0 / (rho * h11) * (p0 - p1)
    hs_l[nx - 1] -= mu * 2.0 / (zs * h11) * (p0 - p1)
    hv_r[0]      -= 2.0 / (rho * h11) * (q1 - q0)
    hs_r[0]      += 2.0 * mu / (zs * h11) * (q1 - q0)

    return dslip, dpsi


# ============================================================================
# Phase 2+3 — Elastic rate for a single RK4 stage
# ============================================================================

@njit(cache=True)
def _elastic_rate(hv_l, hs_l, v_l, s_l,
                  hv_r, hs_r, v_r, s_r,
                  slip, psi,
                  rho, mu, nx, dx, order, r0_l, r1_r,
                  d_l, d_r,
                  friction_params):
    """
    Compute du/dt rates for one RK4 stage.
    Writes into hv_l, hs_l, hv_r, hs_r in place.
    Returns (dslip, dpsi) from the interface kernel.
    d_l, d_r are (nx,) PML damping profiles; pass zero arrays when inactive.
    """
    h11     = _penalty_weight(order, dx)
    inv_rho = 1.0 / rho

    vx_l = np.empty(nx)
    sx_l = np.empty(nx)
    vx_r = np.empty(nx)
    sx_r = np.empty(nx)

    sbp_dx(vx_l, v_l, nx, dx, order)
    sbp_dx(sx_l, s_l, nx, dx, order)
    sbp_dx(vx_r, v_r, nx, dx, order)
    sbp_dx(sx_r, s_r, nx, dx, order)

    for i in range(nx):
        hv_l[i] = inv_rho * sx_l[i] - d_l[i] * v_l[i]
        hs_l[i] = mu * vx_l[i]      - d_l[i] * s_l[i]
        hv_r[i] = inv_rho * sx_r[i] - d_r[i] * v_r[i]
        hs_r[i] = mu * vx_r[i]      - d_r[i] * s_r[i]

    _bc_left(hv_l, hs_l, v_l, s_l, rho, mu, r0_l, h11)
    _bc_right(hv_r, hs_r, v_r, s_r, rho, mu, r1_r, h11, nx)

    dslip, dpsi = _interface_kernel(hv_l, hs_l, hv_r, hs_r,
                                    v_l, s_l, v_r, s_r,
                                    slip, psi,
                                    nx, h11, rho, mu,
                                    friction_params)
    return dslip, dpsi


# ============================================================================
# Phase 4 — Full RK4 step (in-place update)
# ============================================================================

@njit(cache=True)
def rk4_step(v_l, s_l, v_r, s_r, slip, psi,
             rho, mu, nx, dx, order, r0_l, r1_r,
             d_l, d_r, dt,
             friction_params):
    """
    Classical 4th-order Runge-Kutta step.  Modifies v_l, s_l, v_r, s_r,
    slip, psi in place.

    Parameters
    ----------
    v_l, s_l : float64[nx]   left domain velocity / stress
    v_r, s_r : float64[nx]   right domain velocity / stress
    slip      : float64[1]   fault slip
    psi       : float64[1]   rate-and-state variable
    rho, mu   : float64      material density / shear modulus (uniform)
    nx        : int64        number of grid points
    dx        : float64      grid spacing
    order     : int64        SBP order (2, 4, or 6)
    r0_l      : float64      left BC flag  (1=free, 0=absorbing, -1=clamped)
    r1_r      : float64      right BC flag
    d_l, d_r  : float64[nx]  PML damping profiles (zeros if inactive)
    dt        : float64      time step
    friction_params : float64[12]  see module docstring

    Returns
    -------
    dslip_k4, dpsi_k4 : float64
        Slip-rate and psi-rate from the final (k4) stage, used for on-fault
        diagnostics without extra computation.
    """
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    k1vl = np.empty(nx); k1sl = np.empty(nx)
    k1vr = np.empty(nx); k1sr = np.empty(nx)
    k1slip = np.zeros(1); k1psi = np.zeros(1)

    ds1, dp1 = _elastic_rate(k1vl, k1sl, v_l, s_l,
                              k1vr, k1sr, v_r, s_r,
                              slip, psi,
                              rho, mu, nx, dx, order, r0_l, r1_r,
                              d_l, d_r, friction_params)
    k1slip[0] = ds1
    k1psi[0]  = dp1

    # ── Stage 2  ( y + 0.5*dt*k1 ) ───────────────────────────────────────────
    v2l = v_l + 0.5 * dt * k1vl
    s2l = s_l + 0.5 * dt * k1sl
    v2r = v_r + 0.5 * dt * k1vr
    s2r = s_r + 0.5 * dt * k1sr
    sl2 = slip + 0.5 * dt * k1slip
    ps2 = psi  + 0.5 * dt * k1psi

    k2vl = np.empty(nx); k2sl = np.empty(nx)
    k2vr = np.empty(nx); k2sr = np.empty(nx)
    k2slip = np.zeros(1); k2psi = np.zeros(1)

    ds2, dp2 = _elastic_rate(k2vl, k2sl, v2l, s2l,
                              k2vr, k2sr, v2r, s2r,
                              sl2, ps2,
                              rho, mu, nx, dx, order, r0_l, r1_r,
                              d_l, d_r, friction_params)
    k2slip[0] = ds2
    k2psi[0]  = dp2

    # ── Stage 3  ( y + 0.5*dt*k2 ) ───────────────────────────────────────────
    v3l = v_l + 0.5 * dt * k2vl
    s3l = s_l + 0.5 * dt * k2sl
    v3r = v_r + 0.5 * dt * k2vr
    s3r = s_r + 0.5 * dt * k2sr
    sl3 = slip + 0.5 * dt * k2slip
    ps3 = psi  + 0.5 * dt * k2psi

    k3vl = np.empty(nx); k3sl = np.empty(nx)
    k3vr = np.empty(nx); k3sr = np.empty(nx)
    k3slip = np.zeros(1); k3psi = np.zeros(1)

    ds3, dp3 = _elastic_rate(k3vl, k3sl, v3l, s3l,
                              k3vr, k3sr, v3r, s3r,
                              sl3, ps3,
                              rho, mu, nx, dx, order, r0_l, r1_r,
                              d_l, d_r, friction_params)
    k3slip[0] = ds3
    k3psi[0]  = dp3

    # ── Stage 4  ( y + dt*k3 ) ────────────────────────────────────────────────
    v4l = v_l + dt * k3vl
    s4l = s_l + dt * k3sl
    v4r = v_r + dt * k3vr
    s4r = s_r + dt * k3sr
    sl4 = slip + dt * k3slip
    ps4 = psi  + dt * k3psi

    k4vl = np.empty(nx); k4sl = np.empty(nx)
    k4vr = np.empty(nx); k4sr = np.empty(nx)
    k4slip = np.zeros(1); k4psi = np.zeros(1)

    ds4, dp4 = _elastic_rate(k4vl, k4sl, v4l, s4l,
                              k4vr, k4sr, v4r, s4r,
                              sl4, ps4,
                              rho, mu, nx, dx, order, r0_l, r1_r,
                              d_l, d_r, friction_params)
    k4slip[0] = ds4
    k4psi[0]  = dp4

    # ── Update state in place ─────────────────────────────────────────────────
    c = dt / 6.0
    for i in range(nx):
        v_l[i] += c * (k1vl[i] + 2.0*k2vl[i] + 2.0*k3vl[i] + k4vl[i])
        s_l[i] += c * (k1sl[i] + 2.0*k2sl[i] + 2.0*k3sl[i] + k4sl[i])
        v_r[i] += c * (k1vr[i] + 2.0*k2vr[i] + 2.0*k3vr[i] + k4vr[i])
        s_r[i] += c * (k1sr[i] + 2.0*k2sr[i] + 2.0*k3sr[i] + k4sr[i])

    slip[0] += c * (k1slip[0] + 2.0*k2slip[0] + 2.0*k3slip[0] + k4slip[0])
    psi[0]  += c * (k1psi[0]  + 2.0*k2psi[0]  + 2.0*k3psi[0]  + k4psi[0])

    return ds4, dp4


# ============================================================================
# ── ANELASTIC EXTENSION (Steps 2, 4, 5, 7) ──────────────────────────────────
# ============================================================================

# ============================================================================
# Phase A — GSLS memory-variable RHS (parallel over grid points)
# ============================================================================

@njit(parallel=True, cache=True)
def _anelastic_rhs(hs, Deta, eta, vx, mu_arr, Qs_inv, tau, weight, nx):
    """
    In-place anelastic stress-rate correction and memory-variable ODE RHS.

    For each grid point i and each Zener mechanism l=0..3:

        hs[i]       -= sum_l  eta[i, l]            (subtract current memory vars)
        Deta[i, l]   = (w_l * mu[i] * Qs^{-1}[i] * (dv/dx)[i]
                        - eta[i, l]) / tau[l]       (memory ODE rate)

    Parameters
    ----------
    hs      : (nx,)     stress rate array (in-place modified)
    Deta    : (nx, 4)   memory variable rates (out)
    eta     : (nx, 4)   current memory variable values
    vx      : (nx,)     velocity spatial derivative d(v)/dx
    mu_arr  : (nx,)     (unrelaxed) shear modulus
    Qs_inv  : (nx,)     Q_S^{-1} at each grid point
    tau     : (4,)      GSLS relaxation times
    weight  : (4,)      GSLS quadrature weights
    nx      : int       grid size
    """
    for i in prange(nx):
        vx_i   = vx[i]
        mu_i   = mu_arr[i]
        q_i    = Qs_inv[i]
        eta_sum = 0.0
        for l in range(4):
            forcing    = weight[l] * mu_i * q_i * vx_i
            Deta[i, l] = (forcing - eta[i, l]) / tau[l]
            eta_sum   += eta[i, l]
        hs[i] -= eta_sum


# ============================================================================
# Phase B — Full rate function for one RK stage (elastic + anelastic + PML)
# ============================================================================

@njit(cache=True)
def _anelastic_rates(hv_l, hs_l, Deta_l,
                     hv_r, hs_r, Deta_r,
                     v_l, s_l, eta_l,
                     v_r, s_r, eta_r,
                     slip, psi,
                     rho_arr, mu_arr, Qs_inv,
                     tau, weight,
                     d_l, d_r,
                     nx, dx, order, r0_l, r1_r,
                     friction_params):
    """
    Compute all du/dt rates for one RK stage in the anelastic case.

    Elastic rates  :  hv = (1/rho) * ds/dx  −  d*v   (PML term)
                      hs =  mu * dv/dx       −  d*s   (PML term)
    Anelastic      :  hs -= sum_l eta_l           (memory-variable correction)
                      Deta filled by _anelastic_rhs
    BCs            :  SAT on outer boundaries (scalars at boundary indices)
    Interface      :  friction + Riemann coupling at fault

    Variable material (rho_arr, mu_arr) is supported point-by-point.
    d_l, d_r are zero arrays when PML is inactive.

    Returns (dslip, dpsi) from the fault interface.
    """
    h11 = _penalty_weight(order, dx)

    vx_l = np.empty(nx)
    sx_l = np.empty(nx)
    vx_r = np.empty(nx)
    sx_r = np.empty(nx)

    sbp_dx(vx_l, v_l, nx, dx, order)
    sbp_dx(sx_l, s_l, nx, dx, order)
    sbp_dx(vx_r, v_r, nx, dx, order)
    sbp_dx(sx_r, s_r, nx, dx, order)

    # Elastic + PML damping rates (point-by-point variable material)
    for i in range(nx):
        hv_l[i] = sx_l[i] / rho_arr[i]  - d_l[i] * v_l[i]
        hs_l[i] = mu_arr[i] * vx_l[i]   - d_l[i] * s_l[i]
        hv_r[i] = sx_r[i] / rho_arr[i]  - d_r[i] * v_r[i]
        hs_r[i] = mu_arr[i] * vx_r[i]   - d_r[i] * s_r[i]

    # Anelastic memory-variable contribution (modifies hs, fills Deta)
    _anelastic_rhs(hs_l, Deta_l, eta_l, vx_l, mu_arr, Qs_inv, tau, weight, nx)
    _anelastic_rhs(hs_r, Deta_r, eta_r, vx_r, mu_arr, Qs_inv, tau, weight, nx)

    # SAT boundary conditions (pass scalar values at the boundary indices)
    _bc_left( hv_l, hs_l, v_l, s_l, rho_arr[0],      mu_arr[0],      r0_l, h11)
    _bc_right(hv_r, hs_r, v_r, s_r, rho_arr[nx - 1], mu_arr[nx - 1], r1_r, h11, nx)

    # Fault interface (arithmetic-mean material at fault faces)
    rho_f = 0.5 * (rho_arr[nx - 1] + rho_arr[0])
    mu_f  = 0.5 * (mu_arr[nx - 1]  + mu_arr[0])
    dslip, dpsi = _interface_kernel(hv_l, hs_l, hv_r, hs_r,
                                    v_l, s_l, v_r, s_r,
                                    slip, psi,
                                    nx, h11, rho_f, mu_f,
                                    friction_params)
    return dslip, dpsi


# ============================================================================
# Phase C — Full RK4 step for the anelastic state
#           State: (v_l, s_l, eta_l,  v_r, s_r, eta_r,  slip, psi)
#           Material: rho_arr(nx,), mu_arr(nx,)   [variable or uniform]
#           Anelastic params: Qs_inv(nx,), tau(4,), weight(4,)
#           PML damping:  d_l(nx,), d_r(nx,)      [zeros if inactive]
# ============================================================================

@njit(cache=True)
def rk4_step_anelastic(v_l, s_l, eta_l,
                       v_r, s_r, eta_r,
                       slip, psi,
                       rho_arr, mu_arr,
                       Qs_inv, tau, weight,
                       d_l, d_r,
                       nx, dx, order, r0_l, r1_r, dt,
                       friction_params):
    """
    Classical 4th-order Runge-Kutta step for the anelastic system.

    Modifies v_l, s_l, eta_l, v_r, s_r, eta_r, slip, psi in place.

    Parameters
    ----------
    v_l, s_l    : (nx,)     left-domain velocity / stress
    eta_l       : (nx, 4)   left-domain GSLS memory variables
    v_r, s_r    : (nx,)     right-domain velocity / stress
    eta_r       : (nx, 4)   right-domain GSLS memory variables
    slip        : (1,)      fault cumulative slip
    psi         : (1,)      rate-and-state variable
    rho_arr     : (nx,)     density (uniform: np.full(nx, rho))
    mu_arr      : (nx,)     unrelaxed shear modulus
    Qs_inv      : (nx,)     Q_S^{-1} per grid point
    tau         : (4,)      GSLS relaxation times
    weight      : (4,)      GSLS quadrature weights
    d_l, d_r    : (nx,)     PML damping profiles (zeros if PML off)
    nx          : int       grid points per domain
    dx          : float     grid spacing
    order       : int       SBP order (2, 4, or 6)
    r0_l        : float     left outer BC flag
    r1_r        : float     right outer BC flag
    dt          : float     time step
    friction_params : (12,) friction parameters array

    Returns
    -------
    ds4, dp4 : float64   slip-rate and psi-rate from the final stage
    """
    # ── Stage 1  (rates at current state) ─────────────────────────────────────
    k1vl   = np.empty(nx);      k1sl   = np.empty(nx)
    k1etal = np.zeros((nx, 4))
    k1vr   = np.empty(nx);      k1sr   = np.empty(nx)
    k1etar = np.zeros((nx, 4))
    k1slip = np.zeros(1);       k1psi  = np.zeros(1)

    ds1, dp1 = _anelastic_rates(k1vl, k1sl, k1etal,
                                 k1vr, k1sr, k1etar,
                                 v_l, s_l, eta_l,
                                 v_r, s_r, eta_r,
                                 slip, psi,
                                 rho_arr, mu_arr, Qs_inv, tau, weight,
                                 d_l, d_r,
                                 nx, dx, order, r0_l, r1_r, friction_params)
    k1slip[0] = ds1;  k1psi[0] = dp1

    # ── Stage 2  (y + 0.5*dt*k1) ──────────────────────────────────────────────
    h = 0.5 * dt
    v2l   = v_l   + h * k1vl;    s2l   = s_l   + h * k1sl
    eta2l = eta_l + h * k1etal
    v2r   = v_r   + h * k1vr;    s2r   = s_r   + h * k1sr
    eta2r = eta_r + h * k1etar
    sl2   = slip  + h * k1slip;  ps2   = psi   + h * k1psi

    k2vl   = np.empty(nx);      k2sl   = np.empty(nx)
    k2etal = np.zeros((nx, 4))
    k2vr   = np.empty(nx);      k2sr   = np.empty(nx)
    k2etar = np.zeros((nx, 4))
    k2slip = np.zeros(1);       k2psi  = np.zeros(1)

    ds2, dp2 = _anelastic_rates(k2vl, k2sl, k2etal,
                                 k2vr, k2sr, k2etar,
                                 v2l, s2l, eta2l,
                                 v2r, s2r, eta2r,
                                 sl2, ps2,
                                 rho_arr, mu_arr, Qs_inv, tau, weight,
                                 d_l, d_r,
                                 nx, dx, order, r0_l, r1_r, friction_params)
    k2slip[0] = ds2;  k2psi[0] = dp2

    # ── Stage 3  (y + 0.5*dt*k2) ──────────────────────────────────────────────
    v3l   = v_l   + h * k2vl;    s3l   = s_l   + h * k2sl
    eta3l = eta_l + h * k2etal
    v3r   = v_r   + h * k2vr;    s3r   = s_r   + h * k2sr
    eta3r = eta_r + h * k2etar
    sl3   = slip  + h * k2slip;  ps3   = psi   + h * k2psi

    k3vl   = np.empty(nx);      k3sl   = np.empty(nx)
    k3etal = np.zeros((nx, 4))
    k3vr   = np.empty(nx);      k3sr   = np.empty(nx)
    k3etar = np.zeros((nx, 4))
    k3slip = np.zeros(1);       k3psi  = np.zeros(1)

    ds3, dp3 = _anelastic_rates(k3vl, k3sl, k3etal,
                                 k3vr, k3sr, k3etar,
                                 v3l, s3l, eta3l,
                                 v3r, s3r, eta3r,
                                 sl3, ps3,
                                 rho_arr, mu_arr, Qs_inv, tau, weight,
                                 d_l, d_r,
                                 nx, dx, order, r0_l, r1_r, friction_params)
    k3slip[0] = ds3;  k3psi[0] = dp3

    # ── Stage 4  (y + dt*k3) ──────────────────────────────────────────────────
    v4l   = v_l   + dt * k3vl;   s4l   = s_l   + dt * k3sl
    eta4l = eta_l + dt * k3etal
    v4r   = v_r   + dt * k3vr;   s4r   = s_r   + dt * k3sr
    eta4r = eta_r + dt * k3etar
    sl4   = slip  + dt * k3slip; ps4   = psi   + dt * k3psi

    k4vl   = np.empty(nx);      k4sl   = np.empty(nx)
    k4etal = np.zeros((nx, 4))
    k4vr   = np.empty(nx);      k4sr   = np.empty(nx)
    k4etar = np.zeros((nx, 4))
    k4slip = np.zeros(1);       k4psi  = np.zeros(1)

    ds4, dp4 = _anelastic_rates(k4vl, k4sl, k4etal,
                                 k4vr, k4sr, k4etar,
                                 v4l, s4l, eta4l,
                                 v4r, s4r, eta4r,
                                 sl4, ps4,
                                 rho_arr, mu_arr, Qs_inv, tau, weight,
                                 d_l, d_r,
                                 nx, dx, order, r0_l, r1_r, friction_params)
    k4slip[0] = ds4;  k4psi[0] = dp4

    # ── Update state in place ─────────────────────────────────────────────────
    c = dt / 6.0
    for i in range(nx):
        v_l[i]  += c * (k1vl[i]  + 2.0*k2vl[i]  + 2.0*k3vl[i]  + k4vl[i])
        s_l[i]  += c * (k1sl[i]  + 2.0*k2sl[i]  + 2.0*k3sl[i]  + k4sl[i])
        v_r[i]  += c * (k1vr[i]  + 2.0*k2vr[i]  + 2.0*k3vr[i]  + k4vr[i])
        s_r[i]  += c * (k1sr[i]  + 2.0*k2sr[i]  + 2.0*k3sr[i]  + k4sr[i])
        for l in range(4):
            eta_l[i, l] += c * (k1etal[i, l] + 2.0*k2etal[i, l]
                                + 2.0*k3etal[i, l] + k4etal[i, l])
            eta_r[i, l] += c * (k1etar[i, l] + 2.0*k2etar[i, l]
                                + 2.0*k3etar[i, l] + k4etar[i, l])

    slip[0] += c * (k1slip[0] + 2.0*k2slip[0] + 2.0*k3slip[0] + k4slip[0])
    psi[0]  += c * (k1psi[0]  + 2.0*k2psi[0]  + 2.0*k3psi[0]  + k4psi[0])

    return ds4, dp4


# ============================================================================
# Energy monitor — Caveat 1 fix
#
# Elastic energy norm:
#   E_elastic = 0.5 * sum_i H[i] * (rho * v[i]^2 + s[i]^2 / mu)
#
# Augmented GSLS energy norm (adds weighted memory-variable contribution):
#   E_aug = E_elastic
#         + 0.5 * sum_{l=0}^{3} sum_i H[i] * tau[l]/(weight[l]*mu[i]*Qs_inv[i])
#                                           * eta[i,l]^2
#
# The cross terms between hs and vx that arise when differentiating E_aug
# cancel exactly against the -sum_l eta_l subtracted inside _anelastic_rhs,
# leaving dE_aug/dt <= (boundary dissipation) <= 0 for homogeneous BCs.
#
# Notes
# -----
# * H is the SBP norm: H[i] = h11*dx at the end nodes, dx in the interior.
# * For the elastic-only path pass eta_l=None (all optional arrays None).
# ============================================================================

@njit(cache=True)
def _sbp_norm_weights(nx, dx, order):
    """Return the diagonal SBP norm H as a flat (nx,) array."""
    h11 = _penalty_weight(order, dx) / dx   # dimensionless corner weight
    h = np.empty(nx)
    for i in range(nx):
        h[i] = dx
    h[0]      = h11 * dx
    h[nx - 1] = h11 * dx
    return h


@njit(cache=True)
def compute_energy(v_l, s_l, v_r, s_r,
                   rho, mu, nx, dx, order,
                   eta_l=None, eta_r=None,
                   Qs_inv=None, tau=None, weight=None):
    """
    Compute the discrete SBP energy norm (uniform material).

    Elastic path  : pass eta_l=None.
    Anelastic path: pass eta_l, eta_r, Qs_inv, tau, weight.

    Returns
    -------
    E_l, E_r, E  : float64  domain energies and total E = E_l + E_r
    """
    H      = _sbp_norm_weights(nx, dx, order)
    inv_mu = 1.0 / mu
    E_l    = 0.0
    E_r    = 0.0

    for i in range(nx):
        Hi   = H[i]
        E_l += Hi * (rho * v_l[i] * v_l[i] + inv_mu * s_l[i] * s_l[i])
        E_r += Hi * (rho * v_r[i] * v_r[i] + inv_mu * s_r[i] * s_r[i])

    E_l *= 0.5
    E_r *= 0.5

    # Anelastic augmentation: memory-variable norm
    if eta_l is not None:
        for l in range(4):
            tl_wl = tau[l] / weight[l]   # tau_l / w_l  (per mechanism)
            for i in range(nx):
                Hi    = H[i]
                qi    = Qs_inv[i]
                coeff = 0.5 * tl_wl / (mu * qi) if qi > 0.0 else 0.0
                E_l  += coeff * Hi * eta_l[i, l] * eta_l[i, l]
                E_r  += coeff * Hi * eta_r[i, l] * eta_r[i, l]

    return E_l, E_r, E_l + E_r


@njit(cache=True)
def compute_energy_variable(v_l, s_l, v_r, s_r,
                             rho_arr, mu_arr, nx, dx, order,
                             eta_l=None, eta_r=None,
                             Qs_inv=None, tau=None, weight=None):
    """
    Variable-material version of compute_energy (layered anelastic runs).

    Parameters
    ----------
    rho_arr, mu_arr : (nx,)  density and unrelaxed shear modulus per grid point.
    All other parameters as in compute_energy().
    """
    H   = _sbp_norm_weights(nx, dx, order)
    E_l = 0.0
    E_r = 0.0

    for i in range(nx):
        Hi     = H[i]
        inv_mu = 1.0 / mu_arr[i]
        E_l   += Hi * (rho_arr[i] * v_l[i] * v_l[i] + inv_mu * s_l[i] * s_l[i])
        E_r   += Hi * (rho_arr[i] * v_r[i] * v_r[i] + inv_mu * s_r[i] * s_r[i])

    E_l *= 0.5
    E_r *= 0.5

    if eta_l is not None:
        for l in range(4):
            tl_wl = tau[l] / weight[l]
            for i in range(nx):
                Hi    = H[i]
                qi    = Qs_inv[i]
                mu_i  = mu_arr[i]
                coeff = 0.5 * tl_wl / (mu_i * qi) if qi > 0.0 else 0.0
                E_l  += coeff * Hi * eta_l[i, l] * eta_l[i, l]
                E_r  += coeff * Hi * eta_r[i, l] * eta_r[i, l]

    return E_l, E_r, E_l + E_r


# ============================================================================
# Duru–Kreiss PML — Caveat 2 fix
#
# Modified state per domain: (v, s, s_tilde)
# Modified stress in spatial derivative: s_eff = s + s_tilde
#
# RHS:
#   hv[i] = (1/rho) * d(s_eff)/dx[i]  -  d[i] * v[i]
#   hs[i] =  mu     * d(v)/dx[i]       -  d[i] * s[i]
#   hs_tilde[i] = -d[i] * s_tilde[i]           (no spatial derivative)
#
# The SAT BCs and interface kernel are applied to (v, s) as before; s_tilde
# is zero outside the PML zone so it does not affect physical BCs.
#
# Energy estimate (Theorem 3.1, Duru & Kreiss 2012):
#   E = (rho/2)v^2 + (1/(2*mu))(s+s_tilde)^2 + (1/(2*mu))s_tilde^2
#   dE/dt <= boundary terms <= 0  for d(x)>=0, homogeneous BCs.
# ============================================================================

@njit(cache=True)
def _elastic_rate_dk(hv_l, hs_l, hst_l,
                     hv_r, hs_r, hst_r,
                     v_l, s_l, st_l,
                     v_r, s_r, st_r,
                     slip, psi,
                     rho, mu, nx, dx, order, r0_l, r1_r,
                     d_l, d_r,
                     friction_params):
    """
    Duru–Kreiss RHS for one RK4 stage (elastic, uniform material).

    Extra arrays vs _elastic_rate:
      st_l, st_r  : (nx,) s_tilde (auxiliary PML variable) for each domain
      hst_l, hst_r: (nx,) rates ds_tilde/dt = -d * s_tilde  (output)

    Returns (dslip, dpsi).
    """
    h11     = _penalty_weight(order, dx)
    inv_rho = 1.0 / rho

    # Effective stress: s_eff = s + s_tilde
    seff_l = np.empty(nx)
    seff_r = np.empty(nx)
    for i in range(nx):
        seff_l[i] = s_l[i] + st_l[i]
        seff_r[i] = s_r[i] + st_r[i]

    vx_l  = np.empty(nx);  sex_l = np.empty(nx)
    vx_r  = np.empty(nx);  sex_r = np.empty(nx)

    sbp_dx(vx_l,  v_l,    nx, dx, order)
    sbp_dx(sex_l, seff_l, nx, dx, order)   # d(s + s_tilde)/dx
    sbp_dx(vx_r,  v_r,    nx, dx, order)
    sbp_dx(sex_r, seff_r, nx, dx, order)

    for i in range(nx):
        hv_l[i]  = inv_rho * sex_l[i] - d_l[i] * v_l[i]
        hs_l[i]  = mu * vx_l[i]       - d_l[i] * s_l[i]
        hst_l[i] = -d_l[i] * st_l[i]
        hv_r[i]  = inv_rho * sex_r[i] - d_r[i] * v_r[i]
        hs_r[i]  = mu * vx_r[i]       - d_r[i] * s_r[i]
        hst_r[i] = -d_r[i] * st_r[i]

    # SAT BCs and fault interface act on (v, s) — s_tilde is zero outside PML
    _bc_left( hv_l, hs_l, v_l, s_l, rho, mu, r0_l, h11)
    _bc_right(hv_r, hs_r, v_r, s_r, rho, mu, r1_r, h11, nx)

    dslip, dpsi = _interface_kernel(hv_l, hs_l, hv_r, hs_r,
                                    v_l, s_l, v_r, s_r,
                                    slip, psi,
                                    nx, h11, rho, mu,
                                    friction_params)
    return dslip, dpsi


@njit(cache=True)
def rk4_step_pml_dk(v_l, s_l, st_l,
                    v_r, s_r, st_r,
                    slip, psi,
                    rho, mu, nx, dx, order, r0_l, r1_r,
                    d_l, d_r, dt,
                    friction_params):
    """
    Full RK4 step with Duru–Kreiss PML (elastic, uniform material).

    Modifies v_l, s_l, st_l, v_r, s_r, st_r, slip, psi in place.

    Parameters
    ----------
    st_l, st_r : (nx,)  s_tilde auxiliary PML variables (init to zero)
    d_l, d_r   : (nx,)  PML damping profiles from init_pml_dk()
    All other parameters as in rk4_step().

    Returns ds4, dp4 (final-stage slip-rate and psi-rate).
    """
    # ── Stage 1  ─────────────────────────────────────────────────────────────
    k1vl  = np.empty(nx);  k1sl  = np.empty(nx);  k1stl = np.empty(nx)
    k1vr  = np.empty(nx);  k1sr  = np.empty(nx);  k1str = np.empty(nx)
    k1slip = np.zeros(1);  k1psi = np.zeros(1)

    ds1, dp1 = _elastic_rate_dk(k1vl, k1sl, k1stl,
                                 k1vr, k1sr, k1str,
                                 v_l, s_l, st_l,
                                 v_r, s_r, st_r,
                                 slip, psi,
                                 rho, mu, nx, dx, order, r0_l, r1_r,
                                 d_l, d_r, friction_params)
    k1slip[0] = ds1;  k1psi[0] = dp1

    # ── Stage 2  ─────────────────────────────────────────────────────────────
    h = 0.5 * dt
    v2l   = v_l  + h * k1vl;   s2l  = s_l  + h * k1sl;   st2l  = st_l  + h * k1stl
    v2r   = v_r  + h * k1vr;   s2r  = s_r  + h * k1sr;   st2r  = st_r  + h * k1str
    sl2   = slip + h * k1slip;  ps2  = psi  + h * k1psi

    k2vl  = np.empty(nx);  k2sl  = np.empty(nx);  k2stl = np.empty(nx)
    k2vr  = np.empty(nx);  k2sr  = np.empty(nx);  k2str = np.empty(nx)
    k2slip = np.zeros(1);  k2psi = np.zeros(1)

    ds2, dp2 = _elastic_rate_dk(k2vl, k2sl, k2stl,
                                 k2vr, k2sr, k2str,
                                 v2l, s2l, st2l,
                                 v2r, s2r, st2r,
                                 sl2, ps2,
                                 rho, mu, nx, dx, order, r0_l, r1_r,
                                 d_l, d_r, friction_params)
    k2slip[0] = ds2;  k2psi[0] = dp2

    # ── Stage 3  ─────────────────────────────────────────────────────────────
    v3l   = v_l  + h * k2vl;   s3l  = s_l  + h * k2sl;   st3l  = st_l  + h * k2stl
    v3r   = v_r  + h * k2vr;   s3r  = s_r  + h * k2sr;   st3r  = st_r  + h * k2str
    sl3   = slip + h * k2slip;  ps3  = psi  + h * k2psi

    k3vl  = np.empty(nx);  k3sl  = np.empty(nx);  k3stl = np.empty(nx)
    k3vr  = np.empty(nx);  k3sr  = np.empty(nx);  k3str = np.empty(nx)
    k3slip = np.zeros(1);  k3psi = np.zeros(1)

    ds3, dp3 = _elastic_rate_dk(k3vl, k3sl, k3stl,
                                 k3vr, k3sr, k3str,
                                 v3l, s3l, st3l,
                                 v3r, s3r, st3r,
                                 sl3, ps3,
                                 rho, mu, nx, dx, order, r0_l, r1_r,
                                 d_l, d_r, friction_params)
    k3slip[0] = ds3;  k3psi[0] = dp3

    # ── Stage 4  ─────────────────────────────────────────────────────────────
    v4l   = v_l  + dt * k3vl;  s4l  = s_l  + dt * k3sl;  st4l  = st_l  + dt * k3stl
    v4r   = v_r  + dt * k3vr;  s4r  = s_r  + dt * k3sr;  st4r  = st_r  + dt * k3str
    sl4   = slip + dt * k3slip; ps4  = psi  + dt * k3psi

    k4vl  = np.empty(nx);  k4sl  = np.empty(nx);  k4stl = np.empty(nx)
    k4vr  = np.empty(nx);  k4sr  = np.empty(nx);  k4str = np.empty(nx)
    k4slip = np.zeros(1);  k4psi = np.zeros(1)

    ds4, dp4 = _elastic_rate_dk(k4vl, k4sl, k4stl,
                                 k4vr, k4sr, k4str,
                                 v4l, s4l, st4l,
                                 v4r, s4r, st4r,
                                 sl4, ps4,
                                 rho, mu, nx, dx, order, r0_l, r1_r,
                                 d_l, d_r, friction_params)
    k4slip[0] = ds4;  k4psi[0] = dp4

    # ── Update state in place  ───────────────────────────────────────────────
    c = dt / 6.0
    for i in range(nx):
        v_l[i]  += c * (k1vl[i]  + 2.0*k2vl[i]  + 2.0*k3vl[i]  + k4vl[i])
        s_l[i]  += c * (k1sl[i]  + 2.0*k2sl[i]  + 2.0*k3sl[i]  + k4sl[i])
        st_l[i] += c * (k1stl[i] + 2.0*k2stl[i] + 2.0*k3stl[i] + k4stl[i])
        v_r[i]  += c * (k1vr[i]  + 2.0*k2vr[i]  + 2.0*k3vr[i]  + k4vr[i])
        s_r[i]  += c * (k1sr[i]  + 2.0*k2sr[i]  + 2.0*k3sr[i]  + k4sr[i])
        st_r[i] += c * (k1str[i] + 2.0*k2str[i] + 2.0*k3str[i] + k4str[i])

    slip[0] += c * (k1slip[0] + 2.0*k2slip[0] + 2.0*k3slip[0] + k4slip[0])
    psi[0]  += c * (k1psi[0]  + 2.0*k2psi[0]  + 2.0*k3psi[0]  + k4psi[0])

    return ds4, dp4
