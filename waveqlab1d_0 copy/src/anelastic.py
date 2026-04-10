"""
anelastic.py — GSLS anelastic initialization for waveqlab1d
=============================================================
Implements N=4 Zener-mechanism attenuation for nearly-constant-Q wave
propagation, matching WaveQLab3D (aQ) src/material.f90 exactly.

Reference: anelastic_analysis.md, init_anelastic_properties()

Q model
-------
  Q_S^{-1}(x) = 1 / (c * V_S(x))
  Q_P^{-1}(x) = 0.5 * Q_S^{-1}(x)

In 1D scalar elasticity only one stress component exists (sigma_xy or
sigma_xz), so only the shear Q matters.  Q_P is retained for completeness.

Relaxation times (N=4 mechanisms, log-spaced over [0.08, 15] Hz):
  tau_k = exp( ln(tau_min) + (2k-1)/16 * (ln(tau_max) - ln(tau_min)) )
  tau_min = 1/(2*pi*15) * FAC
  tau_max = 1/(2*pi*0.08) * 200 * FAC     (FAC = 1.0)

Two pre-tabulated weight sets (from WaveQLab3D material.f90):
  weight_exp ≈ 0.0  →  [1.6126, 0.6255, 0.6382, 1.5969]
  weight_exp ≈ 0.6  →  [0.0336, 0.6873, 0.8767, 1.5202]

Unrelaxed modulus correction:
  val_S = sum_l  weight[l] / ((omega_ref^2 * tau[l]^2 + 1) / Q_S^{-1})
  mu_unrelax = rho * V_S^2 / (1 - val_S)

The unrelaxed mu is what appears in the spatial derivative term of the RHS;
the memory variables eta then subtract the relaxation contribution.
"""

import numpy as np

_PI     = np.pi
N_MECH  = 4          # number of Zener mechanisms (hard-coded, matches 3D code)

# Pre-tabulated weight sets from material.f90
_WEIGHTS_0  = np.array([1.6126, 0.6255, 0.6382, 1.5969], dtype=np.float64)
_WEIGHTS_06 = np.array([0.0336, 0.6873, 0.8767, 1.5202], dtype=np.float64)


# ---------------------------------------------------------------------------
# Relaxation times
# ---------------------------------------------------------------------------

def _compute_tau(fac=1.0):
    """
    Log-spaced relaxation times over the [0.08, 15] Hz target band.
    Returns float64 array of shape (4,).
    """
    taumin = fac / (2.0 * _PI * 15.0)
    taumax = fac * 200.0 / (2.0 * _PI * 0.08)
    tau = np.empty(N_MECH, dtype=np.float64)
    log_range = np.log(taumax) - np.log(taumin)
    for k in range(1, N_MECH + 1):
        tau[k - 1] = np.exp(np.log(taumin) + (2.0 * k - 1.0) / 16.0 * log_range)
    return tau


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_anelastic(nx, mu_arr, rho_arr, c=1.0, weight_exp=0.0, fref=1.0,
                   Qs_inv_arr=None):
    """
    Initialize GSLS anelastic attenuation parameters.

    Parameters
    ----------
    nx          : int         number of grid points
    mu_arr      : (nx,)       elastic shear modulus
    rho_arr     : (nx,)       density
    c           : float       Q_S = c * V_S  (used when Qs_inv_arr is None)
    weight_exp  : float       weight-set selector: 0.0 or 0.6
    fref        : float       reference frequency [Hz] for unrelaxed modulus
    Qs_inv_arr  : (nx,) or None
                  If provided, overrides the c*V_S formula pointwise.
                  Useful for layered Q models with independent Q per zone.

    Returns
    -------
    dict
      'Qs_inv'     : (nx,)   pointwise Q_S^{-1}
      'tau'        : (4,)    relaxation times [s] for 4 mechanisms
      'weight'     : (4,)    quadrature weights
      'mu_unrelax' : (nx,)   unrelaxed shear modulus (replaces elastic mu in RHS)
    """
    mu_arr  = np.asarray(mu_arr,  dtype=np.float64)
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    if mu_arr.shape != (nx,) or rho_arr.shape != (nx,):
        raise ValueError("mu_arr and rho_arr must have shape (nx,)")

    # vs_arr needed for mu_unrelax even when Qs_inv is externally supplied
    vs_arr = np.sqrt(mu_arr / rho_arr)

    # Q_S^{-1}: use supplied array or derive from c * V_S
    if Qs_inv_arr is not None:
        Qs_inv = np.asarray(Qs_inv_arr, dtype=np.float64).ravel()
        if Qs_inv.shape != (nx,):
            raise ValueError(f"Qs_inv_arr must have shape ({nx},), got {Qs_inv.shape}")
    else:
        Qs_inv = 1.0 / (c * vs_arr)

    # Relaxation times
    tau = _compute_tau(fac=1.0)

    # Weight set
    if weight_exp < 0.01:
        weight = _WEIGHTS_0.copy()
    elif 0.59 < weight_exp < 0.61:
        weight = _WEIGHTS_06.copy()
    else:
        raise ValueError(
            f"weight_exp={weight_exp}: only 0.0 (default) or 0.6 are supported")

    # Unrelaxed modulus correction (velocity dispersion at fref)
    wref = 2.0 * _PI * fref
    mu_unrelax = np.empty(nx, dtype=np.float64)
    for i in range(nx):
        val_S = 0.0
        for l in range(N_MECH):
            denom = (wref**2 * tau[l]**2 + 1.0) / Qs_inv[i]
            val_S += weight[l] / denom
        mu_unrelax[i] = rho_arr[i] * vs_arr[i]**2 / (1.0 - val_S)

    return {
        'Qs_inv':     Qs_inv,
        'tau':        tau,
        'weight':     weight,
        'mu_unrelax': mu_unrelax,
    }
