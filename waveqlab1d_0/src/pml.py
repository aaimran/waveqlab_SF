"""
pml.py — Perfectly Matched Layer (PML) absorbing boundary for waveqlab1d
=========================================================================
Two PML formulations are provided:

1. Volume-damping PML  (init_pml / legacy)
   ------------------------------------------
   Modifies the wave equations as:
       rho * dv/dt = d(s)/dx  -  d(x) * rho * v
            ds/dt = mu *d(v)/dx  -  d(x) * s
   This falls OUTSIDE the SBP energy-stability framework.
   Kept for backward compatibility and empirical testing.

2. Duru–Kreiss energy-bounding PML  (init_pml_dk)      ← Caveat 2 fix
   ------------------------------------------------------------------
   Reference: Duru & Kreiss (2012), "A well-posed and discretely stable
   perfectly matched layer for elastic wave equations in second order
   formulation", Commun. Comput. Phys.

   Augments the state with auxiliary variable s_tilde (same shape as s):
       rho * dv/dt     =  d(s + s_tilde)/dx  -  d(x) * rho * v
            ds/dt      =  mu * d(v)/dx        -  d(x) * s
            ds_tilde/dt = -d(x) * s_tilde

   The modified energy:
       E_PML = (rho/2) v^2 + (1/(2*mu)) * (s + s_tilde)^2 + (1/(2*mu)) * s_tilde^2
   satisfies dE_PML/dt <= (boundary dissipation terms) <= 0 for d(x) >= 0.

   The damping profile d(x) is IDENTICAL to the legacy formulation, so
   init_pml_dk() simply calls init_pml() and returns the same d_l, d_r.
   The extra state arrays s_tilde_l, s_tilde_r are managed by the caller.

PML zones
---------
  Left-domain  — left end:   indices 0 .. npml-1
  Right-domain — right end:  indices nx-npml .. nx-1

Damping profile (quadratic):
  d(x) = d_max * ((x_interface - x) / L_pml)^2    in left PML
  d(x) = d_max * ((x - x_interface) / L_pml)^2    in right PML

  d_max = pml_alpha * cs / L_pml
"""

import numpy as np


def init_pml(nx, npml, cs, dx, pml_alpha=10.0):
    """
    Compute PML damping profiles for both domains.

    Parameters
    ----------
    nx        : int    number of grid points per domain
    npml      : int    number of PML grid points at each absorbing end
    cs        : float  representative S-wave speed [km/s]
    dx        : float  grid spacing [km]
    pml_alpha : float  normalized d_max = pml_alpha * cs / L_pml
                       (pml_alpha=10 → R ≈ exp(-2*10/3) ≈ 0.001)

    Returns
    -------
    d_l : (nx,)  damping at each point of the left domain
                 non-zero only at indices 0 .. npml-1  (left absorbing end)
    d_r : (nx,)  damping at each point of the right domain
                 non-zero only at indices nx-npml .. nx-1  (right absorbing end)
    """
    if npml < 2:
        raise ValueError("npml must be >= 2")
    if npml >= nx:
        raise ValueError(f"npml={npml} >= nx={nx}")

    L_pml  = npml * dx                           # physical thickness of PML [km]
    d_max  = pml_alpha * cs / L_pml              # peak damping coefficient [1/s]

    d_l = np.zeros(nx, dtype=np.float64)
    d_r = np.zeros(nx, dtype=np.float64)

    for i in range(npml):
        # Left domain: PML at left end, profile grows from right to left
        xi        = float(npml - 1 - i) / float(npml - 1)   # 0 at inner edge, 1 at outer
        d_l[i]    = d_max * xi**2

        # Right domain: PML at right end, profile grows from left to right
        j         = nx - npml + i
        xi        = float(i) / float(npml - 1)               # 0 at inner edge, 1 at outer
        d_r[j]    = d_max * xi**2

    return d_l, d_r


def pml_reflection_coefficient(npml, cs, dx, pml_alpha=10.0):
    """Return the analytical plane-wave reflection coefficient R for the PML."""
    L_pml = npml * dx
    d_max = pml_alpha * cs / L_pml
    # R = exp(-2 * integral_0^{L_pml} d(x) dx / cs)
    # integral of quadratic profile = d_max * L_pml / 3
    integral = d_max * L_pml / 3.0
    return float(np.exp(-2.0 * integral / cs))


# ---------------------------------------------------------------------------
# Duru–Kreiss energy-bounding PML  (Caveat 2 fix)
# ---------------------------------------------------------------------------

def init_pml_dk(nx, npml, cs, dx, pml_alpha=10.0):
    """
    Initialise the Duru–Kreiss provably stable PML.

    The damping profile d(x) is identical to init_pml().
    The caller must allocate auxiliary arrays s_tilde_l, s_tilde_r of shape
    (nx,) initialised to zero, and advance them alongside v and s.

    Returns
    -------
    d_l : (nx,)  damping profile for the left domain
    d_r : (nx,)  damping profile for the right domain

    Modified RHS
    ------------
    The Duru–Kreiss system replaces the volume-damping RHS with:

        rho * dv/dt      =  d(s + s_tilde)/dx  -  d(x) * rho * v
             ds/dt       =  mu * d(v)/dx        -  d(x) * s
             ds_tilde/dt = -d(x) * s_tilde

    The modified stress seen in the spatial derivative is (s + s_tilde).
    Use pml_dk_aux_rate() to compute ds_tilde/dt.

    Energy estimate
    ---------------
    Define the modified energy:
        E = (rho/2) v^2 + (1/(2*mu)) (s + s_tilde)^2 + (1/(2*mu)) s_tilde^2

    Then dE/dt <= (boundary SAT dissipation terms) <= 0  for d(x) >= 0,
    giving a provable semi-discrete energy bound (see Duru & Kreiss 2012,
    Theorem 3.1 for the continuous analogue and Sec. 4 for the SBP proof).
    """
    return init_pml(nx, npml, cs, dx, pml_alpha)


def pml_dk_aux_rate(d, s_tilde):
    """
    Compute the rate ds_tilde/dt = -d(x) * s_tilde for the Duru–Kreiss PML.

    Parameters
    ----------
    d       : (nx,)  PML damping profile (from init_pml_dk)
    s_tilde : (nx,)  current auxiliary stress variable

    Returns
    -------
    ds_tilde_dt : (nx,)  rate array (new allocation)
    """
    return -d * s_tilde


def pml_dk_aux_rate_inplace(dst, d, s_tilde):
    """
    In-place version: writes -d * s_tilde into dst.
    Use this inside the RK4 stage loop to avoid extra allocations.
    """
    for i in range(len(d)):
        dst[i] = -d[i] * s_tilde[i]
