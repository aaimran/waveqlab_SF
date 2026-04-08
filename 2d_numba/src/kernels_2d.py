"""
kernels_2d.py — Numba-JIT kernels for 2D SBP-SAT elastic-wave + friction simulation
======================================================================================
State layout
  F_l, F_r : float64[nx, ny, nf]   — C-contiguous 3-D arrays
  slip, psi : float64[ny]           — flat (no column-vector shape)
  friction_params : float64[12, ny] — same indexing convention as 2d_serial

Mode II   nf=5  fields: [vx, vy, sxx, syy, sxy]
Mode III  nf=3  fields: [vz, sxz, syz]

friction_params rows (same as 2d_serial friction_parameters):
  [0] alpha     (large initial penalty → locked IC)
  [1] alpha     (duplicate)
  [2] Tau_0     background shear traction [MPa]
  [3] L0        RS: characteristic slip distance / SW: 1.0
  [4] f0        RS: reference friction coefficient / SW: 1.0
  [5] a         RS: direct effect / SW: 1.0
  [6] b         RS: evolution effect / SW: 1.0
  [7] V0        RS: reference slip velocity / SW: 1.0
  [8] sigma_n   normal stress [MPa]  (negative = compressive convention)
  [9] alp_s     SW: static friction / RS: 1.0
  [10] alp_d    SW: dynamic friction / RS: 1.0
  [11] D_c      SW: critical slip / RS: 1.0

Parallelism
  - sbp_dx_2d: prange over interior x-rows (each task: full (ny,nf) strip)
  - sbp_dy_2d: prange over interior y-cols (each task: full (nx,nf) strip)
  - _interface_kernel_2d: prange over fault points j (fully independent)
  - _assemble_rates_2d: prange over i-rows
  - call numba.set_num_threads(N) before first kernel call to control threads
"""

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Friction law flags  (float64 for compatibility with friction_params array)
# ---------------------------------------------------------------------------
FRIC_SW = np.float64(0.0)
FRIC_RS = np.float64(1.0)

# Mode flags  (int64)
MODE_II  = np.int64(2)
MODE_III = np.int64(3)


# ============================================================================
# Phase 1 — SBP first-derivative operators
# ============================================================================

@njit(parallel=True, cache=True)
def sbp_dx_2d(Dxu, u, nx, ny, nf, dx, order):
    """
    In-place SBP x-derivative: Dxu = d(u)/dx.
    u, Dxu: float64[nx, ny, nf]
    Boundary nodes applied as array-wide slice ops.
    Interior uses prange over i.
    """
    m       = nx - 1
    inv_dx  = 1.0 / dx

    if order == 2:
        Dxu[0, :, :]  = (u[1, :, :] - u[0, :, :]) * inv_dx
        Dxu[m, :, :]  = (u[m, :, :] - u[m-1, :, :]) * inv_dx
        for i in prange(1, m):
            Dxu[i, :, :] = 0.5 * (u[i+1, :, :] - u[i-1, :, :]) * inv_dx

    elif order == 4:
        Dxu[0, :, :] = inv_dx * (-24.0/17.0*u[0,:,:] + 59.0/34.0*u[1,:,:]
                                  - 4.0/17.0*u[2,:,:] - 3.0/34.0*u[3,:,:])
        Dxu[1, :, :] = inv_dx * (-0.5*u[0,:,:] + 0.5*u[2,:,:])
        Dxu[2, :, :] = inv_dx * (4.0/43.0*u[0,:,:] - 59.0/86.0*u[1,:,:]
                                  + 59.0/86.0*u[3,:,:] - 4.0/43.0*u[4,:,:])
        Dxu[3, :, :] = inv_dx * (3.0/98.0*u[0,:,:] - 59.0/98.0*u[2,:,:]
                                  + 32.0/49.0*u[4,:,:] - 4.0/49.0*u[5,:,:])

        Dxu[m,   :, :] = inv_dx * (24.0/17.0*u[m,:,:] - 59.0/34.0*u[m-1,:,:]
                                    + 4.0/17.0*u[m-2,:,:] + 3.0/34.0*u[m-3,:,:])
        Dxu[m-1, :, :] = inv_dx * (0.5*u[m,:,:] - 0.5*u[m-2,:,:])
        Dxu[m-2, :, :] = inv_dx * (-4.0/43.0*u[m,:,:] + 59.0/86.0*u[m-1,:,:]
                                    - 59.0/86.0*u[m-3,:,:] + 4.0/43.0*u[m-4,:,:])
        Dxu[m-3, :, :] = inv_dx * (-3.0/98.0*u[m,:,:] + 59.0/98.0*u[m-2,:,:]
                                    - 32.0/49.0*u[m-4,:,:] + 4.0/49.0*u[m-5,:,:])

        for i in prange(4, m - 3):
            Dxu[i, :, :] = inv_dx * (
                0.083333333333333 * u[i-2, :, :] - 0.666666666666667 * u[i-1, :, :]
                + 0.666666666666667 * u[i+1, :, :] - 0.083333333333333 * u[i+2, :, :])

    else:  # order == 6
        Dxu[0, :, :] = inv_dx * (
            -1.694834962162858*u[0,:,:] + 2.245634824947698*u[1,:,:]
            - 0.055649692295628*u[2,:,:] - 0.670383570370653*u[3,:,:]
            - 0.188774952148393*u[4,:,:] + 0.552135032829910*u[5,:,:]
            - 0.188126680800077*u[6,:,:])
        Dxu[1, :, :] = inv_dx * (
            -0.434411786832708*u[0,:,:] + 0.107043134706685*u[2,:,:]
            + 0.420172642668695*u[3,:,:] + 0.119957288069806*u[4,:,:]
            - 0.328691543801578*u[5,:,:] + 0.122487487014485*u[6,:,:]
            - 0.006557221825386*u[7,:,:])
        Dxu[2, :, :] = inv_dx * (
            0.063307644169533*u[0,:,:] - 0.629491308812471*u[1,:,:]
            + 0.809935419586724*u[3,:,:] - 0.699016381364484*u[4,:,:]
            + 0.850345731199969*u[5,:,:] - 0.509589652965290*u[6,:,:]
            + 0.114508548186019*u[7,:,:])
        Dxu[3, :, :] = inv_dx * (
            0.110198643174386*u[0,:,:] - 0.357041083340051*u[1,:,:]
            - 0.117033418681039*u[2,:,:] + 0.120870009174558*u[4,:,:]
            + 0.349168902725368*u[5,:,:] - 0.104924741749615*u[6,:,:]
            - 0.001238311303608*u[7,:,:])
        Dxu[4, :, :] = inv_dx * (
            0.133544619364965*u[0,:,:] - 0.438678347579289*u[1,:,:]
            + 0.434686341173840*u[2,:,:] - 0.520172867814934*u[3,:,:]
            + 0.049912002176267*u[5,:,:] + 0.504693510958978*u[6,:,:]
            - 0.163985258279827*u[7,:,:])
        Dxu[5, :, :] = inv_dx * (
            -0.127754693486067*u[0,:,:] + 0.393149407857401*u[1,:,:]
            - 0.172955234680916*u[2,:,:] - 0.491489487857764*u[3,:,:]
            - 0.016325050231672*u[4,:,:] + 0.428167552785852*u[6,:,:]
            - 0.025864364383975*u[7,:,:] + 0.013071869997141*u[8,:,:])
        Dxu[6, :, :] = inv_dx * (
            0.060008241515128*u[0,:,:] - 0.201971348965594*u[1,:,:]
            + 0.142885356631256*u[2,:,:] + 0.203603636754774*u[3,:,:]
            - 0.227565385120003*u[4,:,:] - 0.590259111130048*u[5,:,:]
            + 0.757462553894374*u[7,:,:] - 0.162184436527372*u[8,:,:]
            + 0.018020492947486*u[9,:,:])
        Dxu[7, :, :] = inv_dx * (
            0.009910488565285*u[1,:,:] - 0.029429452176588*u[2,:,:]
            + 0.002202493355677*u[3,:,:] + 0.067773581604826*u[4,:,:]
            + 0.032681945726690*u[5,:,:] - 0.694285851935105*u[6,:,:]
            + 0.743286642396343*u[8,:,:] - 0.148657328479269*u[9,:,:]
            + 0.016517480942141*u[10,:,:])

        Dxu[m-7, :, :] = inv_dx * (
            -0.016517480942141*u[m-10,:,:] + 0.148657328479269*u[m-9,:,:]
            - 0.743286642396343*u[m-8,:,:] + 0.694285851935105*u[m-6,:,:]
            - 0.032681945726690*u[m-5,:,:] - 0.067773581604826*u[m-4,:,:]
            - 0.002202493355677*u[m-3,:,:] + 0.029429452176588*u[m-2,:,:]
            - 0.009910488565285*u[m-1,:,:])
        Dxu[m-6, :, :] = inv_dx * (
            -0.018020492947486*u[m-9,:,:] + 0.162184436527372*u[m-8,:,:]
            - 0.757462553894374*u[m-7,:,:] + 0.590259111130048*u[m-5,:,:]
            + 0.227565385120003*u[m-4,:,:] - 0.203603636754774*u[m-3,:,:]
            - 0.142885356631256*u[m-2,:,:] + 0.201971348965594*u[m-1,:,:]
            - 0.060008241515128*u[m,:,:])
        Dxu[m-5, :, :] = inv_dx * (
            -0.013071869997141*u[m-8,:,:] + 0.025864364383975*u[m-7,:,:]
            - 0.428167552785852*u[m-6,:,:] + 0.016325050231672*u[m-4,:,:]
            + 0.491489487857764*u[m-3,:,:] + 0.172955234680916*u[m-2,:,:]
            - 0.393149407857401*u[m-1,:,:] + 0.127754693486067*u[m,:,:])
        Dxu[m-4, :, :] = inv_dx * (
            0.163985258279827*u[m-7,:,:] - 0.504693510958978*u[m-6,:,:]
            - 0.049912002176267*u[m-5,:,:] + 0.520172867814934*u[m-3,:,:]
            - 0.434686341173840*u[m-2,:,:] + 0.438678347579289*u[m-1,:,:]
            - 0.133544619364965*u[m,:,:])
        Dxu[m-3, :, :] = inv_dx * (
            0.001238311303608*u[m-7,:,:] + 0.104924741749615*u[m-6,:,:]
            - 0.349168902725368*u[m-5,:,:] - 0.120870009174558*u[m-4,:,:]
            + 0.117033418681039*u[m-2,:,:] + 0.357041083340051*u[m-1,:,:]
            - 0.110198643174386*u[m,:,:])
        Dxu[m-2, :, :] = inv_dx * (
            -0.114508548186019*u[m-7,:,:] + 0.509589652965290*u[m-6,:,:]
            - 0.850345731199969*u[m-5,:,:] + 0.699016381364484*u[m-4,:,:]
            - 0.809935419586724*u[m-3,:,:] + 0.629491308812471*u[m-1,:,:]
            - 0.063307644169533*u[m,:,:])
        Dxu[m-1, :, :] = inv_dx * (
            0.006557221825386*u[m-7,:,:] - 0.122487487014485*u[m-6,:,:]
            + 0.328691543801578*u[m-5,:,:] - 0.119957288069806*u[m-4,:,:]
            - 0.420172642668695*u[m-3,:,:] - 0.107043134706685*u[m-2,:,:]
            + 0.434411786832708*u[m,:,:])
        Dxu[m, :, :] = inv_dx * (
            0.188126680800077*u[m-6,:,:] - 0.552135032829910*u[m-5,:,:]
            + 0.188774952148393*u[m-4,:,:] + 0.670383570370653*u[m-3,:,:]
            + 0.055649692295628*u[m-2,:,:] - 2.245634824947698*u[m-1,:,:]
            + 1.694834962162858*u[m,:,:])

        for i in prange(8, m - 7):
            Dxu[i, :, :] = inv_dx * (
                -0.016666666666667 * u[i-3, :, :] + 0.15 * u[i-2, :, :]
                - 0.75 * u[i-1, :, :] + 0.75 * u[i+1, :, :]
                - 0.15 * u[i+2, :, :] + 0.016666666666667 * u[i+3, :, :])


@njit(parallel=True, cache=True)
def sbp_dy_2d(Dyu, u, nx, ny, nf, dy, order):
    """
    In-place SBP y-derivative: Dyu = d(u)/dy.
    u, Dyu: float64[nx, ny, nf]
    Boundary nodes applied as array-wide slice ops.
    Interior uses prange over j.
    """
    m      = ny - 1
    inv_dy = 1.0 / dy

    if order == 2:
        Dyu[:, 0, :]  = (u[:, 1, :] - u[:, 0, :]) * inv_dy
        Dyu[:, m, :]  = (u[:, m, :] - u[:, m-1, :]) * inv_dy
        for j in prange(1, m):
            Dyu[:, j, :] = 0.5 * (u[:, j+1, :] - u[:, j-1, :]) * inv_dy

    elif order == 4:
        Dyu[:, 0, :] = inv_dy * (-24.0/17.0*u[:,0,:] + 59.0/34.0*u[:,1,:]
                                  - 4.0/17.0*u[:,2,:] - 3.0/34.0*u[:,3,:])
        Dyu[:, 1, :] = inv_dy * (-0.5*u[:,0,:] + 0.5*u[:,2,:])
        Dyu[:, 2, :] = inv_dy * (4.0/43.0*u[:,0,:] - 59.0/86.0*u[:,1,:]
                                  + 59.0/86.0*u[:,3,:] - 4.0/43.0*u[:,4,:])
        Dyu[:, 3, :] = inv_dy * (3.0/98.0*u[:,0,:] - 59.0/98.0*u[:,2,:]
                                  + 32.0/49.0*u[:,4,:] - 4.0/49.0*u[:,5,:])

        Dyu[:, m,   :] = inv_dy * (24.0/17.0*u[:,m,:] - 59.0/34.0*u[:,m-1,:]
                                    + 4.0/17.0*u[:,m-2,:] + 3.0/34.0*u[:,m-3,:])
        Dyu[:, m-1, :] = inv_dy * (0.5*u[:,m,:] - 0.5*u[:,m-2,:])
        Dyu[:, m-2, :] = inv_dy * (-4.0/43.0*u[:,m,:] + 59.0/86.0*u[:,m-1,:]
                                    - 59.0/86.0*u[:,m-3,:] + 4.0/43.0*u[:,m-4,:])
        Dyu[:, m-3, :] = inv_dy * (-3.0/98.0*u[:,m,:] + 59.0/98.0*u[:,m-2,:]
                                    - 32.0/49.0*u[:,m-4,:] + 4.0/49.0*u[:,m-5,:])

        for j in prange(4, m - 3):
            Dyu[:, j, :] = inv_dy * (
                0.083333333333333 * u[:, j-2, :] - 0.666666666666667 * u[:, j-1, :]
                + 0.666666666666667 * u[:, j+1, :] - 0.083333333333333 * u[:, j+2, :])

    else:  # order == 6
        Dyu[:, 0, :] = inv_dy * (
            -1.694834962162858*u[:,0,:] + 2.245634824947698*u[:,1,:]
            - 0.055649692295628*u[:,2,:] - 0.670383570370653*u[:,3,:]
            - 0.188774952148393*u[:,4,:] + 0.552135032829910*u[:,5,:]
            - 0.188126680800077*u[:,6,:])
        Dyu[:, 1, :] = inv_dy * (
            -0.434411786832708*u[:,0,:] + 0.107043134706685*u[:,2,:]
            + 0.420172642668695*u[:,3,:] + 0.119957288069806*u[:,4,:]
            - 0.328691543801578*u[:,5,:] + 0.122487487014485*u[:,6,:]
            - 0.006557221825386*u[:,7,:])
        Dyu[:, 2, :] = inv_dy * (
            0.063307644169533*u[:,0,:] - 0.629491308812471*u[:,1,:]
            + 0.809935419586724*u[:,3,:] - 0.699016381364484*u[:,4,:]
            + 0.850345731199969*u[:,5,:] - 0.509589652965290*u[:,6,:]
            + 0.114508548186019*u[:,7,:])
        Dyu[:, 3, :] = inv_dy * (
            0.110198643174386*u[:,0,:] - 0.357041083340051*u[:,1,:]
            - 0.117033418681039*u[:,2,:] + 0.120870009174558*u[:,4,:]
            + 0.349168902725368*u[:,5,:] - 0.104924741749615*u[:,6,:]
            - 0.001238311303608*u[:,7,:])
        Dyu[:, 4, :] = inv_dy * (
            0.133544619364965*u[:,0,:] - 0.438678347579289*u[:,1,:]
            + 0.434686341173840*u[:,2,:] - 0.520172867814934*u[:,3,:]
            + 0.049912002176267*u[:,5,:] + 0.504693510958978*u[:,6,:]
            - 0.163985258279827*u[:,7,:])
        Dyu[:, 5, :] = inv_dy * (
            -0.127754693486067*u[:,0,:] + 0.393149407857401*u[:,1,:]
            - 0.172955234680916*u[:,2,:] - 0.491489487857764*u[:,3,:]
            - 0.016325050231672*u[:,4,:] + 0.428167552785852*u[:,6,:]
            - 0.025864364383975*u[:,7,:] + 0.013071869997141*u[:,8,:])
        Dyu[:, 6, :] = inv_dy * (
            0.060008241515128*u[:,0,:] - 0.201971348965594*u[:,1,:]
            + 0.142885356631256*u[:,2,:] + 0.203603636754774*u[:,3,:]
            - 0.227565385120003*u[:,4,:] - 0.590259111130048*u[:,5,:]
            + 0.757462553894374*u[:,7,:] - 0.162184436527372*u[:,8,:]
            + 0.018020492947486*u[:,9,:])
        Dyu[:, 7, :] = inv_dy * (
            0.009910488565285*u[:,1,:] - 0.029429452176588*u[:,2,:]
            + 0.002202493355677*u[:,3,:] + 0.067773581604826*u[:,4,:]
            + 0.032681945726690*u[:,5,:] - 0.694285851935105*u[:,6,:]
            + 0.743286642396343*u[:,8,:] - 0.148657328479269*u[:,9,:]
            + 0.016517480942141*u[:,10,:])

        Dyu[:, m-7, :] = inv_dy * (
            -0.016517480942141*u[:,m-10,:] + 0.148657328479269*u[:,m-9,:]
            - 0.743286642396343*u[:,m-8,:] + 0.694285851935105*u[:,m-6,:]
            - 0.032681945726690*u[:,m-5,:] - 0.067773581604826*u[:,m-4,:]
            - 0.002202493355677*u[:,m-3,:] + 0.029429452176588*u[:,m-2,:]
            - 0.009910488565285*u[:,m-1,:])
        Dyu[:, m-6, :] = inv_dy * (
            -0.018020492947486*u[:,m-9,:] + 0.162184436527372*u[:,m-8,:]
            - 0.757462553894374*u[:,m-7,:] + 0.590259111130048*u[:,m-5,:]
            + 0.227565385120003*u[:,m-4,:] - 0.203603636754774*u[:,m-3,:]
            - 0.142885356631256*u[:,m-2,:] + 0.201971348965594*u[:,m-1,:]
            - 0.060008241515128*u[:,m,:])
        Dyu[:, m-5, :] = inv_dy * (
            -0.013071869997141*u[:,m-8,:] + 0.025864364383975*u[:,m-7,:]
            - 0.428167552785852*u[:,m-6,:] + 0.016325050231672*u[:,m-4,:]
            + 0.491489487857764*u[:,m-3,:] + 0.172955234680916*u[:,m-2,:]
            - 0.393149407857401*u[:,m-1,:] + 0.127754693486067*u[:,m,:])
        Dyu[:, m-4, :] = inv_dy * (
            0.163985258279827*u[:,m-7,:] - 0.504693510958978*u[:,m-6,:]
            - 0.049912002176267*u[:,m-5,:] + 0.520172867814934*u[:,m-3,:]
            - 0.434686341173840*u[:,m-2,:] + 0.438678347579289*u[:,m-1,:]
            - 0.133544619364965*u[:,m,:])
        Dyu[:, m-3, :] = inv_dy * (
            0.001238311303608*u[:,m-7,:] + 0.104924741749615*u[:,m-6,:]
            - 0.349168902725368*u[:,m-5,:] - 0.120870009174558*u[:,m-4,:]
            + 0.117033418681039*u[:,m-2,:] + 0.357041083340051*u[:,m-1,:]
            - 0.110198643174386*u[:,m,:])
        Dyu[:, m-2, :] = inv_dy * (
            -0.114508548186019*u[:,m-7,:] + 0.509589652965290*u[:,m-6,:]
            - 0.850345731199969*u[:,m-5,:] + 0.699016381364484*u[:,m-4,:]
            - 0.809935419586724*u[:,m-3,:] + 0.629491308812471*u[:,m-1,:]
            - 0.063307644169533*u[:,m,:])
        Dyu[:, m-1, :] = inv_dy * (
            0.006557221825386*u[:,m-7,:] - 0.122487487014485*u[:,m-6,:]
            + 0.328691543801578*u[:,m-5,:] - 0.119957288069806*u[:,m-4,:]
            - 0.420172642668695*u[:,m-3,:] - 0.107043134706685*u[:,m-2,:]
            + 0.434411786832708*u[:,m,:])
        Dyu[:, m, :] = inv_dy * (
            0.188126680800077*u[:,m-6,:] - 0.552135032829910*u[:,m-5,:]
            + 0.188774952148393*u[:,m-4,:] + 0.670383570370653*u[:,m-3,:]
            + 0.055649692295628*u[:,m-2,:] - 2.245634824947698*u[:,m-1,:]
            + 1.694834962162858*u[:,m,:])

        for j in prange(8, m - 7):
            Dyu[:, j, :] = inv_dy * (
                -0.016666666666667 * u[:, j-3, :] + 0.15 * u[:, j-2, :]
                - 0.75 * u[:, j-1, :] + 0.75 * u[:, j+1, :]
                - 0.15 * u[:, j+2, :] + 0.016666666666667 * u[:, j+3, :])


# ============================================================================
# Phase 2 — Penalty weight (h11 * dx for SBP boundary norm)
# ============================================================================

@njit(cache=True)
def _h11(order, dx):
    """Return h11 * dx — SBP boundary norm weight."""
    if order == 2:
        return 0.5 * dx
    elif order == 4:
        return (17.0 / 48.0) * dx
    else:  # 6
        return (13649.0 / 43200.0) * dx


# ============================================================================
# Phase 3 — SAT boundary condition kernels (Mode II elastic)
# x-outer boundary: left domain i=0, right domain i=nx-1
# x-fault  boundary: left domain i=nx-1, right domain i=0  (handled by interface kernel)
# y-bottom boundary: j=0
# y-top    boundary: j=ny-1
#
# SAT penalty sign conventions exactly match boundarycondition.bcm2dx/bcp2dx/bcm2dy/bcp2dy
# ============================================================================

@njit(cache=True)
def _bc_x_left(D, F, rho, twomulam, mu, hx, r0):
    """
    SAT penalty for LEFT outer boundary (x_outer face, i=0) of left domain.
    Mirrors boundarycondition.bcm2dx with r[0]=r0.
    D, F: float64[nx, ny, nf]
    """
    cp  = math.sqrt(twomulam / rho)
    cs  = math.sqrt(mu / rho)
    zp  = rho * cp
    zs  = rho * cs
    lam = twomulam - 2.0 * mu
    inv_hx = 1.0 / hx

    ny = F.shape[1]
    for j in range(ny):
        vx  = F[0, j, 0]; vy  = F[0, j, 1]
        sxx = F[0, j, 2];                   sxy = F[0, j, 4]
        px = 0.5 * (zp * vx - sxx);  qx = 0.5 * (zp * vx + sxx)
        py = 0.5 * (zs * vy - sxy);  qy = 0.5 * (zs * vy + sxy)
        sat_x = px - r0 * qx
        sat_y = py - r0 * qy
        D[0, j, 0] -= inv_hx * (1.0 / rho) * sat_x
        D[0, j, 1] -= inv_hx * (1.0 / rho) * sat_y
        D[0, j, 2] -= inv_hx * (-twomulam / zp) * sat_x
        D[0, j, 3] -= inv_hx * (-lam / zp) * sat_x
        D[0, j, 4] -= inv_hx * (-mu / zs) * sat_y


@njit(cache=True)
def _bc_x_right(D, F, rho, twomulam, mu, hx, rnx, nx):
    """
    SAT penalty for RIGHT outer boundary (x_outer face, i=nx-1) of right domain.
    Mirrors boundarycondition.bcp2dx with r[0]=rnx.
    """
    cp  = math.sqrt(twomulam / rho)
    cs  = math.sqrt(mu / rho)
    zp  = rho * cp
    zs  = rho * cs
    lam = twomulam - 2.0 * mu
    inv_hx = 1.0 / hx
    m = nx - 1

    ny = F.shape[1]
    for j in range(ny):
        vx  = F[m, j, 0]; vy  = F[m, j, 1]
        sxx = F[m, j, 2];                   sxy = F[m, j, 4]
        px = 0.5 * (zp * vx + sxx);  qx = 0.5 * (zp * vx - sxx)
        py = 0.5 * (zs * vy + sxy);  qy = 0.5 * (zs * vy - sxy)
        sat_x = px - rnx * qx
        sat_y = py - rnx * qy
        D[m, j, 0] -= inv_hx * (1.0 / rho) * sat_x
        D[m, j, 1] -= inv_hx * (1.0 / rho) * sat_y
        D[m, j, 2] -= inv_hx * (twomulam / zp) * sat_x
        D[m, j, 3] -= inv_hx * (lam / zp) * sat_x
        D[m, j, 4] -= inv_hx * (mu / zs) * sat_y


@njit(cache=True)
def _bc_y_bottom(D, F, rho, twomulam, mu, hy, r0y):
    """
    SAT penalty for y=0 boundary (bottom). Mirrors boundarycondition.bcm2dy with r[2]=r0y.
    """
    cp = math.sqrt(twomulam / rho)
    cs = math.sqrt(mu / rho)
    zp = rho * cp
    zs = rho * cs
    inv_hy = 1.0 / hy
    lam = twomulam - 2.0 * mu

    nx = D.shape[0]
    for i in range(nx):
        vx  = F[i, 0, 0]; vy  = F[i, 0, 1]
        syy = F[i, 0, 3];                   sxy = F[i, 0, 4]
        px = 0.5 * (zs * vx - sxy);  qx = 0.5 * (zs * vx + sxy)
        py = 0.5 * (zp * vy - syy);  qy = 0.5 * (zp * vy + syy)
        sat_x = px - r0y * qx
        sat_y = py - r0y * qy
        D[i, 0, 0] -= inv_hy * (1.0 / rho) * sat_x
        D[i, 0, 1] -= inv_hy * (1.0 / rho) * sat_y
        D[i, 0, 2] -= inv_hy * (-lam / zp) * sat_y
        D[i, 0, 3] -= inv_hy * (-twomulam / zp) * sat_y
        D[i, 0, 4] -= inv_hy * (-mu / zs) * sat_x


@njit(cache=True)
def _bc_y_top(D, F, rho, twomulam, mu, hy, rny, ny):
    """
    SAT penalty for y=Ly boundary (top). Mirrors boundarycondition.bcp2dy with r[3]=rny.
    """
    cp = math.sqrt(twomulam / rho)
    cs = math.sqrt(mu / rho)
    zp = rho * cp
    zs = rho * cs
    inv_hy = 1.0 / hy
    lam = twomulam - 2.0 * mu
    m = ny - 1

    nx = D.shape[0]
    for i in range(nx):
        vx  = F[i, m, 0]; vy  = F[i, m, 1]
        syy = F[i, m, 3];                   sxy = F[i, m, 4]
        px = 0.5 * (zs * vx + sxy);  qx = 0.5 * (zs * vx - sxy)
        py = 0.5 * (zp * vy + syy);  qy = 0.5 * (zp * vy - syy)
        sat_x = px - rny * qx
        sat_y = py - rny * qy
        D[i, m, 0] -= inv_hy * (1.0 / rho) * sat_x
        D[i, m, 1] -= inv_hy * (1.0 / rho) * sat_y
        D[i, m, 2] -= inv_hy * (lam / zp) * sat_y
        D[i, m, 3] -= inv_hy * (twomulam / zp) * sat_y
        D[i, m, 4] -= inv_hy * (mu / zs) * sat_x


# ============================================================================
# Phase 4 — Friction helpers (scalar @njit — same as 1d_numba)
# ============================================================================

@njit(cache=True)
def _tau_strength_sw(slip_val, alp_s, alp_d, sigma_n, D_c):
    """Fault strength for slip-weakening law. sigma_n positive = compressive."""
    s_capped = slip_val if slip_val < D_c else D_c
    coeff = alp_s - (alp_s - alp_d) * s_capped / D_c
    return coeff * sigma_n   # sigma_n > 0 here (absolute value)


@njit(cache=True)
def _regula_falsi(V, Phi, eta, sigma_n, psi, V0, a):
    """
    Solve: V + (a*sigma_n/eta)*asinh(V/(2*V0)*exp(psi/a)) = Phi/eta
    Illinois-variant Regula-Falsi.  Identical to 1d_numba.
    """
    tol    = 1.0e-12
    Vl     = 0.0
    Vr     = Phi / eta
    maxit  = 5000

    coeff      = a * sigma_n / eta
    rhs        = Phi / eta
    exp_psi_a  = math.exp(psi / a)
    inv_2V0    = 0.5 / V0

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
            Vl = V;  V = Vr - (Vr - Vl) * fr / (fr - fl);  fl = fv
        elif fv * fr > tol:
            Vr = V;  V = Vr - (Vr - Vl) * fr / (fr - fl);  fr = fv
        fv  = V + coeff * math.asinh(inv_2V0 * V * exp_psi_a) - rhs
        err = math.fabs(V - V1)
        k  += 1
    return V


# ============================================================================
# Phase 5 — Fault interface kernel (parallel over fault points j)
# ============================================================================

@njit(parallel=True, cache=True)
def _interface_kernel_2d(D_l, D_r, F_l, F_r,
                          slip, psi, dslip, dpsi,
                          friction_params, fault_output0,
                          nx, ny, nf, hx,
                          rho, twomulam, mu,
                          mode, t, Y_arr, Y0,
                          fric_flag):
    """
    Fault interface coupling + friction + state evolution.
    Applies SAT penalties to D_l[nx-1,:,:] and D_r[0,:,:] in place.
    Records fault_output0[j, 0:4] (stage-0 diagnostics).

    For Mode II:  normal coupling on vx/sxx (locked), shear on vy/sxy (friction).
    For Mode III: shear coupling on vz/sxz.

    Y_arr: float64[ny]  — y-coordinates along fault
    fric_flag: 0.0=SW, 1.0=RS
    """
    cp  = math.sqrt(twomulam / rho)
    cs  = math.sqrt(mu / rho)
    zp  = rho * cp
    zs  = rho * cs
    lam = twomulam - 2.0 * mu

    inv_hx = 1.0 / hx
    m = nx - 1

    for j in prange(ny):

        # ---------- extract fault-face fields ----------
        if mode == 2:  # Mode II: [vx, vy, sxx, syy, sxy]
            vx_l = F_l[m, j, 0];  vx_r = F_r[0, j, 0]
            vy_l = F_l[m, j, 1];  vy_r = F_r[0, j, 1]
            Tx_l = F_l[m, j, 2];  Tx_r = F_r[0, j, 2]   # sxx
            Ty_l = F_l[m, j, 4];  Ty_r = F_r[0, j, 4]   # sxy
        else:          # Mode III: [vz, sxz, syz]
            vx_l = 0.0;  vx_r = 0.0
            vy_l = F_l[m, j, 0];  vy_r = F_r[0, j, 0]
            Tx_l = 0.0;  Tx_r = 0.0
            Ty_l = F_l[m, j, 1];  Ty_r = F_r[0, j, 1]   # sxz

        # ---------- NORMAL direction: locked interface (x coupling) ----------
        if mode == 2:
            q_mx = zp * vx_l - Tx_l
            p_px = zp * vx_r + Tx_r
            eta_p = 0.5 * zp
            Phi_x = eta_p * (p_px / zp - q_mx / zp)
            vv_x  = 0.0   # locked
            Tx_m  = Phi_x
            Tx_p  = Phi_x
            Vx_m  = (p_px - Tx_p) / zp - vv_x
            Vx_p  = (q_mx + Tx_m) / zp + vv_x
        else:
            Tx_m = 0.0; Tx_p = 0.0; Vx_m = 0.0; Vx_p = 0.0

        # ---------- sigma_n update (exact formula from rate2d.py) ----------
        # stored fp[8,j] is background normal stress (compressive = negative)
        raw_sigma = -(Tx_m + friction_params[8, j])
        sigma_n   = raw_sigma if raw_sigma > 0.0 else 0.0

        # ---------- SHEAR direction: friction (y coupling) ----------
        q_sy  = zs * vy_l - Ty_l    # left outgoing S-characteristic
        p_py  = zs * vy_r + Ty_r    # right outgoing S-characteristic
        eta_s = 0.5 * zs            # equal-material half-impedance
        Phi_y = 0.5 * (p_py - q_sy) # stress-transfer functional

        # Tau_0 / nucleation perturbation
        tau = friction_params[2, j]
        if fric_flag == FRIC_RS:
            r_nuc = math.fabs(Y_arr[j] - Y0)
            F_nuc = 0.0
            if r_nuc < 3.0:
                F_nuc = math.exp(r_nuc * r_nuc / (r_nuc * r_nuc - 9.0))
            G_nuc = 0.0
            if t > 0.0 and t < 1.0:
                G_nuc = math.exp((t - 1.0) ** 2 / (t * (t - 2.0)))
            elif t >= 1.0:
                G_nuc = 1.0
            tau = tau + 25.0 * F_nuc * G_nuc

        # Friction solve → vv (slip rate), Tau_h (absolute interface traction)
        if fric_flag == FRIC_SW:
            alp_s  = friction_params[9,  j]
            alp_d  = friction_params[10, j]
            D_c_j  = friction_params[11, j]
            Tau_lock = Phi_y + tau
            T_str    = _tau_strength_sw(slip[j], alp_s, alp_d, sigma_n, D_c_j)
            if Tau_lock >= T_str:    # slipping
                vv    = (Tau_lock - T_str) / eta_s
                Tau_h = Tau_lock - eta_s * vv    # = T_str
            else:                    # locked
                vv    = 0.0
                Tau_h = Tau_lock
        else:  # FRIC_RS
            V0_j    = friction_params[7, j]
            a_j     = friction_params[5, j]
            vv_init = math.fabs(vy_r - vy_l)
            if vv_init < 1.0e-16:
                vv_init = 1.0e-16
            vv    = _regula_falsi(vv_init, Phi_y + tau, eta_s, sigma_n,
                                   psi[j], V0_j, a_j)
            Tau_h = (Phi_y + tau) - eta_s * vv

        # Interface traction relative to background (= what Interface_Fault uses as Sxy)
        Ty_m = Tau_h - tau
        Ty_p = Tau_h - tau

        # Hat velocities (serial: V_m = (p_p - T_p)/Zs_p - vv, V_p = (q_m + T_m)/Zs_m + vv)
        Vy_m = (p_py - Ty_p) / zs - vv
        Vy_p = (q_sy + Ty_m) / zs + vv

        # ---------- SAT interface flux penalties for LEFT domain (i = nx-1) -------
        # Mirrors Interface_Fault(side='left') in interface.py: D -= (1/hx) * BF
        # NO factor of 2 (unlike 1D couple_friction which has factor 2)
        if mode == 2:
            # x-direction (normal P-wave)
            px_l   = 0.5 * (zp * vx_l + Tx_l)    # outgoing P-char from left
            Px_hat = 0.5 * (zp * Vx_m + Tx_m)    # hat P-char
            dP_l   = px_l - Px_hat
            D_l[m, j, 0] -= inv_hx * (1.0 / rho)     * dP_l
            D_l[m, j, 2] -= inv_hx * (twomulam / zp) * dP_l
            D_l[m, j, 3] -= inv_hx * (lam / zp)      * dP_l

            # y-direction (shear S-wave)
            py_l   = 0.5 * (zs * vy_l + Ty_l)
            Py_hat = 0.5 * (zs * Vy_m + Ty_m)
            dS_l   = py_l - Py_hat
            D_l[m, j, 1] -= inv_hx * (1.0 / rho) * dS_l
            D_l[m, j, 4] -= inv_hx * (mu / zs)   * dS_l
        else:  # Mode III: shear only (vz, sxz index 0, 1)
            py_l   = 0.5 * (zs * vy_l + Ty_l)
            Py_hat = 0.5 * (zs * Vy_m + Ty_m)
            dS_l   = py_l - Py_hat
            D_l[m, j, 0] -= inv_hx * (1.0 / rho) * dS_l
            D_l[m, j, 1] -= inv_hx * (mu / zs)   * dS_l

        # ---------- SAT interface flux penalties for RIGHT domain (i = 0) ----------
        # Mirrors Interface_Fault(side='right'): D -= (1/hx) * BF (note sign conventions flip)
        if mode == 2:
            # x-direction (normal P-wave; minus sign in right-side char)
            qx_r   = 0.5 * (zp * vx_r - Tx_r)
            Qx_hat = 0.5 * (zp * Vx_p - Tx_p)
            dP_r   = qx_r - Qx_hat
            D_r[0, j, 0] -= inv_hx * (1.0 / rho)      * dP_r
            D_r[0, j, 2] -= inv_hx * (-twomulam / zp) * dP_r
            D_r[0, j, 3] -= inv_hx * (-lam / zp)      * dP_r

            # y-direction (shear S-wave)
            qy_r   = 0.5 * (zs * vy_r - Ty_r)
            Qy_hat = 0.5 * (zs * Vy_p - Ty_p)
            dS_r   = qy_r - Qy_hat
            D_r[0, j, 1] -= inv_hx * (1.0 / rho) * dS_r
            D_r[0, j, 4] -= inv_hx * (-mu / zs)  * dS_r
        else:  # Mode III
            qy_r   = 0.5 * (zs * vy_r - Ty_r)
            Qy_hat = 0.5 * (zs * Vy_p - Ty_p)
            dS_r   = qy_r - Qy_hat
            D_r[0, j, 0] -= inv_hx * (1.0 / rho) * dS_r
            D_r[0, j, 1] -= inv_hx * (-mu / zs)  * dS_r

        # ---------- state evolution (aging law): dslip, dpsi ----------
        # vv_abs = |Vy_p - Vy_m| = 2*vv for equal-material case
        vv_abs = math.fabs(Vy_p - Vy_m)
        dslip[j] = vv_abs
        b_j  = friction_params[6, j]
        V0_j = friction_params[7, j]
        L0_j = friction_params[3, j]
        f0_j = friction_params[4, j]
        dpsi[j] = b_j * V0_j / L0_j * math.exp(-(psi[j] - f0_j) / b_j) - vv_abs * b_j / L0_j

        # ---------- fault diagnostics (matches serial FaultOutput0[j, 0:4]) ----------
        fault_output0[j, 0] = math.fabs(Vx_p - Vx_m)       # normal slip rate (0 for locked)
        fault_output0[j, 1] = vv_abs                          # shear slip rate
        fault_output0[j, 2] = Tx_m + friction_params[8, j]  # normal traction
        fault_output0[j, 3] = Ty_m + friction_params[3, j]  # matches serial code (fp[3]=L0)


# ============================================================================
# Phase 6 — Assemble elastic rate equations (interior)
# Mode II: [vx, vy, sxx, syy, sxy]
# Mode III: [vz, sxz, syz]
# ============================================================================

@njit(parallel=True, cache=True)
def _assemble_rates_2d(D_l, D_r, DxF_l, DyF_l, DxF_r, DyF_r,
                        rho, twomulam, mu, nx, ny, mode):
    """
    Assemble D = A*DxF + B*DyF pointwise.
    D, DxF, DyF: float64[nx, ny, nf]
    Homogeneous material assumption.
    """
    inv_rho = 1.0 / rho
    lam     = twomulam - 2.0 * mu

    if mode == 2:  # Mode II, nf=5: [vx, vy, sxx, syy, sxy]
        for i in prange(nx):
            for j in range(ny):
                D_l[i, j, 0] = inv_rho * (DxF_l[i, j, 2] + DyF_l[i, j, 4])
                D_l[i, j, 1] = inv_rho * (DxF_l[i, j, 4] + DyF_l[i, j, 3])
                D_l[i, j, 2] = twomulam * DxF_l[i, j, 0] + lam * DyF_l[i, j, 1]
                D_l[i, j, 3] = twomulam * DyF_l[i, j, 1] + lam * DxF_l[i, j, 0]
                D_l[i, j, 4] = mu * (DyF_l[i, j, 0] + DxF_l[i, j, 1])

                D_r[i, j, 0] = inv_rho * (DxF_r[i, j, 2] + DyF_r[i, j, 4])
                D_r[i, j, 1] = inv_rho * (DxF_r[i, j, 4] + DyF_r[i, j, 3])
                D_r[i, j, 2] = twomulam * DxF_r[i, j, 0] + lam * DyF_r[i, j, 1]
                D_r[i, j, 3] = twomulam * DyF_r[i, j, 1] + lam * DxF_r[i, j, 0]
                D_r[i, j, 4] = mu * (DyF_r[i, j, 0] + DxF_r[i, j, 1])

    else:  # Mode III, nf=3: [vz, sxz, syz]
        for i in prange(nx):
            for j in range(ny):
                D_l[i, j, 0] = inv_rho * (DxF_l[i, j, 1] + DyF_l[i, j, 2])
                D_l[i, j, 1] = mu * DxF_l[i, j, 0]
                D_l[i, j, 2] = mu * DyF_l[i, j, 0]

                D_r[i, j, 0] = inv_rho * (DxF_r[i, j, 1] + DyF_r[i, j, 2])
                D_r[i, j, 1] = mu * DxF_r[i, j, 0]
                D_r[i, j, 2] = mu * DyF_r[i, j, 0]


# ============================================================================
# Phase 7 — Point source injection (optional)
# ============================================================================

@njit(cache=True)
def _inject_source_2d(D_l, t, nx, ny,
                       x0, y0, t_src, T_src, M0,
                       source_moment, dx, dy):
    """
    Add point source to left domain rates.
    source_moment: float64[nf]  — moment tensor diagonal [M0,M1,M2,M3,M4]
    source_moment[4] is always 0 for the default M=[0,0,1,1,0].
    Spatial: 2D Gaussian with width 2*dx × 2*dy centred at (x0, y0).
    Temporal: Gaussian pulse centred at t_src, width T_src.
    """
    if M0 == 0.0:
        return

    sigma_x = 2.0 * dx
    sigma_y = 2.0 * dy

    inv_sx2 = 1.0 / (2.0 * sigma_x * sigma_x)
    inv_sy2 = 1.0 / (2.0 * sigma_y * sigma_y)
    norm_x  = 1.0 / math.sqrt(2.0 * math.pi * sigma_x * sigma_x)
    norm_y  = 1.0 / math.sqrt(2.0 * math.pi * sigma_y * sigma_y)

    # temporal function: Gaussian
    sigma_t = T_src
    f_t = (1.0 / math.sqrt(2.0 * math.pi * sigma_t * sigma_t)
           * math.exp(-((t - t_src) ** 2) / (2.0 * sigma_t * sigma_t)))

    nf = D_l.shape[2]
    for i in range(nx):
        for j in range(ny):
            # x coord of left domain: -Lx + i*dx  → we don't have coords here,
            # but the runner passes x0 relative to left domain origin.
            # Approximate: evaluate g from grid indices using dx,dy and x0,y0
            # (runner ensures x0<0 for left domain)
            x_ij = -(nx - 1) * dx + i * dx   # X_l[i,0]
            y_ij = j * dy
            g = (norm_x * norm_y
                 * math.exp(-((x_ij - x0) ** 2) * inv_sx2)
                 * math.exp(-((y_ij - y0) ** 2) * inv_sy2))
            ft = M0 * g * f_t
            for f in range(nf):
                D_l[i, j, f] += source_moment[f] * ft


# ============================================================================
# Phase 8 — Full elastic rate for one RK4 stage
# ============================================================================

@njit(cache=True)
def elastic_rate_2d(D_l, D_r,
                     F_l, F_r,
                     slip, psi, dslip, dpsi,
                     fault_output0,
                     friction_params, Y_arr, Y0, t,
                     nx, ny, nf, dx, dy, order,
                     r_l, r_r,
                     rho, twomulam, mu,
                     mode, fric_flag,
                     x0_src, y0_src, t0_src, T_src, M0_src, source_moment):
    """
    Compute rates D_l, D_r, dslip, dpsi for one RK4 stage (in-place).

    r_l, r_r: float64[4]  — [x_outer, x_fault(unused), y_bottom, y_top]
    source_moment: float64[nf]
    """
    hx = _h11(order, dx)
    hy = _h11(order, dy)

    # 1 — SBP derivatives
    DxF_l = np.empty_like(F_l)
    DyF_l = np.empty_like(F_l)
    DxF_r = np.empty_like(F_r)
    DyF_r = np.empty_like(F_r)
    sbp_dx_2d(DxF_l, F_l, nx, ny, nf, dx, order)
    sbp_dy_2d(DyF_l, F_l, nx, ny, nf, dy, order)
    sbp_dx_2d(DxF_r, F_r, nx, ny, nf, dx, order)
    sbp_dy_2d(DyF_r, F_r, nx, ny, nf, dy, order)

    # 2 — Assemble rate equations
    _assemble_rates_2d(D_l, D_r, DxF_l, DyF_l, DxF_r, DyF_r,
                        rho, twomulam, mu, nx, ny, mode)

    # 3 — Point source (left domain only)
    if M0_src != 0.0:
        _inject_source_2d(D_l, t, nx, ny,
                           x0_src, y0_src, t0_src, T_src, M0_src,
                           source_moment, dx, dy)

    # 4 — Fault interface + friction
    _interface_kernel_2d(D_l, D_r, F_l, F_r,
                          slip, psi, dslip, dpsi,
                          friction_params, fault_output0,
                          nx, ny, nf, hx,
                          rho, twomulam, mu,
                          mode, t, Y_arr, Y0,
                          fric_flag)

    # 5 — Outer SAT boundary conditions
    r0_l = r_l[0]   # x_outer left domain
    r0y  = r_l[2]   # y_bottom
    rny  = r_l[3]   # y_top
    r0_r = r_r[0]   # x_outer right domain

    _bc_x_left (D_l, F_l, rho, twomulam, mu, hx, r0_l)
    _bc_x_right(D_r, F_r, rho, twomulam, mu, hx, r0_r, nx)
    _bc_y_bottom(D_l, F_l, rho, twomulam, mu, hy, r0y)
    _bc_y_top   (D_l, F_l, rho, twomulam, mu, hy, rny, ny)
    _bc_y_bottom(D_r, F_r, rho, twomulam, mu, hy, r0y)
    _bc_y_top   (D_r, F_r, rho, twomulam, mu, hy, rny, ny)


# ============================================================================
# Phase 9 — Classical 4-stage RK4 step (in-place update)
# ============================================================================

@njit(cache=True)
def rk4_step_2d(F_l, F_r, slip, psi,
                friction_params, Y_arr, Y0, t,
                nx, ny, nf, dx, dy, dt, order,
                r_l, r_r,
                rho, twomulam, mu,
                mode, fric_flag,
                x0_src, y0_src, t0_src, T_src, M0_src, source_moment,
                fault_output0):
    """
    Classical 4th-order Runge-Kutta step.
    Modifies F_l, F_r, slip, psi, fault_output0 in place.

    fault_output0: float64[ny, 6] — updated from stage-1 (k1) rates.

    Returns (dslip_k4, dpsi_k4): fault-point rates from final stage
    (can be used for convergence diagnostics).
    """
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    k1_l    = np.empty_like(F_l)
    k1_r    = np.empty_like(F_r)
    k1_slip = np.empty(ny)
    k1_psi  = np.empty(ny)
    fo1     = np.empty((ny, 6))   # stage-1 fault output → write to fault_output0

    elastic_rate_2d(k1_l, k1_r, F_l, F_r,
                     slip, psi, k1_slip, k1_psi, fo1,
                     friction_params, Y_arr, Y0, t,
                     nx, ny, nf, dx, dy, order, r_l, r_r,
                     rho, twomulam, mu, mode, fric_flag,
                     x0_src, y0_src, t0_src, T_src, M0_src, source_moment)
    fault_output0[:, :] = fo1[:, :]   # record stage-1 data as "current" fault state

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    k2_l    = np.empty_like(F_l)
    k2_r    = np.empty_like(F_r)
    k2_slip = np.empty(ny)
    k2_psi  = np.empty(ny)
    fo2     = np.empty((ny, 6))

    elastic_rate_2d(k2_l, k2_r, F_l + 0.5 * dt * k1_l, F_r + 0.5 * dt * k1_r,
                     slip + 0.5 * dt * k1_slip, psi + 0.5 * dt * k1_psi,
                     k2_slip, k2_psi, fo2,
                     friction_params, Y_arr, Y0, t + 0.5 * dt,
                     nx, ny, nf, dx, dy, order, r_l, r_r,
                     rho, twomulam, mu, mode, fric_flag,
                     x0_src, y0_src, t0_src, T_src, M0_src, source_moment)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    k3_l    = np.empty_like(F_l)
    k3_r    = np.empty_like(F_r)
    k3_slip = np.empty(ny)
    k3_psi  = np.empty(ny)
    fo3     = np.empty((ny, 6))

    elastic_rate_2d(k3_l, k3_r, F_l + 0.5 * dt * k2_l, F_r + 0.5 * dt * k2_r,
                     slip + 0.5 * dt * k2_slip, psi + 0.5 * dt * k2_psi,
                     k3_slip, k3_psi, fo3,
                     friction_params, Y_arr, Y0, t + 0.5 * dt,
                     nx, ny, nf, dx, dy, order, r_l, r_r,
                     rho, twomulam, mu, mode, fric_flag,
                     x0_src, y0_src, t0_src, T_src, M0_src, source_moment)

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    k4_l    = np.empty_like(F_l)
    k4_r    = np.empty_like(F_r)
    k4_slip = np.empty(ny)
    k4_psi  = np.empty(ny)
    fo4     = np.empty((ny, 6))

    elastic_rate_2d(k4_l, k4_r, F_l + dt * k3_l, F_r + dt * k3_r,
                     slip + dt * k3_slip, psi + dt * k3_psi,
                     k4_slip, k4_psi, fo4,
                     friction_params, Y_arr, Y0, t + dt,
                     nx, ny, nf, dx, dy, order, r_l, r_r,
                     rho, twomulam, mu, mode, fric_flag,
                     x0_src, y0_src, t0_src, T_src, M0_src, source_moment)

    # ── In-place state update ─────────────────────────────────────────────────
    c = dt / 6.0
    F_l   += c * (k1_l    + 2.0 * k2_l    + 2.0 * k3_l    + k4_l)
    F_r   += c * (k1_r    + 2.0 * k2_r    + 2.0 * k3_r    + k4_r)
    slip  += c * (k1_slip  + 2.0 * k2_slip + 2.0 * k3_slip + k4_slip)
    psi   += c * (k1_psi   + 2.0 * k2_psi  + 2.0 * k3_psi  + k4_psi)

    return k4_slip, k4_psi
