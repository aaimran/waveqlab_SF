"""
physics_loss.py — PDE and fault boundary condition residuals for PINO training.

The elastic wave equation (velocity-stress form):
    ∂_t v = (1/ρ) ∂_x s
    ∂_t s = μ     ∂_x v

Derivatives are evaluated exactly in Fourier space:
    ∂_x u  ↔  i k_x  û(k_x, k_t)
    ∂_t u  ↔  i ω    û(k_x, k_t)

Fault interface (Slip-Weakening friction):
    τ = σ_n [α_s - (α_s - α_d) min(D / D_c, 1)]
    τ = τ_0 + s_l(x_f)      (stress continuity at left  boundary)
    τ = τ_0 - s_r(x_f)      (stress continuity at right boundary)
    [v] = v_l(x_f) - v_r(x_f)  slip rate must be ≥ 0 during rupture

All tensors use layout (batch, channel, NX, NT) consistent with FNO2d output.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PDE residual (interior elastic wave equation)
# ---------------------------------------------------------------------------

def fourier_derivative_x(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Spatial derivative ∂_x u via FFT along dim=-2 (NX axis).

    u   : (..., NX, NT)
    dx  : grid spacing (physical units)
    """
    nx = u.shape[-2]
    u_ft = torch.fft.rfft(u, n=nx, dim=-2, norm='ortho')
    # Wavenumbers: k_j = 2π j / (NX * dx),  j = 0, …, NX//2
    kx = torch.fft.rfftfreq(nx, d=dx / (2.0 * torch.pi)).to(u.device)
    # Multiply by i kx (broadcast over batch, channel, NT)
    u_ft = u_ft * (1j * kx).view(1, 1, -1, 1)
    return torch.fft.irfft(u_ft, n=nx, dim=-2, norm='ortho')


def fourier_derivative_t(u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Temporal derivative ∂_t u via FFT along dim=-1 (NT axis).

    u   : (..., NX, NT)
    dt  : time step (physical units)
    """
    nt = u.shape[-1]
    u_ft = torch.fft.rfft(u, n=nt, dim=-1, norm='ortho')
    omega = torch.fft.rfftfreq(nt, d=dt / (2.0 * torch.pi)).to(u.device)
    u_ft = u_ft * (1j * omega).view(1, 1, 1, -1)
    return torch.fft.irfft(u_ft, n=nt, dim=-1, norm='ortho')


def pde_residual(
    v: torch.Tensor,   # (batch, NX, NT) — velocity field
    s: torch.Tensor,   # (batch, NX, NT) — stress field
    rho: float,        # density (kg/m³)
    mu: float,         # shear modulus (Pa)
    dx: float,         # spatial grid spacing (m)
    dt: float,         # temporal step (s)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (res_v, res_s):
        res_v = ∂_t v - (1/ρ) ∂_x s   — momentum equation residual
        res_s = ∂_t s - μ ∂_x v        — constitutive equation residual

    Each has shape (batch, NX, NT).
    """
    dv_dt = fourier_derivative_t(v, dt)
    ds_dt = fourier_derivative_t(s, dt)
    dv_dx = fourier_derivative_x(v, dx)
    ds_dx = fourier_derivative_x(s, dx)

    res_v = dv_dt - (1.0 / rho) * ds_dx
    res_s = ds_dt - mu * dv_dx
    return res_v, res_s


def pde_loss(
    pred: torch.Tensor,  # (batch, 4, NX, NT) — (v_l, s_l, v_r, s_r)
    rho: float,
    mu: float,
    dx: float,
    dt: float,
) -> torch.Tensor:
    """
    Scalar PDE loss averaged over both domains.

    Computes mean squared PDE residuals for left and right domains separately
    (each domain has its own v, s) and returns the average.
    """
    v_l, s_l = pred[:, 0], pred[:, 1]    # (batch, NX, NT)
    v_r, s_r = pred[:, 2], pred[:, 3]

    res_vl, res_sl = pde_residual(v_l, s_l, rho, mu, dx, dt)
    res_vr, res_sr = pde_residual(v_r, s_r, rho, mu, dx, dt)

    return (
        res_vl.pow(2).mean() + res_sl.pow(2).mean() +
        res_vr.pow(2).mean() + res_sr.pow(2).mean()
    ) / 4.0


# ---------------------------------------------------------------------------
# Fault boundary condition residual
# ---------------------------------------------------------------------------

def friction_sw(
    slip: torch.Tensor,          # (batch,) or (batch, NT) — total slip D
    params: torch.Tensor,        # (batch, N_params) — [Tau_0, alp_s, alp_d, D_c]
    sigma_n: float = 120.0,      # normal stress (MPa)
) -> torch.Tensor:
    """
    Slip-weakening friction strength τ_SW = σ_n [α_s - (α_s - α_d) min(D/D_c, 1)].

    Returns traction strength in same units as Tau_0 (MPa).
    slip   : (batch,) or (batch, NT)
    params : columns [Tau_0(0), alp_s(1), alp_d(2), D_c(3)]
    """
    alp_s = params[:, 1].unsqueeze(-1)   # (batch, 1)
    alp_d = params[:, 2].unsqueeze(-1)
    D_c   = params[:, 3].unsqueeze(-1)

    healed = torch.clamp(slip / (D_c + 1e-10), max=1.0)
    mu_eff = alp_s - (alp_s - alp_d) * healed
    return sigma_n * mu_eff                # (batch, NT) or (batch,)


def fault_bc_loss(
    pred: torch.Tensor,          # (batch, 4, NX, NT)
    params: torch.Tensor,        # (batch, N_params) — raw or normalised per below
    Tau_0_raw: torch.Tensor,     # (batch,)  raw background stress (MPa)
    sigma_n: float = 120.0,
) -> torch.Tensor:
    """
    Fault interface residual for Slip-Weakening friction:

    1. Traction continuity:   s_l(x_f, t) + s_r(x_f, t) == 0
    2. Friction law:          τ = Tau_0 + s_l(x_f, t) == τ_SW(D, params)
       (τ_SW evaluated at the fault boundary using accumulated slip)

    Returns scalar loss (mean squared residual).
    """
    # Fault is at the right edge of left domain (index -1) and left edge of right domain (index 0)
    s_l_fault = pred[:, 1, -1, :]    # (batch, NT) — stress at x_f from left
    s_r_fault = pred[:, 3, 0, :]     # (batch, NT) — stress at x_f from right
    v_l_fault = pred[:, 0, -1, :]    # (batch, NT) — velocity at x_f from left
    v_r_fault = pred[:, 2, 0, :]     # (batch, NT) — velocity at x_f from right

    # 1. Traction continuity: τ_l + τ_r = 0  (equal and opposite)
    res_continuity = (s_l_fault + s_r_fault).pow(2).mean()

    # 2. Friction law: τ = Tau_0 + s_l = τ_SW
    #    Estimate accumulated slip by integrating slip rate over NT steps
    #    slip_rate = |v_l - v_r| at fault; cumsum as proxy for D(t)
    slip_rate = torch.abs(v_l_fault - v_r_fault)    # (batch, NT)
    # Normalised cumulative slip (delta_T unknown here; this is a relative measure)
    slip_cum  = torch.cumsum(slip_rate, dim=-1)       # (batch, NT)

    tau_pred = Tau_0_raw.unsqueeze(-1) + s_l_fault    # (batch, NT)
    tau_sw   = friction_sw(slip_cum, params, sigma_n) # (batch, NT)
    res_friction = (tau_pred - tau_sw).pow(2).mean()

    return res_continuity + res_friction


# ---------------------------------------------------------------------------
# Relative L2 loss (per-field, used as primary training objective)
# ---------------------------------------------------------------------------

def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Relative L2 loss averaged over fields and batch:
        L = mean_fields( ||pred - target||_2 / (||target||_2 + eps) )

    pred, target : (batch, C, NX, NT)
    """
    diff = (pred - target).pow(2).sum(dim=(-2, -1))          # (batch, C)
    norm = target.pow(2).sum(dim=(-2, -1)).clamp(min=eps)    # (batch, C)
    return (diff / norm).sqrt().mean()
