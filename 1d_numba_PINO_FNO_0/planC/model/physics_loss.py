"""
planC/model/physics_loss.py — Plan-C physics loss for SW rupture on 1D domain.

Additions over the original physics_loss.py:
  - energy_stability_loss()  (Plan-C requirement: SBP-SAT energy non-growth)
  - outer_bc_loss()          (free-surface or absorbing at outer edges)
  - ic_loss()                (fields zero at t=0)
  - PINOLoss class with 6-term loss and epoch-20 weight ramp

Derivatives are exact Fourier spectral derivatives (same as original).
All tensors: (batch, channel, NX, NT).
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spectral derivatives  (identical to original, kept self-contained)
# ---------------------------------------------------------------------------

def _deriv_x(u: torch.Tensor, dx: float) -> torch.Tensor:
    """∂_x u via rfft along dim=-2."""
    nx  = u.shape[-2]
    u_f = torch.fft.rfft(u, n=nx, dim=-2, norm='ortho')
    kx  = torch.fft.rfftfreq(nx, d=dx / (2.0 * math.pi)).to(u.device)
    return torch.fft.irfft(u_f * (1j * kx)[..., None], n=nx, dim=-2, norm='ortho')


def _deriv_t(u: torch.Tensor, dt: float) -> torch.Tensor:
    """∂_t u via rfft along dim=-1."""
    nt  = u.shape[-1]
    u_f = torch.fft.rfft(u, n=nt, dim=-1, norm='ortho')
    om  = torch.fft.rfftfreq(nt, d=dt / (2.0 * math.pi)).to(u.device)
    return torch.fft.irfft(u_f * (1j * om), n=nt, dim=-1, norm='ortho')


# ---------------------------------------------------------------------------
# Data loss
# ---------------------------------------------------------------------------

def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """Relative L² averaged over (batch, channels)."""
    diff = (pred - target).pow(2).sum(dim=(-2, -1))
    norm = target.pow(2).sum(dim=(-2, -1)).clamp(min=eps)
    return (diff / norm).sqrt().mean()


# ---------------------------------------------------------------------------
# PDE loss  (elastic momentum + constitutive, both domains)
# ---------------------------------------------------------------------------

def pde_loss(
    pred: torch.Tensor,   # (B, 4, NX, NT)
    rho: float,           # kg/m³
    mu: float,            # Pa
    dx: float,            # m
    dt: float,            # s
) -> torch.Tensor:
    v_l, s_l = pred[:, 0], pred[:, 1]
    v_r, s_r = pred[:, 2], pred[:, 3]

    def domain_pde(v, s):
        res_v = _deriv_t(v, dt) - (1.0 / rho) * _deriv_x(s, dx)
        res_s = _deriv_t(s, dt) - mu * _deriv_x(v, dx)
        return res_v.pow(2).mean() + res_s.pow(2).mean()

    return (domain_pde(v_l, s_l) + domain_pde(v_r, s_r)) * 0.5


# ---------------------------------------------------------------------------
# Fault BC loss  (SW)
# ---------------------------------------------------------------------------

def fault_bc_loss_sw(
    pred: torch.Tensor,        # (B, 4, NX, NT)
    slip_rate_dt: float,       # dt used to integrate slip rate → slip  (m/step)
    Tau_0: torch.Tensor,       # (B,) MPa
    sigma_n: float,            # MPa
    alp_s: torch.Tensor,       # (B,)
    alp_d: torch.Tensor,       # (B,)
    D_c: torch.Tensor,         # (B,) m
) -> torch.Tensor:
    """
    Two residuals:
      1. Traction continuity:  s_l(x_f) + s_r(x_f) = 0
      2. Friction law:         τ_pred = τ_SW(D, params)
    """
    s_l_f = pred[:, 1, -1, :]   # (B, NT) stress at fault from left
    s_r_f = pred[:, 3,  0, :]   # (B, NT) stress at fault from right
    v_l_f = pred[:, 0, -1, :]   # (B, NT) velocity at fault from left
    v_r_f = pred[:, 2,  0, :]   # (B, NT) velocity at fault from right

    # 1. Traction continuity
    res_cont = (s_l_f + s_r_f).pow(2).mean()

    # 2. SW friction law
    slip_rate = (v_l_f - v_r_f).abs()
    D = torch.cumsum(slip_rate, dim=-1) * slip_rate_dt   # (B, NT)

    alp_s_ = alp_s.unsqueeze(-1)
    alp_d_ = alp_d.unsqueeze(-1)
    D_c_   = D_c.unsqueeze(-1)
    xi     = (D / D_c_.clamp(min=1e-10)).clamp(max=1.0)
    tau_sw = sigma_n * (alp_s_ - (alp_s_ - alp_d_) * xi)   # (B, NT)

    tau_pred = Tau_0.unsqueeze(-1) + s_l_f                  # (B, NT)
    res_fric = (tau_pred - tau_sw).pow(2).mean()

    return res_cont + res_fric


# ---------------------------------------------------------------------------
# Outer BC loss  (free-surface or absorbing)
# ---------------------------------------------------------------------------

def outer_bc_loss(
    pred: torch.Tensor,   # (B, 4, NX, NT)
    r0_l: int,            # 0 = absorbing, 1 = free
    r1_r: int,            # 0 = absorbing, 1 = free
    rho: float,
    cs: float,
    dx: float,
    dt: float,
) -> torch.Tensor:
    """
    Free-surface:  s(x_outer, t) = 0
    Absorbing:     v + s / (ρ cs) = 0  (1st-order impedance)
    """
    Z = rho * cs   # impedance
    losses = []

    # Left outer boundary of left domain (index 0)
    v_l0 = pred[:, 0, 0, :]
    s_l0 = pred[:, 1, 0, :]
    if r0_l == 1:
        losses.append(s_l0.pow(2).mean())
    else:
        losses.append((v_l0 + s_l0 / Z).pow(2).mean())

    # Right outer boundary of right domain (index -1)
    v_rN = pred[:, 2, -1, :]
    s_rN = pred[:, 3, -1, :]
    if r1_r == 1:
        losses.append(s_rN.pow(2).mean())
    else:
        losses.append((v_rN - s_rN / Z).pow(2).mean())

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Initial condition loss  (all fields zero at t=0)
# ---------------------------------------------------------------------------

def ic_loss(pred: torch.Tensor) -> torch.Tensor:
    """All four fields should be zero at t=0 (index 0 along the NT axis)."""
    return pred[:, :, :, 0].pow(2).mean()


# ---------------------------------------------------------------------------
# Energy stability loss  (Plan-C addition)
# SBP-SAT discrete energy: E(t) = ½ dx Σ_x [ρ v² + s²/μ]
# Penalise any time step where E increases: relu(dE/dt).
# ---------------------------------------------------------------------------

def energy_stability_loss(
    pred: torch.Tensor,   # (B, 4, NX, NT)
    rho: float,
    mu: float,
    dx: float,
) -> torch.Tensor:
    """
    E(t) = ½ dx Σ_x [ρ v² + s²/μ]  summed over left + right domains.
    Penalise energy growth: L_stab = mean( relu(E[:,1:] - E[:,:-1])² ).
    """
    v_l = pred[:, 0]   # (B, NX, NT)
    s_l = pred[:, 1]
    v_r = pred[:, 2]
    s_r = pred[:, 3]

    E_l = 0.5 * dx * (rho * v_l.pow(2) + s_l.pow(2) / mu).sum(dim=1)   # (B, NT)
    E_r = 0.5 * dx * (rho * v_r.pow(2) + s_r.pow(2) / mu).sum(dim=1)
    E   = E_l + E_r                                                       # (B, NT)

    dE  = E[:, 1:] - E[:, :-1]                                           # (B, NT-1)
    return torch.relu(dE).pow(2).mean()


# ---------------------------------------------------------------------------
# PINOLoss — unified 6-term loss class with epoch-20 ramp
# ---------------------------------------------------------------------------

class PINOLoss(nn.Module):
    """
    6-term Plan-C loss:
      L = w_data*L_data + w_pde*L_pde + w_fault*L_fault
        + w_bc*L_bc + w_ic*L_ic + w_stab*L_stab

    Weights ramp at epoch 20 (curriculum: data-first, then physics).

    Parameters
    ----------
    r0_l, r1_r : int    outer BC flags (0=absorbing, 1=free)
    rho, mu, cs : float material constants (SI)
    """

    def __init__(
        self,
        r0_l     : int   = 0,
        r1_r     : int   = 0,
        rho      : float = 2670.0,   # kg/m³
        mu       : float = None,     # Pa — derived from rho*cs² if None
        cs       : float = 3464.0,   # m/s
        sigma_n  : float = 120.0,    # MPa
        w_data   : float = 1.0,
        w_pde    : float = 0.05,
        w_fault  : float = 0.10,
        w_bc     : float = 0.08,
        w_ic     : float = 0.05,
        w_stab   : float = 0.02,
    ):
        super().__init__()
        self.r0_l    = r0_l
        self.r1_r    = r1_r
        self.rho     = rho
        self.mu      = mu if mu is not None else rho * cs ** 2
        self.cs      = cs
        self.sigma_n = sigma_n
        self.w_data  = w_data
        self.w_pde   = w_pde
        self.w_fault = w_fault
        self.w_bc    = w_bc
        self.w_ic    = w_ic
        self.w_stab  = w_stab

    def update_weights(self, epoch: int):
        """Ramp physics weights after data-fit warm-up (epoch 20)."""
        if epoch == 20:
            self.w_pde   = 0.10
            self.w_fault = 0.20
            self.w_stab  = 0.05

    def forward(
        self,
        pred     : torch.Tensor,   # (B, 4, NX, NT)
        target   : torch.Tensor,   # (B, 4, NX, NT)
        dx       : float,
        dt       : float,
        Tau_0    : torch.Tensor,   # (B,) MPa
        alp_s    : torch.Tensor,   # (B,)
        alp_d    : torch.Tensor,   # (B,)
        D_c      : torch.Tensor,   # (B,) m
    ) -> dict:
        """Returns dict with 'total' and per-term keys."""
        losses = {}

        losses['data']  = relative_l2_loss(pred, target)

        losses['pde']   = pde_loss(pred, self.rho, self.mu, dx, dt)

        losses['fault'] = fault_bc_loss_sw(
            pred, dt, Tau_0, self.sigma_n, alp_s, alp_d, D_c)

        losses['bc']    = outer_bc_loss(
            pred, self.r0_l, self.r1_r, self.rho, self.cs, dx, dt)

        losses['ic']    = ic_loss(pred)

        losses['stab']  = energy_stability_loss(pred, self.rho, self.mu, dx)

        losses['total'] = (
            self.w_data  * losses['data']  +
            self.w_pde   * losses['pde']   +
            self.w_fault * losses['fault'] +
            self.w_bc    * losses['bc']    +
            self.w_ic    * losses['ic']    +
            self.w_stab  * losses['stab']
        )
        return losses


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    B, NX, NT = 2, 80, 80
    pred   = torch.randn(B, 4, NX, NT)
    target = torch.randn(B, 4, NX, NT)
    rho, cs = 2670.0, 3464.0
    mu = rho * cs ** 2
    dx, dt = 80.0, 0.012

    crit = PINOLoss(r0_l=0, r1_r=0, rho=rho, cs=cs)
    L = crit(
        pred, target, dx, dt,
        Tau_0=torch.full((B,), 81.6),
        alp_s=torch.full((B,), 0.677),
        alp_d=torch.full((B,), 0.525),
        D_c=torch.full((B,), 0.4),
    )
    for k, v in L.items():
        print(f'  {k:8s}: {v.item():.6f}')

    # energy_stability_loss standalone
    e = energy_stability_loss(pred, rho, mu, dx)
    print(f'  stab standalone: {e.item():.6f}')
    crit.update_weights(20)
    print(f'  w_stab after epoch-20 ramp: {crit.w_stab}')
