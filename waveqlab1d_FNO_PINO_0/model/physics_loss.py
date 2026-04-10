"""
physics_loss.py — PINO losses for waveqlab1d FNO (Plans A and B)
=================================================================

All losses operate on FNO output tensors of shape (B, 4, NX, NT):
  channel 0 = v_l(x,t)   left-domain particle velocity  [km/s]
  channel 1 = s_l(x,t)   left-domain shear stress       [MPa]
  channel 2 = v_r(x,t)   right-domain particle velocity [km/s]
  channel 3 = s_r(x,t)   right-domain shear stress      [MPa]

Loss components:
  L_data    — relative L2 between prediction and ground truth
  L_pde     — elastic/anelastic PDE residuals (Fourier-space derivatives)
  L_fault   — fault BC: traction continuity + friction law residual
  L_outer   — outer boundary condition residual
  L_ic      — initial condition (zero at t=0)
  L_stab    — SBP-SAT energy stability (∂_t E ≤ 0, E = ½∫ρv² + s²/μ dx)
  L_total   — weighted sum

Reference: Li et al. (2021); Hao et al. (2023) "GNOT"
"""

import math
import torch
import torch.nn.functional as F


# ─── Spectral derivatives ─────────────────────────────────────────────────────

def _deriv_x(u: torch.Tensor, dx: float) -> torch.Tensor:
    """∂_x u via rfft along dim=-2 (NX).  u: (..., NX, NT)."""
    nx  = u.shape[-2]
    u_f = torch.fft.rfft(u, n=nx, dim=-2, norm='ortho')
    kx  = torch.fft.rfftfreq(nx, d=dx / (2.0 * math.pi)).to(u.device)
    return torch.fft.irfft(u_f * (1j * kx)[..., None], n=nx, dim=-2, norm='ortho')


def _deriv_t(u: torch.Tensor, dt: float) -> torch.Tensor:
    """∂_t u via rfft along dim=-1 (NT).  u: (..., NX, NT)."""
    nt  = u.shape[-1]
    u_f = torch.fft.rfft(u, n=nt, dim=-1, norm='ortho')
    om  = torch.fft.rfftfreq(nt, d=dt / (2.0 * math.pi)).to(u.device)
    return torch.fft.irfft(u_f * (1j * om), n=nt, dim=-1, norm='ortho')


# ─── Data loss ────────────────────────────────────────────────────────────────

def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Relative L2 loss, averaged over batch and channels."""
    diff = pred - target
    return (
        diff.pow(2).sum(dim=(-2, -1)) /
        (target.pow(2).sum(dim=(-2, -1)) + eps)
    ).sqrt().mean()


def data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 averaged over (v_l, s_l, v_r, s_r)."""
    return relative_l2(pred, target)


# ─── PDE residual — elastic ───────────────────────────────────────────────────

def pde_residual_elastic(
    v    : torch.Tensor,   # (B, NX, NT)
    s    : torch.Tensor,   # (B, NX, NT)
    rho  : torch.Tensor,   # (B, NX) or scalar
    mu   : torch.Tensor,   # (B, NX) or scalar
    dx   : float,
    dt   : float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ρ ∂_t v = ∂_x s   →   res_v = ∂_t v - (1/ρ) ∂_x s
    ∂_t s   = μ ∂_x v →   res_s = ∂_t s - μ ∂_x v

    rho, mu can be (B, NX) spatially variable; broadcast over time internally.
    """
    dv_dt = _deriv_t(v, dt)    # (B, NX, NT)
    ds_dt = _deriv_t(s, dt)
    dv_dx = _deriv_x(v, dx)
    ds_dx = _deriv_x(s, dx)

    if isinstance(rho, torch.Tensor) and rho.ndim == 2:
        rho = rho.unsqueeze(-1)   # (B, NX, 1) for broadcasting
        mu  = mu.unsqueeze(-1)

    res_v = dv_dt - ds_dx / rho
    res_s = ds_dt - mu * dv_dx
    return res_v, res_s


def pde_loss_elastic(
    pred : torch.Tensor,   # (B, 4, NX, NT)
    rho  : torch.Tensor,   # (B, NX) or float
    mu   : torch.Tensor,   # (B, NX) or float
    dx   : float,
    dt   : float,
) -> torch.Tensor:
    v_l, s_l = pred[:, 0], pred[:, 1]
    v_r, s_r = pred[:, 2], pred[:, 3]
    rv_l, rs_l = pde_residual_elastic(v_l, s_l, rho, mu, dx, dt)
    rv_r, rs_r = pde_residual_elastic(v_r, s_r, rho, mu, dx, dt)
    return (rv_l.pow(2).mean() + rs_l.pow(2).mean() +
            rv_r.pow(2).mean() + rs_r.pow(2).mean()) / 4.0


# ─── PDE residual — anelastic (GSLS, N=4 mechanisms) ─────────────────────────

def pde_loss_anelastic(
    pred       : torch.Tensor,   # (B, 4, NX, NT)
    eta_l      : torch.Tensor,   # (B, NX, NT, 4) memory variables left domain
    eta_r      : torch.Tensor,   # (B, NX, NT, 4) memory variables right domain
    rho        : torch.Tensor,   # (B, NX) or scalar
    mu_unrelax : torch.Tensor,   # (B, NX) unrelaxed modulus
    tau_mech   : torch.Tensor,   # (4,)  relaxation times
    weight     : torch.Tensor,   # (4,)  quadrature weights
    Qs_inv     : torch.Tensor,   # (B, NX)
    dx         : float,
    dt         : float,
) -> torch.Tensor:
    """
    Anelastic PINO loss.  Uses the two main equations:

    ρ ∂_t v = ∂_x s_eff   where s_eff = s - Σ_k η_k    (effective stress)
    ∂_t s   = μ_u ∂_x v                                 (constitutive, unrelaxed)
    ∂_t η_k = -η_k/τ_k + (w_k/τ_k) μ_u ∂_x v          (memory ODE per mechanism)

    If eta_l / eta_r are not predicted by the FNO, approximate with just the main
    two-equation residual (elastic-like but with μ_unrelax).  Flag `eta_l=None`
    to activate this fallback.
    """
    v_l, s_l = pred[:, 0], pred[:, 1]
    v_r, s_r = pred[:, 2], pred[:, 3]

    # Effective stress = s - Σ_k η_k  (if eta available), else use s directly
    if eta_l is not None:
        s_eff_l = s_l - eta_l.sum(dim=-1)   # (B, NX, NT)
        s_eff_r = s_r - eta_r.sum(dim=-1)
    else:
        s_eff_l = s_l
        s_eff_r = s_r

    rv_l, rs_l = pde_residual_elastic(v_l, s_eff_l, rho, mu_unrelax, dx, dt)
    rv_r, rs_r = pde_residual_elastic(v_r, s_eff_r, rho, mu_unrelax, dx, dt)
    pde = (rv_l.pow(2).mean() + rs_l.pow(2).mean() +
           rv_r.pow(2).mean() + rs_r.pow(2).mean()) / 4.0

    if eta_l is None:
        return pde

    # Memory variable ODEs: ∂_t η_k + η_k/τ_k = (w_k/τ_k) μ_u ∂_x v
    dv_dx_l = _deriv_x(v_l, dx)   # (B, NX, NT)
    dv_dx_r = _deriv_x(v_r, dx)

    if mu_unrelax.ndim == 2:
        mu_u = mu_unrelax.unsqueeze(-1)   # (B, NX, 1)
    else:
        mu_u = mu_unrelax

    eta_loss = torch.tensor(0.0, device=pred.device)
    for k in range(4):
        τk = float(tau_mech[k])
        wk = float(weight[k])
        src_l = (wk / τk) * mu_u * dv_dx_l.unsqueeze(-1)   # (B, NX, NT, 1)
        src_r = (wk / τk) * mu_u * dv_dx_r.unsqueeze(-1)

        deta_dt_l = _deriv_t(eta_l[..., k], dt)   # (B, NX, NT)
        deta_dt_r = _deriv_t(eta_r[..., k], dt)

        res_l = deta_dt_l + eta_l[..., k] / τk - src_l.squeeze(-1)
        res_r = deta_dt_r + eta_r[..., k] / τk - src_r.squeeze(-1)
        eta_loss = eta_loss + res_l.pow(2).mean() + res_r.pow(2).mean()

    return pde + eta_loss / 8.0


# ─── Fault BC residuals ───────────────────────────────────────────────────────

def traction_from_pred(pred: torch.Tensor, Tau_0: float) -> torch.Tensor:
    """
    τ_pred(t) = Tau_0 + s_l(x_fault, t)
    x_fault is the last node (index nx-1) of the left domain.
    pred: (B, 4, NX, NT) → returns (B, NT)
    """
    return Tau_0 + pred[:, 1, -1, :]   # s_l[nx-1, :]


def fault_traction_continuity(pred: torch.Tensor, Tau_0: float) -> torch.Tensor:
    """
    Traction must be equal from left and right:
    τ_0 + s_l[nx-1, t] = τ_0 - s_r[0, t]  →  s_l[nx-1, t] + s_r[0, t] = 0
    Returns mean squared residual.
    """
    s_l_fault = pred[:, 1, -1, :]   # (B, NT)
    s_r_fault = pred[:, 3,  0, :]   # (B, NT)
    return (s_l_fault + s_r_fault).pow(2).mean()


def fault_loss_sw(
    pred    : torch.Tensor,   # (B, 4, NX, NT)
    slip    : torch.Tensor,   # (B, NT)  cumulative slip
    Tau_0   : float,
    sigma_n : float,
    alp_s   : torch.Tensor,   # (B,)
    alp_d   : torch.Tensor,   # (B,)
    D_c     : torch.Tensor,   # (B,)
) -> torch.Tensor:
    """
    SW friction residual:
    τ_SW = σ_n [α_s - (α_s - α_d) min(D/D_c, 1)]
    L_fault = ||τ_pred - τ_SW||² + traction continuity
    """
    tau_pred = traction_from_pred(pred, Tau_0)   # (B, NT)

    # Slip-weakening strength
    ratio = (slip / D_c.unsqueeze(-1)).clamp(max=1.0)   # (B, NT)
    tau_sw = sigma_n * (alp_s.unsqueeze(-1) -
                        (alp_s - alp_d).unsqueeze(-1) * ratio)   # (B, NT)

    friction_res = (tau_pred - tau_sw).pow(2).mean()
    continuity   = fault_traction_continuity(pred, Tau_0)
    return friction_res + continuity


def fault_loss_rs(
    pred     : torch.Tensor,   # (B, 4, NX, NT)
    slip_rate: torch.Tensor,   # (B, NT) |v_l_fault - v_r_fault|
    state    : torch.Tensor,   # (B, NT) ψ(t) from solver
    Tau_0    : float,
    sigma_n  : float,
    f0       : torch.Tensor,   # (B,)
    a        : torch.Tensor,   # (B,)
    V0       : float,
) -> torch.Tensor:
    """
    RS friction residual (Dieterich aging law, quasi-static approximation):
    τ_RS = σ_n [f0 + a ln(V/V0) + ψ]
    Slip rate clipped to avoid log(0).
    """
    tau_pred = traction_from_pred(pred, Tau_0)   # (B, NT)

    V_clipped = slip_rate.clamp(min=1e-12)
    tau_rs    = sigma_n * (
        f0.unsqueeze(-1) +
        a.unsqueeze(-1) * torch.log(V_clipped / V0) +
        state
    )   # (B, NT)

    friction_res = (tau_pred - tau_rs).pow(2).mean()
    continuity   = fault_traction_continuity(pred, Tau_0)
    return friction_res + continuity


# ─── Outer BC residuals ───────────────────────────────────────────────────────

def outer_bc_loss_free(pred: torch.Tensor) -> torch.Tensor:
    """
    Free surface (r=1): s_l[0, t] = 0  (stress-free left outer wall)
                         s_r[-1, t] = 0 (stress-free right outer wall)
    """
    s_l_wall = pred[:, 1, 0, :]    # (B, NT)
    s_r_wall = pred[:, 3, -1, :]   # (B, NT)
    return s_l_wall.pow(2).mean() + s_r_wall.pow(2).mean()


def outer_bc_loss_absorbing(
    pred : torch.Tensor,   # (B, 4, NX, NT)
    rho  : float,
    cs   : float,
) -> torch.Tensor:
    """
    SAT absorbing (r=0): upwind impedance characteristic at outer wall.
    Left wall:  v_l[0, t] - s_l[0, t] / Z = 0    (Z = ρ cs)
    Right wall: v_r[-1, t] + s_r[-1, t] / Z = 0
    """
    Z  = rho * cs
    v_l_wall = pred[:, 0,  0, :]   # (B, NT)
    s_l_wall = pred[:, 1,  0, :]
    v_r_wall = pred[:, 2, -1, :]
    s_r_wall = pred[:, 3, -1, :]
    res_l = (v_l_wall - s_l_wall / Z).pow(2).mean()
    res_r = (v_r_wall + s_r_wall / Z).pow(2).mean()
    return res_l + res_r


def outer_bc_loss_clamped(pred: torch.Tensor) -> torch.Tensor:
    """
    Clamped (r=-1): v_l[0, t] = 0, v_r[-1, t] = 0.
    """
    v_l_wall = pred[:, 0,  0, :]   # (B, NT)
    v_r_wall = pred[:, 2, -1, :]   # (B, NT)
    return v_l_wall.pow(2).mean() + v_r_wall.pow(2).mean()


def outer_bc_loss(
    pred    : torch.Tensor,
    bc_mode : str,            # 'free' | 'absorbing' | 'clamped' | 'pml'
    rho     : float = 2.7,
    cs      : float = 3.464,
) -> torch.Tensor:
    """Dispatch outer BC loss by mode string."""
    if bc_mode == 'free':
        return outer_bc_loss_free(pred)
    elif bc_mode in ('absorbing', 'absorb'):
        return outer_bc_loss_absorbing(pred, rho, cs)
    elif bc_mode == 'clamped':
        return outer_bc_loss_clamped(pred)
    elif bc_mode == 'pml':
        # PML is a volume term; outer BC residual is weaker — use absorbing as proxy
        return outer_bc_loss_absorbing(pred, rho, cs)
    else:
        raise ValueError(f"Unknown bc_mode '{bc_mode}'")


# ─── SBP-SAT energy stability loss ──────────────────────────────────────────

def energy_stability_loss(
    pred : torch.Tensor,   # (B, 4, NX, NT)
    rho  : torch.Tensor,   # (B, NX)  density
    mu   : torch.Tensor,   # (B, NX)  shear modulus
    dx   : float,
) -> torch.Tensor:
    """
    Penalise positive time-growth of the discrete elastic energy:

        E(t) = ½ Σ_x [ ρ(x) v(x,t)² + s(x,t)²/μ(x) ] dx

    Stability requires ∂_t E ≤ 0 in the interior (boundary terms handled by BCs).
    We penalise the mean of max(∂_t E, 0) across the batch.

    This enforces the SBP-SAT energy norm constraint:

        ∂_t||u||²_H ≤ boundary terms  (must be non-positive for stability)
    """
    v_l, s_l = pred[:, 0], pred[:, 1]   # (B, NX, NT)
    v_r, s_r = pred[:, 2], pred[:, 3]

    rho_ = rho.unsqueeze(-1)             # (B, NX, 1) → broadcast over NT
    mu_  = mu.unsqueeze(-1)

    # Discrete energy at each time step: E(t) = ½ Σ_x [...] dx
    E_l = 0.5 * dx * (rho_ * v_l.pow(2) + s_l.pow(2) / mu_).sum(dim=1)  # (B, NT)
    E_r = 0.5 * dx * (rho_ * v_r.pow(2) + s_r.pow(2) / mu_).sum(dim=1)  # (B, NT)
    E   = E_l + E_r                      # (B, NT)

    # Time derivative of energy: ΔE / Δt  (forward difference)
    dE_dt = E[:, 1:] - E[:, :-1]        # (B, NT-1)

    # Penalise only positive growth (violations of ∂_t E ≤ 0)
    return torch.relu(dE_dt).pow(2).mean()


# ─── Initial condition loss ───────────────────────────────────────────────────

def ic_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    All fields must be zero at t=0 (quiescent initial condition).
    pred: (B, 4, NX, NT) → penalise pred[:, :, :, 0]
    """
    return pred[:, :, :, 0].pow(2).mean()


# ─── Combined PINO loss ───────────────────────────────────────────────────────

class PINOLoss(torch.nn.Module):
    """
    Configurable PINO loss for Plans A and B.

    Parameters
    ----------
    fric_law  : 'SW' | 'RS'
    bc_mode   : 'free' | 'absorbing' | 'pml' | 'clamped'
    response  : 'elastic' | 'anelastic'
    w_data    : float   weight for data loss
    w_pde     : float   weight for PDE residual
    w_fault   : float   weight for fault BC loss
    w_outer   : float   weight for outer BC loss
    w_ic      : float   weight for IC loss
    """

    def __init__(
        self,
        fric_law : str   = 'SW',
        bc_mode  : str   = 'absorbing',
        response : str   = 'elastic',
        w_data   : float = 1.0,
        w_pde    : float = 0.05,
        w_fault  : float = 0.1,
        w_outer  : float = 0.1,
        w_ic     : float = 0.05,
        w_stab   : float = 0.02,
    ):
        super().__init__()
        self.fric_law = fric_law.upper()
        self.bc_mode  = bc_mode.lower()
        self.response = response.lower()
        self.w_data   = w_data
        self.w_pde    = w_pde
        self.w_fault  = w_fault
        self.w_outer  = w_outer
        self.w_ic     = w_ic
        self.w_stab   = w_stab

    def update_weights(self, epoch: int):
        """Ramp up physics weights after warm-up phase."""
        if epoch == 20:
            self.w_pde   = 0.1
            self.w_fault = 0.2
            self.w_stab  = 0.05

    def forward(
        self,
        pred    : torch.Tensor,        # (B, 4, NX, NT)
        target  : torch.Tensor,        # (B, 4, NX, NT)
        batch   : dict,                # collateral data from dataset
        dx      : float,
        dt      : float,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with 'total' key plus individual component keys."""
        losses = {}

        # ── Data loss
        losses['data'] = data_loss(pred, target)

        # ── PDE loss
        rho = batch['rho']   # (B, NX)
        mu  = batch['mu']    # (B, NX)
        if self.response == 'elastic':
            losses['pde'] = pde_loss_elastic(pred, rho, mu, dx, dt)
        else:
            # Anelastic: use mu_unrelax; eta fields optional
            mu_u      = batch.get('mu_unrelax', mu)
            eta_l_b   = batch.get('eta_l', None)
            eta_r_b   = batch.get('eta_r', None)
            tau_mech  = batch.get('tau_mech', None)
            weight    = batch.get('weight', None)
            Qs_inv    = batch.get('Qs_inv', None)
            losses['pde'] = pde_loss_anelastic(
                pred, eta_l_b, eta_r_b, rho, mu_u,
                tau_mech, weight, Qs_inv, dx, dt)

        # ── Fault BC loss
        Tau_0   = float(batch['Tau_0'][0])
        sigma_n = float(batch['sigma_n'][0])
        if self.fric_law == 'SW':
            slip = batch['slip']   # (B, NT)
            losses['fault'] = fault_loss_sw(
                pred, slip, Tau_0, sigma_n,
                batch['alp_s'], batch['alp_d'], batch['D_c'])
        else:
            slip_rate = batch['sliprate']   # (B, NT)
            state     = batch.get('psi', torch.zeros_like(slip_rate))
            losses['fault'] = fault_loss_rs(
                pred, slip_rate, state, Tau_0, sigma_n,
                batch['f0'], batch['a'], float(batch['V0'][0]))

        # ── Outer BC loss
        cs_mean = float(batch.get('cs', torch.tensor(3.464))[0])
        rho_mean = float(rho[:, 0].mean())
        losses['outer_bc'] = outer_bc_loss(pred, self.bc_mode, rho_mean, cs_mean)

        # ── IC loss
        losses['ic'] = ic_loss(pred)

        # ── SBP-SAT energy stability loss
        losses['stab'] = energy_stability_loss(pred, rho, mu, dx)

        # ── Total
        losses['total'] = (
            self.w_data  * losses['data'] +
            self.w_pde   * losses['pde'] +
            self.w_fault * losses['fault'] +
            self.w_outer * losses['outer_bc'] +
            self.w_ic    * losses['ic'] +
            self.w_stab  * losses['stab']
        )
        return losses
