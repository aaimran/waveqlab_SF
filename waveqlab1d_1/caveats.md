# SBP-SAT Energy Stability — Analysis & Caveats

## SBP Operators (`kernels.py` `sbp_dx`)

Standard diagonal-norm operators of orders 2, 4, 6. They satisfy $D = H^{-1}Q$ with:
$$Q + Q^T = B = \text{diag}(-1, 0, \ldots, 0, +1)$$
This is the foundational SBP property. Integration by parts in the discrete energy gives boundary terms only:
$$\dot{E}_{\text{interior}} = -v_0 s_0 + v_N s_N$$

---

## Outer Boundary SAT (`_bc_left`, `_bc_right`)

The penalty uses characteristic variables with **τ₁ = τ₂ = 1 hardcoded** (comment in kernels.py: "standard penalty"). For the left boundary:

$$\text{sat} = \underbrace{\tfrac{1}{2}(z_s v - s)}_{p,\;\text{outgoing}} - r_0 \underbrace{\tfrac{1}{2}(z_s v + s)}_{q,\;\text{incoming}}$$

Computing the energy rate at the left wall (SBP boundary term + SAT contribution):
$$\dot{E}_{\text{left}} = -\frac{(1-r_0)}{2}\, z_s\, v_0^2 \;-\; \frac{(1+r_0)}{2}\, \frac{s_0^2}{z_s}$$

The **cross term** ($v_0 s_0$) **vanishes exactly when τ₁ = τ₂ = 1**, leaving a negative semi-definite form for all $r_0 \in [-1, 1]$:

| BC | r | $\dot{E}_\text{left}$ |
|---|---|---|
| Free surface | 1 | $-s_0^2/z_s \leq 0$ ✓ |
| SAT absorbing | 0 | $-\tfrac{1}{2}(z_s v_0^2 + s_0^2/z_s) \leq 0$ ✓ |
| Clamped | −1 | $-z_s v_0^2 \leq 0$ ✓ |

**Provably energy stable** for the three supported BC options.

---

## Fault Interface SAT (`_interface_kernel`)

The coupling uses Riemann characteristics $(p_0, q_0, p_1, q_1)$ with the friction-resolved target state $(V_m, V_p, T_m, T_p)$. The total power at the interface includes the friction dissipation term $-T \cdot |V^+ - V^-|$. Since both SW and RS laws enforce $T > 0$ and oppose slip, this term is **always non-positive → energy dissipative**.

---

## Caveats

- **GSLS anelastic**: The memory variable ODE $\dot{\eta}_l = (w_l \mu Q_s^{-1} \partial_x v - \eta_l)/\tau_l$ with $\tau_l > 0$ is linearly dissipative, but a complete semi-discrete energy proof for GSLS+SBP requires the analysis in Withers et al. (2015).

- **PML** (`pml.py`): The PML adds volume damping $-d(x)v$, $-d(x)s$ which modifies the equations. This falls **outside the standard SBP energy-stability framework** — no formal proof applies, though it is practically stable and dissipative by construction.

- **Time integration**: RK4 is conditionally stable; the CFL condition (`cfl ≤ 1`) in the input files is a separate requirement for the fully-discrete scheme.
