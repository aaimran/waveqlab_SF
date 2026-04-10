# Plan: Upwind and Upwind-DRP SBP Stencils

## Current State

`sbp_dx()` in `kernels.py` implements one operator family: the **standard diagonal-norm central SBP** $D = H^{-1}Q$. Interior stencils are anti-symmetric (zero numerical dissipation):

| Order | Interior stencil | Width | Dissipation |
|---|---|---|---|
| 2 | $(u_{i+1} - u_{i-1})/(2h)$ | 3-pt | none |
| 4 | $(-u_{i-2}+8u_{i-1}-8u_{i+1}+u_{i+2})/(12h)$ | 5-pt | none |
| 6 | 7-pt anti-symmetric | 7-pt | none |

There is no `stencil` key in the input parser or DEFAULTS.

---

## Mathematical Foundation

### Upwind SBP (Mattsson 2017)

Two one-sided operators $D_\pm = H^{-1}(Q \pm R/2)$ where:
- $Q + Q^T = B$ — same SBP property as central
- $R = R^T \geq 0$ in the $H$-inner product — symmetric positive semi-definite dissipation
- $(D_+ + D_-)/2 = D_c$ — average equals central operator
- $\Sigma := (D_- - D_+)/2 = H^{-1}R/2 \geq 0$ — the upwind dissipation operator

**Boundary stencils are identical to the central operator of the same order.** Only interior stencils differ.

### Characteristic splitting for the elastic wave system

For $\rho v_t = s_x$, $\ s_t = \mu v_x$ with Riemann variables $p = z_s v + s$ (right-going, $+c$) and $q = z_s v - s$ (left-going, $-c$):

$$p_t = c\, D_- p, \qquad q_t = -c\, D_+ q$$

Converting back to $(v, s)$ yields the **upwind elastic RHS**:

$$\rho\, \dot{v}_i = (D_c\, s)_i + \rho c\,(\Sigma v)_i$$
$$\dot{s}_i = \mu\,(D_c v)_i + c\,(\Sigma s)_i$$

Since $\Sigma \geq 0$ in the $H$-norm, the correction terms $\rho c\,\|\Sigma^{1/2} v\|_H^2$ and $c/\mu\,\|\Sigma^{1/2} s\|_H^2$ are always non-negative energy sinks — **provable stability without SAT in the interior**.

### DRP optimization (`upwind_drp`)

Replaces the algebraically-derived interior stencil with coefficients minimizing integrated dispersion error:

$$\min_{a_j} \int_0^{\xi_c} \left|\tilde{k}(\xi) - \xi\right|^2 d\xi, \qquad \tilde{k}(\xi) = h\sum_j a_j\, e^{ij\xi}$$

- Sacrifices one algebraic order ($p-1$ instead of $p$) for accurate dispersion over a wider wavenumber band
- Maintains upwind bias (real part of transfer function provides dissipation)
- Boundary stencils unchanged — energy stability preserved

---

## Extension to Anelastic (GSLS)

The anelastic RHS adds memory variables $\eta_l$ to the stress rate:

$$\dot{s}_i = \mu_{\text{unrelax}}\,(D_c v)_i + c\,(\Sigma s)_i - \sum_l \eta_{i,l}$$
$$\dot{\eta}_{i,l} = \frac{w_l\,\mu\,Q_s^{-1}\,(D_c v)_i - \eta_{i,l}}{\tau_l}$$

**Key rule: the $\eta$ ODE forcing must use $D_c v$ (central part only), not $D_- v$ or the full upwind derivative.**

This preserves the cross-cancellation in the augmented energy proof. Differentiating the memory energy:

$$\frac{d}{dt}\!\left(\frac{\tau_l}{2 w_l \mu Q_s^{-1}}\|\boldsymbol{\eta}_l\|_H^2\right) = \boldsymbol{\eta}_l^T H\,(D_c v) - \frac{1}{\tau_l}\frac{\tau_l}{w_l \mu Q_s^{-1}}\|\boldsymbol{\eta}_l\|_H^2$$

The $+\boldsymbol{\eta}_l^T H\,(D_c v)$ cross-term cancels against $-\sum_l \eta_l$ in $\dot{s}$ through the SBP summation-by-parts identity. The upwind correction $c(\Sigma s)$ in $\dot{s}$ adds **additional dissipation** without disturbing this cancellation.

**Result:** The augmented energy estimate holds for the upwind anelastic system:

$$\dot{E}_{\text{aug}} \leq \text{(boundary dissipation)} - \underbrace{c\,\mathbf{v}^T H \rho\Sigma v - c\,\mathbf{s}^T H \mu^{-1}\Sigma s}_{\text{upwind interior dissipation}\,\geq\,0} - \underbrace{\sum_l \frac{1}{\tau_l}\frac{\tau_l}{w_l \mu Q_s^{-1}}\|\boldsymbol{\eta}_l\|_H^2}_{\text{GSLS dissipation}\,\geq\,0} \leq 0$$

For the **anelastic** `_anelastic_rates` kernel the modification is thus:
1. Compute `D_c v` (from `sbp_dx_pm` average) and `Sigma s`, `Sigma v`
2. `hv[i] += rho * c * Sigma_v[i]` — upwind velocity correction
3. `hs[i] += c * Sigma_s[i]` — upwind stress correction
4. `_anelastic_rhs(...)` is called with `D_c v` (not `D_- v`) unchanged

---

## Implementation Plan

### Phase 0 — New input key (no kernel changes)

**File:** `rupture_1d.py`

Add:
```python
stencil = 'central'   # 'central' | 'upwind' | 'upwind_drp'
```
to `DEFAULTS` and `_str_keys`. Add to `validate()`:
```python
VALID_STENCILS = ('central', 'upwind', 'upwind_drp')
# upwind implemented for orders 2, 4 only
# upwind_drp requires order=4
```
Update `_CFL_MAX` with upwind limits:

| Order | Central | Upwind | Upwind-DRP |
|---|---|---|---|
| 2 | 1.00 | ~0.85 | — |
| 4 | 0.68 | ~0.60 | ~0.45 |

---

### Phase 1 — Upwind operator pair `sbp_dx_pm` in `kernels.py`

**File:** `src/kernels.py`

```python
@njit(parallel=True, cache=True)
def sbp_dx_pm(dp, dm, u, nx, dx, order):
    """
    Compute D_+ u (dp) and D_- u (dm) in-place.
    Boundary stencils identical to sbp_dx of same order.
    Interior: dm is left-biased, dp is right-biased (dp[i] = -dm_reversed[i]).
    """
```

Coefficients from Mattsson (2017), Tables 1–3.

Boundary rows for $D_-$ order 2 and 4 are **identical** to `sbp_dx` for those orders (same $H$ diagonal, same $Q$ rows at boundary).

Interior stencil for $D_-$:
- Order 2: backward $(u_i - u_{i-1})/h$
- Order 4: 5-point left-biased (from Table 2, Mattsson 2017)

$D_+ u_i$ is obtained by reflection: $D_+ u_i = -(-1)\cdot D_-\tilde{u}_i$ on the reversed grid (anti-symmetry).

Also add a companion:
```python
@njit(parallel=True, cache=True)
def sbp_dx_sigma(sigma_u, u, nx, dx, order):
    """Compute Sigma u = (D_- u - D_+ u)/2 in-place (upwind dissipation)."""
```

---

### Phase 2 — Upwind elastic RHS `_elastic_rate_upwind`

**File:** `src/kernels.py`

```python
@njit(cache=True)
def _elastic_rate_upwind(hv_l, hs_l, v_l, s_l,
                         hv_r, hs_r, v_r, s_r,
                         slip, psi,
                         rho, mu, nx, dx, order, r0_l, r1_r,
                         d_l, d_r, friction_params):
    cs = sqrt(mu / rho)
    h11 = _penalty_weight(order, dx)
    sigma_v_l, sigma_v_r = empty(nx), empty(nx)
    sigma_s_l, sigma_s_r = empty(nx), empty(nx)
    vx_l, sx_l = empty(nx), empty(nx)   # central D_c v, D_c s
    vx_r, sx_r = empty(nx), empty(nx)

    sbp_dx(vx_l, v_l, ...)           # central derivative for stress rate + eta ODE
    sbp_dx(sx_l, s_l, ...)
    sbp_dx_sigma(sigma_v_l, v_l, ...) # upwind dissipation for velocity equation
    sbp_dx_sigma(sigma_s_l, s_l, ...) # upwind dissipation for stress equation

    for i in range(nx):
        hv_l[i] = (1/rho)*sx_l[i] + cs*sigma_v_l[i] - d_l[i]*v_l[i]
        hs_l[i] = mu*vx_l[i]       + cs*sigma_s_l[i] - d_l[i]*s_l[i]
    ...
    # SAT BCs and interface kernel unchanged (same h11, same characteristics)
```

---

### Phase 3 — Upwind anelastic RHS `_anelastic_rates_upwind`

**File:** `src/kernels.py`

Modifies `_anelastic_rates` to pass the **central** `vx` to `_anelastic_rhs`:

```python
# central derivative used in eta ODE (preserves cross-cancellation)
sbp_dx(vx_l, v_l, ...)         # D_c v  → into _anelastic_rhs

# upwind corrections added to stress rate only
sbp_dx_sigma(sigma_s_l, s_l, ...)
for i in range(nx):
    hs_l[i] += cs * sigma_s_l[i]     # adds after central elastic rate

# _anelastic_rhs called with D_c v, unchanged
_anelastic_rhs(hs_l, Deta_l, eta_l, vx_l, mu_arr, Qs_inv, tau, weight, nx)
```

---

### Phase 4 — DRP interior stencil constants

**File:** `src/kernels.py` (module-level constants)

11-point DRP interior stencil coefficients (Ålund & Nordström 2019, upwind-biased, optimized to $\xi_c = \pi/2$):

```python
_DRP_DM_INT = np.array([...11 coefficients...], dtype=np.float64)  # D_- interior
_DRP_DP_INT = -_DRP_DM_INT[::-1]                                    # D_+ by symmetry
```

Minimum grid size increases: 11-point stencil requires ≥5 interior guard points from each boundary. Validate with `nx >= 2*(n_bdry + 5) + 1`.

`sbp_dx_pm` gains a `drp` branch triggered by `stencil == 'upwind_drp'` passed as integer flag.

---

### Phase 5 — RK4 dispatch

**File:** `src/kernels.py`, `rupture_1d.py`

Add integer constants:
```python
STENCIL_CENTRAL    = np.int64(0)
STENCIL_UPWIND     = np.int64(1)
STENCIL_UPWIND_DRP = np.int64(2)
```

`rk4_step` and `rk4_step_anelastic` gain a `stencil` argument.  
Dispatch table (inside `_elastic_rate` / `_anelastic_rates`):

```
stencil == 0 → current _elastic_rate (central)
stencil == 1 → _elastic_rate_upwind  (upwind, algebraic order)
stencil == 2 → _elastic_rate_upwind  (upwind, DRP interior)
```

---

### Phase 6 — Verification tests

New input files:
- `input/test-upwind_order2.in`, `test-upwind_order4.in` — SW problem, compare central vs upwind seismograms (agree to truncation error)
- `input/test-drp_order4.in` — same; check DRP reduces dispersion vs central order 4 at identical grid count

New auxiliary script:
```
python3 auxiliary/plot_dispersion.py --order 4 --stencil central|upwind|upwind_drp
```
Plots modified wavenumber $\tilde{k}(k)$ of each operator for visual comparison.

---

## Summary

| Component | File | New symbols |
|---|---|---|
| Input key + validation + CFL limits | `rupture_1d.py` | `stencil`, `VALID_STENCILS`, `_CFL_MAX` extended |
| $D_+$/$D_-$ and $\Sigma$ operators | `src/kernels.py` | `sbp_dx_pm()`, `sbp_dx_sigma()` |
| Upwind elastic RHS | `src/kernels.py` | `_elastic_rate_upwind()` |
| Upwind anelastic RHS | `src/kernels.py` | `_anelastic_rates_upwind()` |
| DRP stencil constants | `src/kernels.py` | `_DRP_DM_INT`, `_DRP_DP_INT` |
| RK4 dispatch | `src/kernels.py` | `stencil` arg to `rk4_step`, `rk4_step_anelastic` |
| Dispersion tests | `input/`, `auxiliary/` | test `.in` files, `plot_dispersion.py` |

### Critical constraint for anelastic

The $\eta$ ODE forcing always uses the **central** $D_c v$ regardless of `stencil`. The upwind correction $c\,\Sigma s$ enters $\dot{s}$ only, after `_anelastic_rhs` computes the memory-variable contribution. This preserves the augmented energy cross-cancellation proved in `caveats_solutions.md`.
