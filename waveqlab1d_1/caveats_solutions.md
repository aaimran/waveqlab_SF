# SBP-SAT Energy Stability — Proposed Solutions

## Solution 1: GSLS Anelastic — Complete Energy Proof & Runtime Monitor

**Problem:** The $\eta_l$ cross terms between `hs` and `vx` are not accounted for in the elastic energy norm.

**Solution:** Augment the discrete energy with a weighted memory-variable norm:

$$E_{\text{aug}} = \underbrace{\frac{1}{2}\mathbf{v}^T H_\rho \mathbf{v} + \frac{1}{2}\mathbf{s}^T H_{\mu^{-1}} \mathbf{s}}_{\text{elastic}} + \underbrace{\frac{1}{2}\sum_{l=1}^{4} \frac{\tau_l}{w_l\,\mu\,Q_s^{-1}}\, \|\boldsymbol{\eta}_l\|^2_H}_{\text{memory}}$$

Then compute $\dot{E}_{\text{aug}}$. The memory contribution:

$$\frac{d}{dt}\left(\frac{1}{2}\frac{\tau_l}{w_l \mu Q_s^{-1}}\|\eta_l\|_H^2\right) = \boldsymbol{\eta}_l^T H\,\partial_x \mathbf{v} - \frac{1}{\tau_l}\frac{\tau_l}{w_l \mu Q_s^{-1}}\|\eta_l\|_H^2$$

The $\boldsymbol{\eta}_l^T H\,\partial_x\mathbf{v}$ cross terms exactly cancel against the $-\sum_l \eta_l$ subtracted from `hs_l` in `_anelastic_rhs`, leaving:

$$\dot{E}_{\text{aug}} \leq \text{(boundary terms)} - \underbrace{\sum_l \frac{1}{\tau_l}\frac{\tau_l}{w_l \mu Q_s^{-1}}\|\boldsymbol{\eta}_l\|_H^2}_{\leq\, 0}$$

**Implementation:** `compute_energy()` in `kernels.py` computes $E_{\text{aug}}$ for both elastic and anelastic states. Called every `w_stride` steps in `rupture_1d.py`; values saved in the output `.npz` under key `energy`.

---

## Solution 2: PML — Duru–Kreiss Energy-Bounding Formulation

**Problem:** Volume-damping PML ($-d\,v$, $-d\,s$) falls outside the SBP energy-stability framework.

**Solution (Option B):** Implement the Duru–Kreiss (2012) provably stable PML. The augmented first-order system with auxiliary variables $\tilde{v}, \tilde{s}$:

$$\rho\,\dot{v} = \partial_x(s + \tilde{s}) - d\,\rho\,v, \quad \dot{s} = \mu\,\partial_x v - d\,s, \quad \dot{\tilde{s}} = -d\,\tilde{s}$$

This admits an SBP energy estimate with $d(x) \geq 0$. The total modified energy is:

$$E_{\text{PML}} = \frac{1}{2}\rho\,v^2 + \frac{1}{2\mu}\,(s + \tilde{s})^2 + \frac{1}{2\mu}\,\tilde{s}^2$$

and satisfies $\dot{E}_{\text{PML}} \leq -\text{(boundary dissipation terms)} \leq 0$.

**Implementation:** `init_pml_dk()` (Duru-Kreiss version) in `pml.py` returns damping profile $d(x)$ unchanged; `pml.py` also provides `pml_dk_aux_rate()` for the $\dot{\tilde{s}}$ ODE. `rk4_step_pml_dk()` in `kernels.py` carries $(\tilde{s}_l, \tilde{s}_r)$ as extra state arrays.

---

## Solution 3: Time Integration — Automated CFL Bound

**Problem:** The CFL limit is set by the user with no check against the order-dependent stability region of RK4.

The classical RK4 stability interval on the imaginary axis is $[-2\sqrt{2}\,i,\; +2\sqrt{2}\,i]$.  The spectral radius of the SBP operator $D = H^{-1}Q$ scales as $c_{\max} / (h_{11}\cdot dx)$, where $h_{11}$ is the boundary norm weight.  The safe CFL limits are:

| Order | $h_{11}$ | Empirical CFL max |
|---|---|---|
| 2 | $0.5$ | 1.00 |
| 4 | $17/48 \approx 0.354$ | 0.68 |
| 6 | $13649/43200 \approx 0.316$ | 0.45 |

**Implementation:** `_CFL_MAX` dict added to `rupture_1d.py`; enforced inside `validate()`.  A `warnings.warn()` also fires when `pml=True` without the Duru-Kreiss flag.
