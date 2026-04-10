# Choosing the Optimal CFL Number

Three competing constraints determine the optimal CFL.

---

## 1. Stability (hard limit)

CFL must stay below the scheme's stability limit. For 6th-order SBP-SAT in 1D it is typically **CFL ≤ 1.0**, but the exact limit depends on the SAT penalty coefficients. Exceeding it causes exponential blow-up.

---

## 2. Accuracy (dominant driver)

The numerical dispersion error of finite-difference schemes grows with CFL **and** with frequency. For SBP-SAT the phase velocity error scales roughly as:

$$\frac{\delta c}{c} \propto \left(\frac{\pi}{\text{PPW}}\right)^{2p} \cdot f(\text{CFL})$$

where $p$ is the spatial order. At **CFL = 0.5**, the temporal (Runge-Kutta) error and spatial error are roughly balanced for 4th-order RK + 6th-order SBP — this is the standard "sweet spot."  
- Going **lower** (CFL = 0.2) wastes time steps with no accuracy gain.  
- Going **higher** (CFL > 0.6) lets temporal error dominate.

---

## 3. Cost

$n_t \propto 1/\text{CFL}$, so halving CFL doubles runtime for zero accuracy benefit once temporal error is already sub-dominant.

---

## Practical Decision Rule

| CFL | When to use |
|:---:|-------------|
| 0.3–0.4 | Stiff problems (rough wavespeeds, near-surface layers) where a stability margin is needed |
| **0.5** | **Standard choice** — balanced accuracy/cost for smooth problems |
| 0.6–0.7 | Acceptable if dispersion error is within tolerance and a ~20–30% speedup is desired |

---

## For This Problem

The source time function (nucleation pulse) has a characteristic frequency well below $f_{max}^{practical}$, so accuracy is not CFL-limited — the spatial grid is the bottleneck. **CFL = 0.5 is optimal**: changing it does not improve accuracy but directly scales runtime.

The only reason to lower CFL would be if a very stiff anelastic relaxation timescale $\tau \ll \Delta t$ were introduced, which would require operator-splitting anyway.
