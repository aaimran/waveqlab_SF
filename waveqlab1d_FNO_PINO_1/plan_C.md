# Plan C: PINO-FNO on 40-Core CPU Cluster

---

## 1. High-Fidelity Data Generation: WaveQLab1d (Numba CPU Parallel)

For a 40-core environment, the strategy uses two levels of parallelism:

- **Inter-simulation** (primary): `multiprocessing.Pool` runs independent parameter
  samples concurrently — each worker executes one full time-domain simulation.
  With 48 workers, the 400-sample train split for one condition runs in ~1/48 of
  serial time.
- **Intra-simulation** (secondary): `@njit(parallel=True)` with `prange` inside
  spatial loops where thread overhead is amortized across NX grid points.

### Simulator Architecture

- **Collocated SBP-SAT grid:** velocity $v$ and stress $\sigma$ are collocated
  (not staggered) on a uniform grid. SBP finite-difference operators provide
  provably stable discretisations; SAT penalties weakly impose boundary conditions.
- **Time integration:** 4th-order Runge-Kutta (RK4); stable under CFL $\leq 0.5$.
- **Anelasticity:** Generalised Standard Linear Solid (GSLS) with $N=4$ Zener
  mechanisms. Each grid point carries $N$ memory variables $\eta_k$ satisfying:

$$\partial_t \eta_k + \frac{\eta_k}{\tau_k} = \frac{w_k}{\tau_k}\,\mu_u\,\partial_x v, \quad k=1,\ldots,4$$

- **Absorbing BCs:** Standard stretched-coordinate PML (20-node layer) or SAT
  impedance condition ($r=0$); free surface via SAT with $r=1$.

### Parallel Workflow (per simulation)

1. **Stress & Anelastic Update** *(RK4 stage, parallel over NX)*
   - Update $\sigma$ and all $\eta_k$ simultaneously from velocity gradient.
   - Apply PML damping for nodes inside the boundary layer.

2. **Interface Prediction** *(serial: split-node)*
   - Compute "free" velocities at the fault node $x_f$ assuming no traction change.

3. **Friction Solve** *(serial)*
   - SW: closed-form solution from slip-weakening law.
   - RS: implicit predictor-corrector for Dieterich aging law state variable $\psi$.

4. **Velocity Update** *(parallel over NX)*
   - Apply corrected traction $\tau$ as SAT penalty at $x_f$; update $v$ globally.

---

## 2. Surrogate Strategy: FNO and PINO

### Architecture

Two plans are implemented, traded off by generality vs. accuracy:

| | Plan A (Unified) | Plan B (Separate) |
|--|--|--|
| Models | 1 `UnifiedFNO2d` | 6 `SeparateFNO2d` |
| Conditioning | FiLM layers (fric×bc embedding) | None — one model per condition |
| $C_\text{in}$ | 22 | 15–19 (depends on condition) |
| Width / Layers | 128 / 6 | 64–96 / 4–5 |
| Use case | Mixed/novel conditions | Best per-condition accuracy |

### Fourier Neural Operator (FNO)

- **Spectral Convolution:** `rfft2` / `irfft2` on the $(x, t)$ plane; two weight
  blocks capture $\pm k_x$ quadrants. Modes are clamped at inference so the same
  model runs at any resolution (resolution-invariant).
- **FiLM Conditioning** (Plan A): discrete $(fric\_law, bc\_mode)$ → learned
  embedding → per-layer $\gamma, \beta$ scale/shift of feature maps.
- **Super-resolution:** Train at $128\times128$; infer at $256\times256$ by
  feeding a finer input grid. Modes are clamped to $\min(m_x, N_x/2)$ so the
  same learned weights apply at any size; `irfft2(s=(256,256))` spectrally
  interpolates the finer grid — no retraining or zero-padding step required.

### Physics-Informed Neural Operator (PINO)

The hybrid loss function is:

$$\mathcal{L} = \mathcal{L}_\text{data}
  + \lambda_\text{pde}\,\mathcal{L}_\text{pde}
  + \lambda_\text{fault}\,\mathcal{L}_\text{fault}
  + \lambda_\text{bc}\,\mathcal{L}_\text{bc}
  + \lambda_\text{ic}\,\mathcal{L}_\text{ic}
  + \lambda_\text{stab}\,\mathcal{L}_\text{stab}$$

| Term | Formula | Default $\lambda$ |
|------|---------|-----------------|
| $\mathcal{L}_\text{data}$ | Relative $L^2$ vs. SBP-SAT ground truth | 1.0 |
| $\mathcal{L}_\text{pde}$ | Residuals of $\rho\partial_t v = \partial_x\sigma$ and $\partial_t\sigma = \mu\partial_x v$ via spectral derivatives | 0.05 → 0.10 |
| $\mathcal{L}_\text{fault}$ | Traction continuity + SW/RS friction law residual at $x_f$ | 0.10 → 0.20 |
| $\mathcal{L}_\text{bc}$ | Outer boundary residual (free / SAT absorbing / PML proxy) | 0.05–0.15 |
| $\mathcal{L}_\text{ic}$ | All fields zero at $t=0$ | 0.05 |
| $\mathcal{L}_\text{stab}$ | SBP-SAT energy stability: $\partial_t E \leq 0$, $E = \tfrac{1}{2}\int(\rho v^2 + \sigma^2/\mu)\,dx$ | 0.02 → 0.05 |

Weights for $\mathcal{L}_\text{pde}$, $\mathcal{L}_\text{fault}$, and $\mathcal{L}_\text{stab}$
are ramped up at epoch 20 (curriculum schedule) to prioritise data fit early
and physics enforcement late.

---

## 3. Multi-Scale Training and Super-Resolution

### Resolutions

Each `.npz` sample stores two versions of every field:

| Key suffix | Grid | Purpose |
|------------|------|---------|
| `_lr` | $128 \times 128$ | Primary training data |
| `_hr` | $256 \times 256$ | Multi-scale training and SR validation |

### Batch-Swap Strategy

Training uses a whole-batch LR/HR swap (implemented in `train_separate.py`):

- With probability `multiscale_hr_frac = 0.3`, a batch is drawn from the HR
  `DataLoader` (`RuptureDataset(res='hr')`).
- Otherwise the batch is drawn from the LR loader.
- The FNO and physics loss are grid-size agnostic — mode clamping ensures
  `SpectralConv2d` handles either size without weight changes.

```yaml
# In each config under training:
multiscale: true
multiscale_hr_frac: 0.3
```

### Physics Weight Ramp

A **single-step** curriculum at epoch 20 ramps physics weights to shift focus
from data fit (early) to physics enforcement (late):

| Epoch | $\lambda_\text{pde}$ | $\lambda_\text{fault}$ | $\lambda_\text{stab}$ |
|-------|---------------------|----------------------|----------------------|
| 0–19 | 0.05 | 0.10 | 0.02 |
| 20 + | 0.10 | 0.20 | 0.05 |

### PDE Derivatives

All physics-loss derivatives ($\partial_x$, $\partial_t$) are computed
**spectrally** in PyTorch — field → `rfft` → multiply by $ik$ → `irfft`. This
is exact for band-limited fields and adds no autograd overhead through the network.

---

## 4. Integrated Optimal Workflow Summary

| Step | Tool | Key Activity |
|------|------|--------------|
| Data Generation | Numba + `multiprocessing` (48 workers) | Parallel SBP-SAT RK4 simulations across all 6 (fric × bc) conditions |
| Pre-Training | PyTorch `SeparateFNO2d` / `UnifiedFNO2d` | Fit data loss on $128\times128$ LR snapshots |
| Physics Correction | PyTorch `PINOLoss` | Add PDE, fault BC, outer BC, IC, and energy stability terms |
| Super-Resolution | Same model, larger input grid | Inference at $256\times256$ without retraining |
| Evaluation | `evaluate.py` | Per-condition rel-$L^2$, error histograms, SR comparison panels |

