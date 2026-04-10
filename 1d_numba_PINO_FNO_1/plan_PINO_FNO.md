# Plan: PINO-FNO for 1D Dynamic Rupture Simulation

**Date:** 2026-04-03  
**Simulator:** `1d_numba/rupture_1d.py` — 1D SBP-SAT elastic wave, RK4, SW/RS friction  
**Goal:** Train a Fourier Neural Operator (FNO) with physics-informed (PINO) loss to emulate the
rupture simulator: given fault/material parameters, predict the full space-time solution 100–1000×
faster than even the Numba version.

---

## 1. Problem Formulation

### Operator to learn

$$\mathcal{G}_\theta : \mathbf{p} \;\mapsto\; \bigl(v_l(x,t),\; s_l(x,t),\; v_r(x,t),\; s_r(x,t)\bigr)$$

where $\mathbf{p}$ is the set of physical input parameters (see §3) and the output is the full
space-time solution on both elastic domains.

Derived fault quantities (slip, slip rate, traction) are post-processed from the domain boundaries.

### Governing equations (residuals used in PINO loss)

**Elastic wave (velocity-stress form):**

$$\frac{\partial v}{\partial t} = \frac{1}{\rho}\frac{\partial s}{\partial x}, \qquad
  \frac{\partial s}{\partial t} = \mu\frac{\partial v}{\partial x}$$

**Fault interface (SW friction):**

$$\tau = \mathrm{friction}(D,\,\sigma_n) = \sigma_n\bigl[\alpha_s - (\alpha_s-\alpha_d)\min(D/D_c,\,1)\bigr]$$

$$\tau = \tau_0 + s_l(x_f,t) = \tau_0 - s_r(x_f,t) \quad \text{(continuity of traction)}$$

PDE residuals are cheap to evaluate in Fourier space — the key PINO advantage over PINNs.

---

## 2. Architecture: FNO-2D

Treat the solution as a function on the 2D grid $(x, t) \in [0, L] \times [0, T]$.

```
Input feature map  a(x) ∈ R^{nx × C_in}
        │
        ▼
  Lifting layer  (linear, R^{C_in} → R^{d_v})
        │
   ┌────┴─────────────────────────────────┐
   │  FNO layer × L_layers                │
   │  ┌──────────────────────────────┐    │
   │  │  FFT2D  →  R_φ (freq filter) │    │
   │  │  + W (local linear bypass)   │    │
   │  │  → σ(·)  (GeLU activation)   │    │
   │  └──────────────────────────────┘    │
   └────────────────────────────────────┘
        │
  Projection layer  (R^{d_v} → R^{C_out})
        │
        ▼
Output  u(x,t) ∈ R^{nx × nt × C_out}
```

- **C_in**: parameter channels broadcast over the spatial grid (see §3)
- **C_out = 4**: (v_l, s_l, v_r, s_r) — alternatively train two separate FNOs for left/right
- **FFT2D**: modes kept = (k_x, k_t) → controlled by `modes_x`, `modes_t` hyperparameters
- **L_layers = 4**, **d_v = 64** (starting point; tune via ablation)

### Alternative: FNO-1D with time-stepping (autoregressive)

Instead of predicting all $t$ at once, predict $u(x, t_{n+1})$ from $u(x, t_n)$. Pros: lower
memory; Cons: error accumulates over long rollouts, harder to enforce physics BC.

**Recommendation:** Start with FNO-2D. Switch to autoregressive if memory exceeds GPU budget.

---

## 3. Input Parameter Space

Each training sample is a draw from the following parameter ranges:

| Parameter | Symbol | Baseline | Range | Notes |
|---|---|---|---|---|
| Background stress | `Tau_0` | 81.60 MPa | [75, 88] MPa | Primary driver of rupture style |
| Static friction coeff | `alp_s` | 0.677 | [0.60, 0.75] | |
| Dynamic friction coeff | `alp_d` | 0.525 | [0.45, 0.60] | must satisfy alp_d < alp_s |
| Critical slip distance | `D_c` | 0.4 m | [0.2, 0.8] m | Controls transition sharpness |
| Normal stress | `sigma_n` | 120 MPa | [100, 140] MPa | |
| Shear wave velocity | `cs` | 3.464 km/s | fixed | Physical constraint |
| Density | `rho` | 2.67 g/cm³ | fixed | Physical constraint |

**Start with 3–4 free parameters** (Tau_0, alp_s, alp_d, D_c) to keep the dataset manageable.
Expand to full parameter space once the single-parameter FNO is validated.

The parameter vector $\mathbf{p} \in \mathbb{R}^{N_p}$ is broadcast to a constant field over
$(x, t)$ and concatenated with the spatial/temporal coordinate channels as input to the FNO.

---

## 4. Dataset

### 4.1 Generation

- Use `1d_numba/rupture_1d.py -np 1` (1868× faster than Python serial)
- Latin Hypercube Sampling (LHS) over the parameter space
- Save full domain output (all `iplot` snapshots) per sample
- Each sample: `output/rupture_SW_{run_id}.npz` — already has `DomainOutput_l`, `DomainOutput_r`

```
data/
  train/  N_train = 2000 samples
  val/    N_val   =  200 samples
  test/   N_test  =  200 samples
```

**Estimated generation time:** 2000 × 0.09 s ≈ 3 min on current 40-CPU node.

### 4.2 Sub-sampling for FNO input

The raw output has shape `(nx, nt_stored, 2)`. For FNO training:
- Downsample to `(nx_fno, nt_fno)` e.g. `(64, 64)` or `(128, 128)`
- Use bilinear interpolation or strided indexing

### 4.3 Normalization

Normalize each field independently (zero-mean, unit-std computed from training set).

---

## 5. Loss Function

### Data loss (supervised)

$$\mathcal{L}_\text{data} = \frac{1}{N}\sum_{i=1}^{N} \left\| \mathcal{G}_\theta(\mathbf{p}_i) - u_i \right\|^2_2$$

### Physics residual loss (PINO)

Evaluate PDE residuals **in Fourier space** (exact derivatives, no finite differences):

$$\mathcal{L}_\text{PDE} = \frac{1}{N}\sum_{i=1}^{N} \left\| \partial_t v_i^\theta - \frac{1}{\rho}\partial_x s_i^\theta \right\|^2 + \left\| \partial_t s_i^\theta - \mu\,\partial_x v_i^\theta \right\|^2$$

Derivatives via FFT: $\partial_x u \leftrightarrow ik_x\hat{u}$,  $\partial_t u \leftrightarrow i\omega\hat{u}$.

### Boundary/interface loss

$$\mathcal{L}_\text{BC} = \left\| \tau^\theta - \mathrm{friction}(D^\theta, \mathbf{p}) \right\|^2 + \left\| \tau_l^\theta - \tau_r^\theta \right\|^2$$

### Total loss

$$\mathcal{L} = \mathcal{L}_\text{data} + \lambda_\text{PDE}\,\mathcal{L}_\text{PDE} + \lambda_\text{BC}\,\mathcal{L}_\text{BC}$$

Start with $\lambda_\text{PDE} = 0$ (pure supervised FNO), then increase to introduce physics.

---

## 6. Implementation Phases

### Phase 1 — Data generation pipeline  *(~1 day)*

**New file:** `data_gen/generate_dataset.py`

- LHS sampling of parameter space
- Batch-runs simulator via `subprocess` or direct `run_sim()` call
- Stores samples in `data/train/`, `data/val/`, `data/test/`
- `generate_dataset.py --n_train 2000 --n_val 200 --n_test 200`

**New file:** `data_gen/dataset.py`

- `RuptureDataset(torch.utils.data.Dataset)`: loads `.npz`, applies normalization, returns
  `(params_tensor, solution_tensor)` pairs
- `Normalizer` class: fit on train, apply to val/test consistently

---

### Phase 2 — FNO model  *(~2 days)*

**New file:** `model/fno.py`

```python
class SpectralConv2d(nn.Module):
    """Fourier integral operator layer: FFT2 → complex weight multiply → IFFT2."""

class FNOBlock2d(nn.Module):
    """Single FNO layer: SpectralConv2d + bypass linear + activation."""

class FNO2d(nn.Module):
    """Full FNO: lifting → N×FNOBlock2d → projection.
    Input:  (batch, nx, nt, C_in)
    Output: (batch, nx, nt, C_out)
    """
```

Key hyperparameters:
```
modes_x  = 16      # Fourier modes in x
modes_t  = 16      # Fourier modes in t
width    = 64      # channel width d_v
n_layers = 4       # number of FNO blocks
C_in     = N_p+2   # params + (x,t) coordinate channels
C_out    = 4       # (v_l, s_l, v_r, s_r)
```

---

### Phase 3 — Physics loss  *(~1 day)*

**New file:** `model/physics_loss.py`

```python
def pde_residual(v, s, rho, mu, dx, dt):
    """Compute PDE residuals via Fourier-space differentiation."""
    # torch.fft.rfft2 → multiply by wavenumbers → torch.fft.irfft2

def friction_residual(v_l_boundary, s_l_boundary, v_r_boundary, s_r_boundary,
                      friction_params):
    """Evaluate fault boundary condition residual."""
```

---

### Phase 4 — Training loop  *(~1 day)*

**New file:** `train.py`

```python
# python3 train.py --config configs/fno_sw_baseline.yaml
```

- AdamW optimizer, cosine LR schedule
- Mixed precision (AMP) for GPU training
- Checkpoint saving every N epochs
- W&B / TensorBoard logging: train loss, val loss, PDE residual, field RMSE
- Validation: relative L2 error per field, visual comparison every M epochs

**New file:** `configs/fno_sw_baseline.yaml`

---

### Phase 5 — Evaluation  *(~0.5 day)*

**New file:** `evaluate.py`

```python
# python3 evaluate.py --checkpoint checkpoints/best.pt --test-data data/test/
```

Metrics:
- Relative L2 error: $\varepsilon = \|u^\theta - u\|_2 / \|u\|_2$ per field
- Peak slip-rate error
- Rupture arrival time error
- Inference time vs. simulator time

---

## 7. File / Directory Structure

```
1d_numba_PINO_FNO/
├── rupture_1d.py              # (existing) Numba simulator
├── src/                       # (existing) simulation kernels
├── input/                     # (existing) .in parameter files
├── output/                    # (existing) simulation .npz outputs
│
├── data_gen/
│   ├── generate_dataset.py    # LHS sampling + batch simulation
│   └── dataset.py             # PyTorch Dataset + Normalizer
│
├── model/
│   ├── fno.py                 # FNO2d, FNOBlock2d, SpectralConv2d
│   └── physics_loss.py        # PDE + BC residuals in Fourier space
│
├── configs/
│   └── fno_sw_baseline.yaml   # all hyperparameters
│
├── data/
│   ├── train/                 # 2000 × rupture_SW_{id}.npz
│   ├── val/                   # 200 samples
│   └── test/                  # 200 samples
│
├── checkpoints/               # model weights + optimizer state
├── train.py                   # training entry point
└── evaluate.py                # evaluation + error metrics
```

---

## 8. Dependencies

```
torch >= 2.0          # FNO, AMP, fft support
scipy                 # LHS sampling (scipy.stats.qmc.LatinHypercube)
numpy                 # already present
matplotlib            # already present
pyyaml                # config files
wandb                 # (optional) experiment tracking
```

---

## 9. Development Sequence

| Step | Task | Output |
|---|---|---|
| 1 | `data_gen/generate_dataset.py` | 2000+400 `.npz` files in `data/` |
| 2 | `data_gen/dataset.py` | `RuptureDataset`, `Normalizer` |
| 3 | `model/fno.py` sanity check | forward pass on random input → correct shape |
| 4 | `train.py` data-loss only ($\lambda_\text{PDE}=0$) | first FNO trained |
| 5 | Evaluate on test set — target $\varepsilon < 5\%$ | baseline accuracy |
| 6 | `model/physics_loss.py` + add PINO loss | PINO-FNO trained |
| 7 | Compare PINO vs FNO at equal data budget | accuracy vs. data efficiency |
| 8 | Add Rate-and-State friction (RS) samples | multi-friction-law FNO |

---

## 10. Expected Performance

| Method | 577-step simulation | Notes |
|---|---|---|
| Python serial | 160 s | baseline |
| Numba np=1 | 0.09 s | 1868× faster |
| FNO inference | ~0.001 s | ~90,000× faster (GPU); one forward pass |

The FNO will trade some accuracy for >4 orders of magnitude speedup, making it suitable for
large-scale ensemble inference (uncertainty quantification, inversion).
