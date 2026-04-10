# Plan A — Unified PINO-FNO (all friction laws + BCs in one model)

**Simulator:** `waveqlab1d/rupture_1d.py` — 1D SBP-SAT elastic/anelastic, RK4, SW/RS friction  
**Goal:** One operator  G_θ : p → (v_l, s_l, v_r, s_r)(x, t)  that generalises across all
boundary conditions and friction laws, trained with PINO (data + physics residuals).

---

## 1. Why one model?

| Argument for unified | Argument against |
|---|---|
| Shares feature representations across physics regimes | Larger input space, harder to train |
| Cross-condition generalization (BC transfer) | Need balanced data across all regimes |
| Single deployment artifact | Scale of model must accommodate all encodings |
| BC/friction law as learnable inductive bias | Training instability if distributions too different |

**Verdict:** Worth doing when data is plentiful and you want a single production model.
Especially powerful for *super-resolution across conditions* without re-training.

---

## 2. Operator to learn

$$\mathcal{G}_\theta : (\mathbf{a}(x),\, \mathbf{p}_s,\, \lambda_{fric},\, \lambda_{bc}) \;\mapsto\;
  (v_l, s_l, v_r, s_r)(x, t)$$

where:
- $\mathbf{a}(x)$: spatial material profiles (6 channels)
- $\mathbf{p}_s$: scalar physics parameters (12 channels, law-conditional)
- $\lambda_{fric} \in \{0,1\}$: friction law flag (SW=0, RS=1) for FiLM modulation
- $\lambda_{bc} \in \{0,1,2\}$: outer BC mode (free=0, absorb=1, PML=2) for FiLM modulation

---

## 3. Full input variable space (C_in = 22 channels)

All channels have shape (NX, NT); spatial profiles are broadcast over time,
scalar params are broadcast over both dimensions.

### Spatial profiles — 6 channels (broadcast over t)
| Ch | Variable | Units | Notes |
|----|----------|-------|-------|
| 0 | `cs(x)` | km/s | from profile file or scalar |
| 1 | `rho(x)` | g/cm³ | spatially variable density |
| 2 | `mu(x) = rho*cs²` | GPa | shear modulus |
| 3 | `Qs_inv(x) = 1/Q_S(x)` | — | zero for elastic response |
| 4 | `d_l(x)` | 1/s | PML damping left domain; zero if no PML |
| 5 | `d_r(x)` | 1/s | PML damping right domain; zero if no PML |

### Scalar friction parameters — 11 channels (broadcast over x,t)
SW-only params are zero for RS inputs; RS-only params are zero for SW inputs.

| Ch | Variable | Units | Active for |
|----|----------|-------|-----------|
| 6  | `Tau_0` | MPa | both |
| 7  | `sigma_n` | MPa | both |
| 8  | `alp_s` | — | SW |
| 9  | `alp_d` | — | SW |
| 10 | `D_c` | m | SW |
| 11 | `f0` | — | RS |
| 12 | `a` | — | RS |
| 13 | `b` | — | RS |
| 14 | `V0` | m/s | RS |
| 15 | `L0` | m | RS |
| 16 | `psi_init` | — | RS |

### Scalar material/anelastic parameters — 3 channels
| Ch | Variable | Notes |
|----|----------|-------|
| 17 | `c` | Q_S = c·V_S; zero for elastic |
| 18 | `weight_exp` | 0.0 = const-Q, 0.6 = power-law Q |
| 19 | `r0_l` | outer left BC: {-1, 0, 1, 2=PML} |

### Coordinate channels — 2
| Ch | Variable | Range |
|----|----------|-------|
| 20 | `x` normalized | [0, 1] |
| 21 | `t` normalized | [0, 1] |

### Discrete conditioning (FiLM layers, not channels)
- `fric_law_idx` ∈ {0=SW, 1=RS}  → learned embedding dim=16 → FiLM γ,β per layer
- `bc_mode_idx`  ∈ {0=free, 1=absorb, 2=PML}  → same

---

## 4. Output (C_out = 4)
| Ch | Variable | Shape |
|----|----------|-------|
| 0 | `v_l(x,t)` | (NX, NT) |
| 1 | `s_l(x,t)` | (NX, NT) |
| 2 | `v_r(x,t)` | (NX, NT) |
| 3 | `s_r(x,t)` | (NX, NT) |

Post-processed: `slip(t) = integral of (v_l[nx-1] - v_r[0]) dt`,
`sliprate(t) = v_l[nx-1] - v_r[0]`,  `traction(t) = Tau_0 + s_l[nx-1]`

---

## 5. Architecture: FNO-2D with FiLM conditioning

```
Input a(x,t) ∈ R^{NX × NT × 22}   ← all channels concat
        │
   Lifting Conv2d(22, W)
        │
   ┌────┴─────────────────────────────────────────────┐
   │  FiLM-FNO Block × L_layers                       │
   │  ┌──────────────────────────────────────────┐    │
   │  │  SpectralConv2d (W→W, modes_x, modes_t)  │    │
   │  │  + Conv2d bypass (1×1)                   │    │
   │  │  + FiLM: γ(e_fric,e_bc)·x + β(·)        │    │
   │  │  → GeLU                                  │    │
   │  └──────────────────────────────────────────┘    │
   └──────────────────────────────────────────────────┘
        │
   Projection MLP: W → W/2 → GeLU → 4
        │
Output u(x,t) ∈ R^{NX × NT × 4}
```

**Hyperparameters (starting point):**
| Param | Value |
|---|---|
| width W | 128 |
| L_layers | 6 |
| modes_x | 24 |
| modes_t | 32 |
| FiLM embedding dim | 16 |
| Activation | GeLU |
| Optimizer | AdamW, lr=5e-4 |
| LR schedule | Cosine with 10-epoch warmup |
| Batch size | 8–16 |

---

## 6. PINO loss

$$\mathcal{L} = \lambda_{data}\,\mathcal{L}_{data} + \lambda_{pde}\,\mathcal{L}_{pde} + \lambda_{bc}\,\mathcal{L}_{bc} + \lambda_{ic}\,\mathcal{L}_{ic}$$

### 6.1 Data loss
$$\mathcal{L}_{data} = \frac{1}{4}\sum_{f \in \{v_l,s_l,v_r,s_r\}} \frac{\|\hat{f} - f\|_2}{\|f\|_2}$$

### 6.2 PDE residual (Fourier-spectral derivatives)
**Elastic:**
$$\mathcal{L}_{pde} = \|\partial_t \hat{v} - \tfrac{1}{\rho}\partial_x \hat{s}\|^2 + \|\partial_t \hat{s} - \mu\partial_x \hat{v}\|^2$$

**Anelastic** (when `response=anelastic`, weighted by `is_anelastic` flag):
- Replace μ → μ_unrelax and add attenuation memory residual (see physics_loss.py)

### 6.3 Fault BC
$$\mathcal{L}_{bc,fault} = \|\hat{\tau}_{pred} - \tau_{friction}(D, \mathbf{p})\|^2$$
where $\hat{\tau}_{pred} = \tau_0 + \hat{s}_l(x_f, t)$ and $\tau_{friction}$ is computed from
SW or RS law.

### 6.4 Initial condition
$$\mathcal{L}_{ic} = \|\hat{v}_l(\cdot, 0)\|^2 + \|\hat{s}_l(\cdot, 0)\|^2 + \ldots$$

### 6.5 Loss weights (schedule)
- Epoch 0–20: λ_data=1.0, λ_pde=0.01, λ_bc=0.1, λ_ic=0.1  (data-dominated)
- Epoch 20+: λ_pde→0.1  (physics kicks in for regularisation)

---

## 7. Dataset

**Latin Hypercube Sampling** over the full joint space:

| Parameter | SW range | RS range |
|---|---|---|
| `Tau_0` | [78, 88] MPa | [78, 88] MPa |
| `sigma_n` | [100, 140] MPa | [100, 140] MPa |
| `alp_s` | [0.62, 0.74] | — |
| `alp_d` | [0.45, 0.55] | — |
| `D_c` | [0.2, 0.8] m | — |
| `f0` | — | [0.5, 0.7] |
| `a` | — | [0.006, 0.012] |
| `b` | — | [0.014, 0.022] |
| `L0` | — | [0.01, 0.05] m |
| `cs` | [3.0, 4.0] km/s | same |
| `c` (Q) | [10, 30] or ∞ | same |
| `weight_exp` | {0.0, 0.6} | same |

**Balanced sampling across conditions:**
- 30% SW+free, 30% SW+absorb, 10% SW+PML
- 20% RS+free, 20% RS+absorb, 10% RS+PML (wait — that's 120%...)
- Actually: split total N equally: N/(6 conditions) per condition type

**Sizes:**
```
N_train = 2400  (400 per condition)
N_val   =  300  (50  per condition)
N_test  =  300  (50  per condition)
```

**Storage resolutions:** Stored at 3 resolutions for multi-scale training:
- LR: (NX=64,  NT=64)   — for fast training / ablation
- MR: (NX=128, NT=128)  — baseline training resolution
- HR: (NX=256, NT=256)  — super-resolution target

---

## 8. Super-resolution

FNO naturally supports resolution changes: train at MR (128×128), infer at HR (256×256).
The spectral modes (modes_x, modes_t) stay fixed; only the grid changes.

### Multi-scale training strategy
During training, randomly sub-sample each batch:
- 50% of batches at LR (64×64)
- 50% of batches at MR (128×128)

This teaches resolution invariance. At test time, evaluate at HR (256×256) — 4× super-resolution.

---

## 9. FNO vs Graph NO recommendation

**For this problem: FNO is clearly superior.**

| Criterion | FNO | Graph NO |
|---|---|---|
| Grid type | Regular 1D uniform (**this case**) | Irregular / adaptive |
| Computational cost | O(NX NT log NX NT) via FFT | O(N_nodes × N_neighbors) |
| Long-range interactions | Global via full Fourier basis | Limited by graph connectivity radius |
| Physics loss | Exact Fourier derivatives (PDE in k-space) | Finite differences on graph |
| Seismic wave propagation | Excellent: waves are Fourier modes | Suboptimal |
| Adaptive mesh | No | Yes |

Graph NO would only be preferred if: (a) the mesh is spatially adaptive, or (b) the geometry is
complex 3D unstructured. For 1D regular grid seismic rupture, FNO dominates.

---

## 10. Files

```
waveqlab1d_FNO_PINO/
├── plan_A_unified.md          ← this file
├── plan_B_separate.md
├── configs/
│   └── unified.yaml           ← hyperparameters + training config
├── data_gen/
│   ├── param_space.py         ← LHS bounds for all conditions
│   ├── generate_dataset.py    ← batch runs waveqlab1d, saves multi-res npz
│   └── dataset.py             ← PyTorch Dataset (all conditions)
├── model/
│   ├── fno.py                 ← FNO2D + FiLM + SpectralConv2d
│   └── physics_loss.py        ← data + PDE + fault BC + IC losses
├── train_unified.py           ← Plan A training loop
└── evaluate.py                ← evaluation, super-res comparison, plots
```
