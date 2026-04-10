# Plan B — Separate PINO-FNOs (specialized per friction law × BC type)

**Simulator:** `waveqlab1d/rupture_1d.py`  
**Goal:** Train 6 independent FNO models, one per (friction law × outer BC) combination.
Each model is smaller, trains on a narrower distribution, and is easier to validate.

---

## 1. Why separate models?

| Argument for separate | Argument against |
|---|---|
| SW and RS dynamics are qualitatively different (state law vs instantaneous) | 6× data generation and training cost |
| BCs qualitatively change solution character (reflections, decay) | No cross-condition generalization |
| Each model learns a sharper mapping → lower prediction error | More complex deployment |
| Easier hyperparameter tuning per model | |
| Failure of one model does not affect others | |

**Verdict:** Best for production-quality prediction when conditions are known at inference time.
Lower validation error per condition than Plan A, especially for RS (complex state evolution).

---

## 2. The 6 models

| Model | fric_law | outer BC | r0_l | r1_r |
|---|---|---|---|---|
| `sw_free` | SW | Free surface | 1 | 1 |
| `sw_absorbing` | SW | SAT absorbing | 0 | 0 |
| `sw_pml` | SW | PML volume | pml=True | pml=True |
| `rs_free` | RS | Free surface | 1 | 1 |
| `rs_absorbing` | RS | SAT absorbing | 0 | 0 |
| `rs_pml` | RS | PML volume | pml=True | pml=True |

Inner (fault-side) BCs always: r1_l=1, r0_r=1 (free-surface coupling through fault).

---

## 3. Input channels per model

Since friction law and BC type are fixed, no FiLM conditioning needed.
Unused parameters (SW params in RS model, etc.) are simply absent.

### All 6 models — shared spatial channels (6 channels, broadcast over t)
| Ch | Variable | Units |
|----|----------|-------|
| 0 | `cs(x)` | km/s |
| 1 | `rho(x)` | g/cm³ |
| 2 | `mu(x)` | GPa |
| 3 | `Qs_inv(x)` | — (zero for elastic) |
| 4 | `d_l(x)` | 1/s (zero if no PML) |
| 5 | `d_r(x)` | 1/s (zero if no PML) |

### SW models — additional scalar channels (7 params + 2 anelastic + 2 coords = 11)
| Ch | Variable | Notes |
|----|----------|-------|
| 6  | `Tau_0` | MPa |
| 7  | `sigma_n` | MPa |
| 8  | `alp_s` | static friction |
| 9  | `alp_d` | dynamic friction |
| 10 | `D_c` | m |
| 11 | `c` | Q factor scalar; 0=elastic |
| 12 | `weight_exp` | attenuation exponent |
| 13 | `x` coord | [0,1] |
| 14 | `t` coord | [0,1] |
| **Total** | **C_in = 15** | |

### RS models — additional scalar channels (9 params + 2 anelastic + 2 coords = 13)
| Ch | Variable | Notes |
|----|----------|-------|
| 6  | `Tau_0` | MPa |
| 7  | `sigma_n` | MPa |
| 8  | `f0` | reference friction |
| 9  | `a` | direct effect |
| 10 | `b` | evolution effect |
| 11 | `V0` | m/s |
| 12 | `L0` | m |
| 13 | `psi_init` | initial state |
| 14 | `c` | Q scalar; 0=elastic |
| 15 | `weight_exp` | attenuation exponent |
| 16 | `x` coord | [0,1] |
| 17 | `t` coord | [0,1] |
| **Total** | **C_in = 18** | |

### PML models — extra `pml_alpha` channel
- Add 1 channel: `pml_alpha` (broadcast scalar) → C_in = 16 (SW_PML) or 19 (RS_PML)

---

## 4. Architecture: FNO-2D (no FiLM, smaller than Plan A)

```
Input a(x,t) ∈ R^{NX × NT × C_in}
        │
   Lifting Conv2d(C_in, W)
        │
   ┌────┴──────────────────────────────────────┐
   │  FNO Block × L_layers                     │
   │  SpectralConv2d + bypass Conv2d + GeLU    │
   └───────────────────────────────────────────┘
        │
   Projection: W → W/2 → GeLU → 4
        │
Output u(x,t) ∈ R^{NX × NT × 4}
```

**Hyperparameters (smaller than Plan A since distribution is narrower):**
| Param | SW models | RS models |
|---|---|---|
| width W | 64 | 96 |
| L_layers | 4 | 5 |
| modes_x | 16 | 20 |
| modes_t | 24 | 28 |
| Batch size | 16 | 12 |
| LR | 1e-3 | 5e-4 |

RS models are wider/deeper because state evolution creates more complex spatiotemporal patterns.

---

## 5. PINO loss (per model, law-specific)

### SW models
$$\mathcal{L} = \mathcal{L}_{data} + \lambda_{pde}\mathcal{L}_{pde} + \lambda_{fault}\mathcal{L}_{fault,SW} + \lambda_{ic}\mathcal{L}_{ic}$$

Fault residual (slip-weakening):
$$\tau_{pred}(t) = \tau_0 + \hat{s}_l(x_f, t)$$
$$\tau_{SW}(t) = \sigma_n[\alpha_s - (\alpha_s-\alpha_d)\min(D(t)/D_c, 1)]$$
$$\mathcal{L}_{fault,SW} = \|\tau_{pred} - \tau_{SW}\|^2$$

### RS models
Rate-and-state fault residual is harder to enforce since it requires integrating the ODE for ψ.
Use a softer penalty via the Dieterich aging law check at each time step:
$$V = |\hat{v}_l(x_f,t) - \hat{v}_r(x_f,t)|, \quad \tau_{RS} = \sigma_n [f_0 + a\ln(V/V_0) + \psi]$$

Include `psi_target` as additional output channel for RS models, with physics constraint:
$$\partial_t \hat{\psi} = - (\hat{V}/L_0)(\hat{\psi} - \ln(V_0/\hat{V}))$$

### PML models
Add outer BC penalty at domain edges based on PML damping equation.

### Outer BC loss (condition-specific)
- **free (r=1):** `s_l[0, t] = 0` → L² penalty at left boundary
- **absorbing (r=0):** `v_l[0, t] + s_l[0, t] / (ρ cs) = 0` → impedance condition
- **PML:** PML damping equation as soft constraint

---

## 6. Dataset (per model)

Each model needs its own dataset with fixed BC and fric_law.

**Sizes per model:**
```
N_train = 800   samples
N_val   =  100  samples
N_test  =  100  samples
```

**Total across all 6 models:** 6000 train, 600 val, 600 test samples.

**Generation estimate:** 6000 × 0.5 s avg ≈ 50 min on 56-thread node.

**Parameter ranges per model:**

SW models — sample over:
| Parameter | Range |
|---|---|
| `Tau_0` | [78, 88] MPa |
| `sigma_n` | [100, 140] MPa |
| `alp_s` | [0.62, 0.74] |
| `alp_d` | [0.45, 0.55] (constrained < alp_s) |
| `D_c` | [0.2, 0.8] m |
| `cs` | [2.8, 4.0] km/s |
| `rho` | [2.4, 3.0] g/cm³ |
| `c` (Q) | [0=elastic, 10, 20, 50] (categorical + continuous) |
| `weight_exp` | {0.0, 0.6} (categorical) |

RS models — sample over:
| Parameter | Range |
|---|---|
| `Tau_0` | [78, 88] MPa |
| `sigma_n` | [100, 140] MPa |
| `f0` | [0.5, 0.7] |
| `a` | [0.005, 0.015] |
| `b` | [0.015, 0.025] (constrained > a) |
| `V0` | [1e-7, 1e-5] m/s |
| `L0` | [0.01, 0.05] m |
| `psi_init` | [0.35, 0.50] |
| `cs`, `rho`, `c`, `weight_exp` | same as SW |

---

## 7. Super-resolution

Same strategy as Plan A:
- Train at MR (128×128)
- Evaluate at HR (256×256): just feed finer grid input

Per-model SR is more accurate than Plan A since distribution is narrower.

---

## 8. Recommended ensemble strategy

For the best accuracy at deployment:
1. Select model by `(fric_law, bc_type)` → one of 6 Plan B models
2. Fall back to Plan A if condition falls outside the 6 standard cases (e.g. mixed BCs)
3. At super-resolution: use the relevant Plan B model since its spectra are sharper

---

## 9. Comparison: Plan A vs Plan B

| Metric | Plan A (Unified) | Plan B (Separate) |
|---|---|---|
| Models to train | 1 | 6 |
| Total parameters | ~40M (W=128, L=6) | 6 × ~5M = 30M total |
| Training data | 3000 mixed | 4800 per-condition |
| Generalization to mixed BCs | Yes | No |
| Per-condition accuracy | Moderate | High |
| Cross-law transfer | Yes | No |
| Training complexity | High | Low |
| Recommended for | Research / exploration | Production / accuracy |

---

## 10. Files

```
waveqlab1d_FNO_PINO/
├── plan_A_unified.md
├── plan_B_separate.md         ← this file
├── configs/
│   ├── sw_free.yaml
│   ├── sw_absorbing.yaml
│   ├── sw_pml.yaml
│   ├── rs_free.yaml
│   ├── rs_absorbing.yaml
│   └── rs_pml.yaml
├── data_gen/
│   ├── param_space.py         ← bounds for each of the 6 models
│   ├── generate_dataset.py    ← one CLI flag selects which model's data to gen
│   └── dataset.py             ← dataset class (handles both plans)
├── model/
│   ├── fno.py                 ← shared backbone, no FiLM for Plan B
│   └── physics_loss.py        ← law-conditional loss functions
├── train_separate.py          ← Plan B training loop (selects model by config key)
└── evaluate.py                ← compare Plan A vs Plan B + super-res
```
