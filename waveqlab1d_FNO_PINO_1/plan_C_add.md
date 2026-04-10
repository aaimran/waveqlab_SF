# Plan C Addendum: Theory, Resolution Invariance, and Architecture Design

---

## Part A: FNO and PINO Architecture (Space-Time 2D)

### 1. Fourier Neural Operator (FNO)

The implementation uses a **2D FNO** operating on the joint **space-time** field
represented as a tensor of shape $(B, C, N_x, N_t)$. The spectral convolution
applies `rfft2` / `irfft2` across *both* spatial and temporal axes simultaneously.

Architecture workflow (`SpectralConv2d` → `FNOBlock2d`):

1. **Input:** Space-time field $(B, C_\text{in}, N_x, N_t)$ — velocity, stress,
   material parameters, and normalised coordinates, all as 2D channels.
2. **Lifting:** Pointwise linear map $C_\text{in} \to d$ (width).
3. **FNO Block** (`FNOBlock2d`, repeated $L$ times):
   - **`rfft2`:** 2D real FFT → $(B, d, N_x, N_t/2+1)$ complex spectrum.
   - **Spectral filter:** Retain $k_x \leq m_x$ and $k_t \leq m_t$ modes; apply
     learnable complex matrices $R_1, R_2 \in \mathbb{C}^{d \times d \times m_x \times m_t}$
     to the positive and negative $k_x$ quadrants respectively. Modes are **clamped**
     to $m_x = \min(\text{modes\_x},\, N_x/2)$ and $m_t = \min(\text{modes\_t},\, N_t/2+1)$
     so the same weights operate correctly at any grid size.
   - **`irfft2`:** Inverse FFT back to $(B, d, N_x, N_t)$.
   - **Bypass:** Pointwise $1{\times}1$ convolution of the block input (ResNet skip).
   - **FiLM** (Plan A `UnifiedFNO2d` only): per-layer $\gamma, \beta$ from
     condition embedding applied after bypass sum.
   - **Activation:** GELU.
4. **Projection:** Two linear layers $d \to d \to C_\text{out}=4$ with GELU.

### 2. Physics-Informed Neural Operator (PINO) — Six Loss Terms

All physics is implemented in **PyTorch**. Six loss terms make the operator
physics-informed:

| Term | Role | Default $\lambda$ |
|------|------|------------------|
| $\mathcal{L}_\text{data}$ | Relative $L^2$ vs. SBP-SAT ground truth | 1.0 |
| $\mathcal{L}_\text{pde}$ | Elastic wave equation residuals via spectral derivatives | 0.05 → 0.10 |
| $\mathcal{L}_\text{fault}$ | Fault traction continuity + SW or RS friction law residual at $x_f$ | 0.10 → 0.20 |
| $\mathcal{L}_\text{bc}$ | Outer boundary: free / SAT absorbing / PML proxy | 0.05–0.15 |
| $\mathcal{L}_\text{ic}$ | All fields zero at $t = 0$ | 0.05 |
| $\mathcal{L}_\text{stab}$ | SBP-SAT energy stability: penalise $\partial_t E > 0$ where $E = \tfrac{1}{2}\int(\rho v^2 + \sigma^2/\mu)\,dx$ | 0.02 → 0.05 |

PDE derivatives are computed **spectrally** — not via `torch.autograd`. The
`rfft` of each field is multiplied by $ik$ along the appropriate axis, then
`irfft` maps back to the spatial/temporal domain. This is exact for band-limited
fields and incurs no gradient overhead through the network weights.

### 3. FNO vs. PINO Comparison

| Feature | FNO alone | PINO (this implementation) |
|---------|-----------|---------------------------|
| Input shape | $(B, C, N_x, N_t)$ — space-time grid | Same |
| Derivatives for loss | Not applicable | Spectral ($\times ik$, `rfft`/`irfft`) — not autograd |
| Fault handling | No explicit enforcement | $\mathcal{L}_\text{fault}$: traction continuity + friction residual |
| Long-time stability | May drift | $\mathcal{L}_\text{stab}$: energy growth penalised |
| Framework | PyTorch | PyTorch |

---

## Part B: Resolution Invariance

### 1. The Kernel Integral Perspective

The mathematical foundation of an FNO is the **kernel integral operator**:

$$\mathcal{G}(u)(x) = \int_{\Omega} \kappa(x, y)\, u(y)\, dy$$

Because an integral is a continuous operator it is independent of the number of
quadrature points used to sample $u(y)$. Whether you use $128$ or $256$ grid
points, the underlying integral — and the physics it encodes — remains the same.

### 2. Resolution-Invariance via Fixed Fourier Modes

The FNO achieves resolution invariance through **clamped spectral modes**, not
spectral zero-padding at inference time:

- Weight matrices $R_1, R_2$ always act on $m_x \times m_t$ modes regardless of
  grid size (clamped at both training and inference).
- At inference on a larger grid $(N_x', N_t')$: `rfft2` produces a larger spectrum;
  the same clamped slice is extracted; `irfft2(s=(N_x', N_t'))` reconstructs on
  the finer grid.
- Intermediate points are **spectrally interpolated** by the inverse FFT — no
  explicit zero-padding step is needed in user code.

Super-resolution workflow:

| Step | Detail |
|------|--------|
| Training | LR SBP-SAT data $128 \times 128$; HR data $256 \times 256$ (30% of batches) |
| SR inference | Pass $256 \times 256$ input to a model trained on $128 \times 128$ |
| Mechanism | `irfft2(s=(256,256))` reconstructs fine grid from fixed $m_x \times m_t$ modes |
| Validation | HR SBP-SAT labels ($256 \times 256$) used as reference for error measurement |

### 3. PINO Spectral Derivatives (Not Off-Grid Autograd)

PDE residuals are evaluated **on the same fixed grid** as the prediction using
spectral finite differences:

$$\partial_x u \approx \mathcal{F}_x^{-1}\!\left[i k_x\,\hat{u}(k_x, \cdot)\right], \quad
\partial_t u \approx \mathcal{F}_t^{-1}\!\left[i \omega\,\hat{u}(\cdot, \omega)\right]$$

There is no autograd-based off-grid query. The PINO physics loss enforces
consistency at the training grid points; resolution generalisation comes from the
FNO's fixed-mode spectral convolution, not from continuous AD.

### 4. Resolution Invariance Summary

| Feature | Standard FDTD / CNN | FNO / PINO (this work) |
|---------|---------------------|------------------------|
| Input dependency | Fixed grid size | Grid-agnostic via clamped spectral modes |
| SR mechanism | Bilinear / cubic interpolation | `irfft2(s=HR)` using fixed learned modes |
| PDE derivatives | None | Spectral: `rfft` $\times ik$ then `irfft` |
| Generalisation | Retrain for each new resolution | Train on LR+HR mixture; infer at HR |

---

## Part C: Multi-Scale Training Strategy

### 1. Why Train on Multiple Resolutions

Training on LR data alone means the model never sees high-frequency features
(sharp stress drop at a crack tip, high-$f$ seismic content). A resolution mixture
helps for three reasons:

- **Energy spectrum coverage:** Seismic waves follow a power-law distribution in
  frequency. Multi-resolution training ensures $R_1, R_2$ learn both global
  propagation and local fine-scale physics.
- **Aliasing protection:** The model learns to discriminate true high-frequency
  physics from numerical aliasing present in coarse simulations.
- **Generalisation:** A model trained on a mixture generalises better to
  resolutions not seen during training.

### 2. Implemented Strategy: LR/HR Batch Swap

The current implementation uses **two resolutions** with a **whole-batch swap**
strategy. There is no multi-stage curriculum — training is single-phase with a
weight ramp at epoch 20:

| Parameter | Value |
|-----------|-------|
| LR grid | $128 \times 128$ (space × time) |
| HR grid | $256 \times 256$ |
| `multiscale_hr_frac` | 0.3 — 30% of training batches drawn from HR loader |
| Swap granularity | Whole batch (all LR or all HR; not mixed within a batch) |
| HR loader | Separate `DataLoader(RuptureDataset(res='hr'))`, wrapped with `itertools.cycle` |
| Physics loss | Computed on whichever resolution the batch uses — grid-size agnostic |

Config keys (`train_separate.py`):
```yaml
training:
  multiscale: true
  multiscale_hr_frac: 0.3
```

### 3. Physics Weight Ramp (Single Step at Epoch 20)

| Epoch range | $\lambda_\text{pde}$ | $\lambda_\text{fault}$ | $\lambda_\text{stab}$ |
|-------------|---------------------|----------------------|----------------------|
| 0–19 | 0.05 | 0.10 | 0.02 |
| 20 + | 0.10 | 0.20 | 0.05 |

This prioritises data fit early (when the model first learns wave propagation
patterns) and ramps physics enforcement once the solution is approximately correct.

### 4. High-Resolution Training Bottlenecks

- **Memory:** FFT and physics loss scale with $N_x \times N_t$. At $256 \times 256$
  a batch uses $4\times$ the memory of the same batch at $128 \times 128$. Reduce
  `batch_size` or `multiscale_hr_frac` if out-of-memory.
- **Data cost:** HR fields are produced by down-sampling a full-NX SBP-SAT
  simulation (run once at high native resolution); HR data cost per sample is
  thus the same as LR.

### 5. Super-Resolution Evaluation Goal

Train on LR ($128 \times 128$) with 30% HR ($256 \times 256$) mixing; evaluate
at $256 \times 256$ against HR SBP-SAT ground truth. If the relative $L^2$ error
at HR is comparable to the LR training error, the model has learned
resolution-invariant physics — the "gold standard" for PINO generalisation.

---

## Part D: Unified vs. Separate Architecture Design

### 1. FiLM Conditioning in the Unified Model (Plan A)

`UnifiedFNO2d` uses **FiLM (Feature-wise Linear Modulation)** conditioned on the
discrete pair $(fric\_law,\, bc\_mode)$ — not on elastic vs. anelastic physics
(all simulations are GSLS $N=4$ anelastic):

- `fric_law ∈ {SW, RS}` → integer index 0 or 1
- `bc_mode ∈ {free, absorbing, pml}` → integer index 0, 1, or 2
- Each index is passed through a learned embedding; the two embeddings are
  concatenated and projected to $2d$ to produce per-layer $(\gamma, \beta)$ that
  scale and shift feature maps inside every `FNOBlock2d`.

### 2. Input Channels: Fixed Spatial + Law-Specific Scalars

All 6 spatial channels are **always present** in every sample, broadcast to
$(N_x, N_t)$. Scalar counts vary only because SW and RS have different friction
parameters — not because channels are conditionally dropped:

| Channel group | Contents | Count |
|---------------|----------|-------|
| Spatial profiles | `cs, rho, mu, Qs_inv, d_l, d_r` broadcast over $t$ | 6 (always) |
| Scalars — SW | `Tau_0, sigma_n, alp_s, alp_d, D_c, c, weight_exp` | 7 |
| Scalars — RS | `Tau_0, sigma_n, f0, a, b, V0, L0, psi_init, c, weight_exp` | 10 |
| Coordinates | `x_norm, t_norm` | 2 (always) |

This gives $C_\text{in} = 15$ or $16$ for SW (free/absorbing vs. pml) and
$18$ or $19$ for RS.

### 3. Modular Physics Loss

| Component | Strategy in this implementation |
|-----------|--------------------------------|
| Core FNO layers | Unified — `FNOBlock2d` shared across all conditions |
| PML loss | Modular — `outer_bc_loss` dispatches to `outer_bc_loss_free` / `outer_bc_loss_absorbing` by `bc_mode` |
| Friction law residual | Modular — `PINOLoss` calls `fault_loss_sw` or `fault_loss_rs` by `fric_law` |
| Per-condition model (Plan B) | 6 separate `SeparateFNO2d` — one per (fric × bc) pair |
| Unified model (Plan A) | 1 `UnifiedFNO2d` with FiLM for all 6 conditions |

### 4. Gradient Conflict and Capacity Tradeoff

- **Gradient conflict:** PML absorption gradients can oppose bulk-propagation
  gradients — mitigated by keeping `w_outer` (0.05–0.15) significantly lower
  than `w_data` (1.0).
- **Capacity tradeoff:** Plan A width=128 / 6 layers must serve all 6 conditions;
  Plan B width=64–96 / 4–5 layers per condition, typically giving better
  per-condition accuracy at lower parameter count per model.
