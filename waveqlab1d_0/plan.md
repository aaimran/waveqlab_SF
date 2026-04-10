# Plan: Build `waveqlab1d` — 1D Anelastic Rupture Simulator

**Goal:** A 1D Python/Numba code matching the full physics of `WaveQLab3D (aQ)` in one spatial dimension, built on top of `1d_numba`. The state vector expands from `(v, s)` to `(v, s, η₁…η₄)`.

---

## Comparison: `1d_numba` vs `WaveQLab3D (aQ)`

| Feature | `1d_numba` | `WaveQLab3D (aQ)` |
|---|---|---|
| **Language / runtime** | Python + Numba JIT | Fortran + MPI |
| **Dimensions** | 1D (two half-domains + fault) | 3D (two blocks + fault interface) |
| **Parallelism** | Numba thread pool (`prange`) | 3D MPI Cartesian domain decomposition |
| **Build system** | None (pure Python) | CMake |
| **Wave physics** | Elastic only (`v`, `s`) | Elastic + Anelastic (GSLS, N=4 memory variables per stress component) |
| **SBP operators** | Orders 2, 4, 6 — traditional only | Orders 2–9 — traditional, upwind, upwind-DRP |
| **Time integrator** | RK4, 4-stage (classic) | Low-storage RK, up to Kennedy–Carpenter (5,4) |
| **Absorbing boundaries** | SAT with r ∈ {-1,0,1} | SAT + PML (6 faces, anelastic-compatible) |
| **Fault coupling** | 1D point, Riemann characteristics | 3D non-planar interface, locally-rotated normals |
| **Friction: Slip-Weakening** | Yes | Yes |
| **Friction: Rate-and-State** | Yes (aging law, Regula Falsi) | Yes (aging + strongly rate-weakening) |
| **Friction: linear** | No | Yes |
| **Off-fault plasticity** | No | Yes (Drucker–Prager, μ, β, η) |
| **Material heterogeneity** | Homogeneous (ρ, μ scalars) | Spatially variable λ(x,y,z), μ(x,y,z), ρ(x,y,z) |
| **Geometry** | Uniform rectilinear | Curvilinear with topography |
| **Anelastic memory variables** | None | η₄–η₉: 6 stress × 4 mechanisms; rates η̇_* |
| **Q model** | None | GSLS: Q_S, Q_P pointwise, log-spaced τ_l, pre-tabulated w_l |
| **Moment tensor source** | No | Yes (multiple, full M_ij tensor, optional mollification) |
| **Output: seismograms** | No | Yes (arbitrary stations by coord or index) |
| **Output: domain snapshots** | .npz at stride iplot | Binary plane/slice snapshots |
| **Output: fault fields** | slip, slip rate, traction arrays | Fault-plane time-series files |
| **MMS verification** | No | Yes (eval_mms) |
| **Input format** | `key = value` flat file | Fortran namelists (&problem_list, &block_list, ...) |
| **Run command** | `python3 rupture_1d.py file.in` | MPI binary from CMake build |

---

## Implementation Steps

### Step 1 — Copy `1d_numba` as the starting skeleton
- [ ] Copy all files from `1d_numba/` into `waveqlab1d/`
- [ ] Verify the copy runs cleanly before touching any code

### Step 2 — Spatially variable material properties
- [ ] Replace scalar `rho`, `mu` with `(nx,)` arrays
- [ ] Update `_bc_left`, `_bc_right`, `_interface_kernel` to accept boundary-point scalars
- [ ] Change material init in `rupture_1d.py` to `np.full(nx, value)` with hook for file input
- [ ] Add input keys: `rho_file`, `cs_file` (optional)

### Step 3 — Anelastic material initialization (GSLS parameters)
- [ ] Create `src/anelastic.py`
- [ ] Implement `init_anelastic(nx, mu, rho, cs_profile, c, weight_exp, fref)` returning:
  - `Qs_inv (nx,)` — Q_S⁻¹ = 1/(c√(μ/ρ))
  - `tau (4,)` — log-spaced relaxation times [0.08, 15] Hz
  - `weight (4,)` — pre-tabulated weights (two sets for weight_exp ≈ 0 or ≈ 0.6)
  - `mu_unrelax (nx,)` — unrelaxed moduli corrected for velocity dispersion
- [ ] Add input keys: `response` (elastic/anelastic), `c`, `weight_exp`, `fref`

### Step 4 — Expand state arrays for memory variables
- [ ] Allocate `eta_l (nx, 4)` for each domain (left and right), initialized to zero
- [ ] Allocate `Deta_l (nx, 4)` for the rates
- [ ] Pass into RK4 step and scale/update alongside `v`, `s`

### Step 5 — Anelastic RHS kernel
- [ ] Add `_anelastic_rhs(hs, Deta, eta, vx, mu, Qs_inv, tau, weight, nx)` to `kernels.py`
- [ ] For each point i:
  - `ṡ_i -= Σ_l η_{i,l}`
  - `η̇_{i,l} = (1/τ_l) [w_l μ_i Q_{S,i}⁻¹ (∂_x v)_i - η_{i,l}]`
- [ ] Annotate `@njit(parallel=True, cache=True)` with `prange` over i

### Step 6 — PML absorbing boundaries
- [ ] Create `src/pml.py`
- [ ] `init_pml(nx, npml, rho, mu)` — allocate `Q_pml (npml,)` per domain end
- [ ] `pml_rhs(...)` — quadratic/cosine damping profile d(x); apply to auxiliary vars
- [ ] Replace `r=0` SAT at outer boundaries with PML when `pml = True`
- [ ] Add input keys: `pml` (true/false), `npml` (int, default 20), `pml_alpha`

### Step 7 — Wire everything into `rk4_step`
Per RK stage:
1. `sbp_dx` → compute ∂_x v and ∂_x s
2. Elastic rates: `hv += (1/ρ) ∂_x s`, `hs += μ_unrelax ∂_x v`
3. `_anelastic_rhs` → subtract Ση_l from `hs`; fill `Deta`
4. `_bc_left` / `_bc_right` (SAT or PML)
5. `_interface_kernel` (friction, unchanged)
6. Scale and accumulate RK stages for `v`, `s`, and all `eta_l`

### Step 8 — Update runner and input file
- [ ] Add `response`, `c`, `weight_exp`, `fref`, `pml`, `npml` to `DEFAULTS`
- [ ] If `response == 'anelastic'`: call `init_anelastic`, override `mu` with `mu_unrelax`, allocate `eta`/`Deta`
- [ ] Optional: save `eta_l` snapshots if `iplot_eta` is set
- [ ] Save `Qs_inv`, `tau`, `weight` in output `.npz` parameter record

### Step 9 — Verification
- [ ] **Elastic regression:** `response = elastic` output must match `1d_numba` reference exactly
- [ ] **Anelastic Q test:** plane wave in homogeneous medium — compare amplitude decay rate
  `α = ω/(2 Q c_s)` and phase velocity dispersion with GSLS analytical dispersion relation
- [ ] **Energy check:** without friction, verify total energy (kinetic + elastic + dissipated) is non-increasing
- [ ] **Thread scaling:** rerun `benchmark_scaling.py` to confirm anelastic `prange` loop scales

### Step 10 — Example input file
- [ ] Create `input/rupture_1d_aQ.in` exercising anelastic attenuation
- [ ] All new keys documented in header comments (same style as `rupture_1d_SW.in`)

---

## Progress Tracker

| Step | Status |
|------|--------|
| 1 — Copy skeleton | ⬜ |
| 2 — Variable materials | ⬜ |
| 3 — GSLS init | ⬜ |
| 4 — Memory variable state | ⬜ |
| 5 — Anelastic RHS | ⬜ |
| 6 — PML | ⬜ |
| 7 — Wire rk4_step | ⬜ |
| 8 — Runner + input | ⬜ |
| 9 — Verification | ⬜ |
| 10 — Example input | ⬜ |
