# Plan: Build 2D Numba-JIT Rupture Solver (`2d_numba`)

**Source:** `2d_serial/`  
**Template:** `1d_numba/`  
**Goal:** Replace the pure-Python double loops in `2d_serial` with Numba-JIT kernels,
achieving the same ~1000–2000× speedup that `1d_numba` achieved over `1d_serial`.

---

## 1. Performance Diagnosis of `2d_serial`

### Where the time is spent

`2d_serial` has three nested-loop hotspots, all in `rate2d.py`:

| Hotspot | Loop structure | Ops/step | Note |
|---|---|---|---|
| **Interior SBP** | `for i in range(nx): for j in range(ny):` calling `dx2d/dy2d` | `O(nx × ny × nf × p)` | biggest cost |
| **Interface** | `for j in range(ny):` — friction solve per fault point | `O(ny)` | regula-falsi is scalar |
| **RK4 allocs** | 8 arrays `k1..k4 × (l,r)` of shape `(nx, ny, nf)` | `4 × 2 × alloc` | repeated each step |

For `nx=26, ny=51, nf=5, nt=346` (current default):  
work ≈ `346 × 4 × 26 × 51 × 5 × ~8 stencil ops` ≈ **1.5 × 10⁸ Python operations**.

At nx=101, ny=201 (10× finer) this becomes **3 × 10¹⁰** — infeasible in pure Python.

### Why `1d_numba` is so fast

`1d_numba` achieved **1868× speedup** purely from `@njit(cache=True)` at np=1 (no parallelism needed
for 1D at current grid sizes). The entire RK4 step is a single compiled function call.
The same strategy applied to 2D will give similarly large speedups.

For 2D, `prange` over the interior SBP loops **will** show genuine parallel benefit because
work per step ∝ `nx × ny` — far above the thread overhead breakeven  
($n_x \times n_y \gtrsim 200 \times n_p$, i.e. np=8 needs only 40×40 = 1600 points).

---

## 2. Key Differences: 2D vs 1D

| Dimension | 1D | 2D |
|---|---|---|
| State arrays | `v_l(nx), s_l(nx)` — flat | `F_l(nx, ny, nf)` — 3D |
| Unknowns | 2 fields | Mode II: 5 fields `[vx,vy,sxx,syy,sxy]`; Mode III: 3 |
| SBP derivative | `sbp_dx(nx)` in one direction | `sbp_dx_2d(nx, ny, nf)` and `sbp_dy_2d(nx, ny, nf)` |
| Interior loop | `prange(1, nx-1)` | `prange(8, nx-8)` over x, vectorised over y (NumPy slice) |
| Boundary conditions | 2 ends | 4 boundaries per domain: x_outer, x_fault, y_bottom, y_top |
| Interface | single scalar slip/psi | `slip(ny), psi(ny)` — fault is a 1D array |
| Interface loop | single scalar call | `prange(ny)` — each j independent |
| friction_params | `float64[12]` — uniform | `float64[12, ny]` — spatially varying along fault |
| BCs per domain | `r0, r1` (2 scalars) | `r[4]` — x_outer, x_fault, y_bottom, y_top |

---

## 3. Target Directory Structure

```
waveqlab_SF/2d_numba/
├── rupture_2d.py               # CLI runner (mirrors 1d_numba/rupture_1d.py)
├── src/
│   ├── __init__.py
│   ├── kernels_2d.py           # ALL numba-JIT kernels (mirrors 1d_numba/src/kernels.py)
│   ├── sbp_operators_2d.py     # (optional) standalone SBP ops for unit tests
│   └── utils_2d.py             # non-JIT helpers: parse_infile, make_run_id, save_output
├── input/
│   ├── rupture_2d_SW.in        # copy from 2d_serial/input/
│   └── rupture_2d_RS.in
├── output/                     # auto-created by runner
├── plots/                      # auto-created by auxiliary scripts
└── auxiliary/
    ├── inspect_npz.py          # copy from 2d_serial/auxiliary/
    ├── output_2d.py
    └── notebook_plots.py
```

All physics lives in `src/kernels_2d.py`. `rupture_2d.py` only handles I/O and the outer time loop.

---

## 4. `src/kernels_2d.py` — Kernel Design

### 4.1 Layout of `F` arrays

State is `F_l[nx, ny, nf]` and `F_r[nx, ny, nf]` — same as 2d_serial.  
Numba JIT works natively on contiguous NumPy arrays; C-order (default) is fine.

### 4.2 Friction flags

```python
# In kernels_2d.py module header
FRIC_SW = np.float64(0.0)    # matches 1d_numba convention
FRIC_RS = np.float64(1.0)
MODE_II  = np.int64(2)
MODE_III = np.int64(3)
```

`friction_params[12, ny]` is passed as a **contiguous 2D float64 array**.  
Inside `@njit` kernels, index as `friction_params[k, j]`.

---

### 4.3 Phase 1 — `sbp_dx_2d` and `sbp_dy_2d`

#### `sbp_dx_2d(Dxu, u, nx, ny, nf, dx, order)`

Applies x-derivative to all `(j, f)` slices simultaneously.

```
Signature: (Dxu[nx,ny,nf], u[nx,ny,nf], nx, ny, nf, dx, order) → void (in-place)
```

**Key observation:** The x-boundary stencils in `first_derivative_sbp_operators.dx()` already operate
on full rows `u[i, :, :]` — these are pure array operations with no loop.
Only the interior stencil `for i in range(8, m-7)` uses a Python loop.

**Numba translation:**
- Boundary nodes i=0..7 and i=m-7..m: express as 2D array slices  
  `Dxu[0, :, :] = coeff0*u[0, :, :] + coeff1*u[1, :, :] + ...` (no loop needed)  
- Interior: `@njit(parallel=True)` + `prange(8, nx-8)` over i, with vectorised `j`-axis:
  ```python
  for i in prange(8, nx-8):
      Dxu[i, :, :] = (-c3*u[i-3,:,:] + c2*u[i-2,:,:] - c1*u[i-1,:,:]
                      + c1*u[i+1,:,:] - c2*u[i+2,:,:] + c3*u[i+3,:,:]) * inv_dx
  ```
  Each prange iteration processes a full `(ny, nf)` slice — giving `~(nx-16)` parallel tasks,
  each doing `ny × nf × 6 ≈ 51 × 5 × 6 = 1530` multiply-adds.

#### `sbp_dy_2d(Dyu, u, nx, ny, nf, dy, order)`

Same structure but transposes role: boundary stencils along j-axis; interior `prange(8, ny-8)`
over j, with i-axis vectorised:
```python
for j in prange(8, ny-8):
    Dyu[:, j, :] = (-c3*u[:,j-3,:] + c2*u[:,j-2,:] - c1*u[:,j-1,:]
                    + c1*u[:,j+1,:] - c2*u[:,j+2,:] + c3*u[:,j+3,:]) * inv_dy
```

---

### 4.4 Phase 2 — Elastic interior rates

#### `_elastic_interior_rates_2d(D_l, D_r, F_l, F_r, Mat_l, Mat_r, nx, ny, nf, dx, dy, order, mode)`

Computes `D_l[i,j,:] = A_l[i,j] @ DxF_l[i,j,:] + B_l[i,j] @ DyF_l[i,j,:]` for Mode II
(and analogously for Mode III).

```
Inputs:  F_l[nx,ny,nf], F_r[nx,ny,nf], Mat_l/r[nx,ny,3]
Outputs: D_l[nx,ny,nf], D_r[nx,ny,nf]  (written in-place)
```

**Mode II field equations** (nf=5: [vx, vy, sxx, syy, sxy]):
```
D[i,j,0] = 1/rho * (DxF[2] + DyF[4])    # dvx/dt
D[i,j,1] = 1/rho * (DxF[4] + DyF[3])    # dvy/dt
D[i,j,2] = (2µ+λ)*DxF[0] + λ*DyF[1]    # dsxx/dt
D[i,j,3] = (2µ+λ)*DyF[1] + λ*DxF[0]    # dsyy/dt
D[i,j,4] = µ*(DyF[0] + DxF[1])          # dsxy/dt
```

**Mode III field equations** (nf=3: [vz, sxz, syz]):
```
D[i,j,0] = 1/rho * (DxF[1] + DyF[2])   # dvz/dt
D[i,j,1] = µ*DxF[0]                     # dsxz/dt
D[i,j,2] = µ*DyF[0]                     # dsyz/dt
```

Since `Mat_l` is homogeneous in the current setup, `rho, mu, lambda` can be extracted once
as scalars before the inner loop (fast path), with a flag to handle heterogeneous material.

**Design:** Within `@njit(parallel=True)`:
1. Call `sbp_dx_2d` and `sbp_dy_2d` → `DxF_l`, `DyF_l`, `DxF_r`, `DyF_r`
2. Assemble rates with `prange(nx)` outer, vectorise y-axis via NumPy slice operations

---

### 4.5 Phase 2a — Point source injection

`_inject_source(D_l, D_r, t, nx, ny, dx, dy, source_params)`

Vectorised over the spatial Gaussian: compute `g[i,j]` as a 2D array, then add `M * g * f(t)`.
No loop needed — pure NumPy array ops, wrapped in `@njit(cache=True)`.

---

### 4.6 Phase 3 — SAT Boundary Conditions

#### `_penalty_weight_2d(order, dx, dy) → (hx, hy)`

Returns h11 for x and y boundaries; same formula as 1d_numba `_penalty_weight`.

#### 4 boundary kernels (all `@njit(cache=True)`):

| Kernel | Applies to | Domain | Loop |
|---|---|---|---|
| `_bc_x_outer_2d` | `D[0,:,:]` (left) or `D[nx-1,:,:]` (right) | x_outer face | vectorise over j |
| `_bc_y_bottom_2d` | `D[:,0,:]` | y=0 face | vectorise over i |
| `_bc_y_top_2d` | `D[:,ny-1,:]` | y=Ly face | vectorise over i |
| `_bc_x_fault_2d` | `D[nx-1,:,:]` (left) or `D[0,:,:]` (right) | fault face | replaced by interface kernel |

BCs follow the same SAT formula as `boundarycondition.bcm2dx/bcp2dx/bcm2dy/bcp2dy`:
- `r=0`: absorbing → penalty kills outgoing+incoming characteristics
- `r=1`: free surface → traction-free
- `r=-1`: clamped → velocity zero

For Mode II, the SAT penalty acts separately on the P-wave and S-wave characteristics:
```
P-char:  p = (2µ+λ) * vx * n_x + sxx * n_x + sxy * n_y   (normal traction direction)
S-char:  q = µ * vy * n_x + sxy * n_x                      (shear direction)
```

---

### 4.7 Phase 4 — Fault Interface + Friction

#### `_interface_kernel_2d(D_l, D_r, F_l, F_r, slip, psi, friction_params, nx, ny, nf, hx, rho_l, mu_l, lam_l, rho_r, mu_r, lam_r, mode, t, Y_arr, Y0, fault_output0)`

This is the most complex kernel. It handles:
1. **Normal traction coupling** (x-component, locked — `interface_condition` in serial code)  
2. **Shear traction coupling with friction** (y-component, `friction_law` in serial code)  
3. **State evolution** (dslip, dpsi per fault point j)  
4. Writing `fault_output0[j, 0:4]`

```
@njit(parallel=True, cache=True)
def _interface_kernel_2d(...):
    for j in prange(ny):
        # --- material at fault face ---
        rho = rho_l; mu = mu_l; lam = lam_l   # uniform material
        twomulam = 2*mu + lam
        
        # --- extract fault-face fields ---
        vx_l = F_l[nx-1, j, 0];   vx_r = F_r[0, j, 0]
        vy_l = F_l[nx-1, j, 1];   vy_r = F_r[0, j, 1]
        Tx_l = F_l[nx-1, j, 2];   Tx_r = F_r[0, j, 2]
        Ty_l = F_l[nx-1, j, 4];   Ty_r = F_r[0, j, 4]   # sxy = shear

        # --- normal direction: locked interface (x-coupling) ---
        Zp_l = rho * math.sqrt(twomulam/rho)
        Zp_r = rho * math.sqrt(twomulam/rho)
        ...  (same Riemann characteristic formula as interfacedata.interface_condition)
        
        # --- shear direction: friction law (y-coupling) ---
        Zs   = rho * math.sqrt(mu/rho)
        eta_s = 0.5 * Zs                     # symmetric material
        q_m   = Zs*vy_l - Ty_l
        p_p   = Zs*vy_r + Ty_r
        Phi   = 0.5*(p_p/Zs - q_m/Zs)*Zs   # == eta_s * (vR - vL + (TR+TL)/Zs)

        # nucleation perturbation to Tau_0
        tau = friction_params[2, j]
        if fric_flag == FRIC_RS:
            r_nuc = math.fabs(Y_arr[j] - Y0)
            F_nuc = 0.0
            if r_nuc < 3.0:
                F_nuc = math.exp(r_nuc**2 / (r_nuc**2 - 9.0))
            G_nuc = _time_taper(t)           # same as G in serial rate2d.py
            tau += 25.0 * F_nuc * G_nuc
        
        if fric_flag == FRIC_SW:
            # nucleation: SW uses Y_fault indicator from friction_params
            # (background already set correctly; no extra perturbation needed)
            ...
            (call _sw_solve — same as 1d_numba _interface_kernel but scalar)
        else:  # FRIC_RS
            ...
            (call _regula_falsi — copy from 1d_numba/kernels.py, unchanged)
        
        # SAT penalties
        D_l[nx-1, j, :] = ...   # in-place penalty subtraction
        D_r[0,    j, :] = ...
        
        # fault output
        fault_output0[j, 0] = math.fabs(vx_p - vx_m)    # x-slip rate
        fault_output0[j, 1] = math.fabs(vy_p - vy_m)    # y-slip rate
        fault_output0[j, 2] = Tx_m + friction_params[8, j]  # normal stress
        fault_output0[j, 3] = Ty_m + tau                 # shear traction
        
        dslip[j] = math.fabs(vy_p - vy_m)               # Mode II: shear slip rate
        dpsi[j]  = ...                                   # aging law
```

**Important:** `prange` over j is safe because each j is **fully independent** (no data dependency across j-points on the fault). This is the key parallelism opportunity.

---

### 4.8 Phase 5 — `elastic_rate_2d`

```python
@njit(parallel=True, cache=True)
def elastic_rate_2d(D_l, D_r, F_l, F_r, Mat_l, Mat_r,
                    slip, psi, dslip, dpsi, fault_output0,
                    friction_params, Y_arr, Y0, t,
                    nx, ny, nf, dx, dy, dt, order,
                    r_l, r_r, source_params, mode):
    """
    Compute rates D_l, D_r, dslip, dpsi for one RK4 stage.
    r_l[4], r_r[4]: BC flags [x_outer, x_fault, y_bottom, y_top]
    """
    hx, hy = _penalty_weight_2d(order, dx, dy)
    
    # 1. SBP interior rates
    DxF_l = np.empty_like(F_l); DyF_l = np.empty_like(F_l)
    DxF_r = np.empty_like(F_r); DyF_r = np.empty_like(F_r)
    sbp_dx_2d(DxF_l, F_l, nx, ny, nf, dx, order)
    sbp_dy_2d(DyF_l, F_l, nx, ny, nf, dy, order)
    sbp_dx_2d(DxF_r, F_r, nx, ny, nf, dx, order)
    sbp_dy_2d(DyF_r, F_r, nx, ny, nf, dy, order)
    
    # 2. Assemble rate equations
    _assemble_rates_2d(D_l, D_r, DxF_l, DyF_l, DxF_r, DyF_r, Mat_l, Mat_r,
                       nx, ny, nf, mode)
    
    # 3. Point source (optional — only if M0 != 0)
    if source_params[4] != 0.0:
        _inject_source(D_l, t, nx, ny, nf, dx, dy, source_params)
    
    # 4. Interface fault condition + friction → updates D_l, D_r, dslip, dpsi
    _interface_kernel_2d(D_l, D_r, F_l, F_r, slip, psi, dslip, dpsi,
                         friction_params, fault_output0,
                         nx, ny, nf, hx, Mat_l, Mat_r, mode, t, Y_arr, Y0)
    
    # 5. SAT outer BCs
    _apply_outer_bc_2d(D_l, F_l, Mat_l, nx, ny, nf, hx, hy, r_l, 'left',  mode)
    _apply_outer_bc_2d(D_r, F_r, Mat_r, nx, ny, nf, hx, hy, r_r, 'right', mode)
```

---

### 4.9 Phase 6 — `rk4_step_2d`

Mirrors `1d_numba/src/kernels.py :: rk4_step` exactly — classical 4-stage RK4.

```python
@njit(cache=True)
def rk4_step_2d(F_l, F_r, slip, psi,
                Mat_l, Mat_r, friction_params, Y_arr, Y0,
                nx, ny, nf, dx, dy, dt, order,
                r_l, r_r, source_params, mode, t,
                fault_output0):
    """
    In-place classical RK4 step for 2D two-block elastic wave + fault.

    Parameters
    ----------
    F_l, F_r   : float64[nx, ny, nf]   state fields, updated in-place
    slip       : float64[ny]            fault slip accumulator
    psi        : float64[ny]            rate-and-state variable
    ...
    fault_output0 : float64[ny, 6]     on-fault diagnostics (stage 0 only)
    """
    # Stage 1
    k1_l = np.empty_like(F_l);  k1_r = np.empty_like(F_r)
    k1_slip = np.empty(ny);     k1_psi = np.empty(ny)
    fo0 = np.zeros((ny, 6))     # fault output for stage=0 recording
    elastic_rate_2d(k1_l, k1_r, F_l, F_r, Mat_l, Mat_r,
                    slip, psi, k1_slip, k1_psi, fo0, ...)
    fault_output0[:, :] = fo0[:, :]     # record stage-0 data
    
    # Stage 2
    k2_l = np.empty_like(F_l);  k2_r = np.empty_like(F_r)
    k2_slip = np.empty(ny);     k2_psi = np.empty(ny)
    elastic_rate_2d(k2_l, k2_r, F_l + 0.5*dt*k1_l, F_r + 0.5*dt*k1_r, ...,
                    slip + 0.5*dt*k1_slip, psi + 0.5*dt*k1_psi, k2_slip, k2_psi, ...)
    
    # Stages 3, 4 — same pattern

    # In-place update
    c = dt / 6.0
    F_l += c * (k1_l + 2.0*k2_l + 2.0*k3_l + k4_l)
    F_r += c * (k1_r + 2.0*k2_r + 2.0*k3_r + k4_r)
    slip += c * (k1_slip + 2.0*k2_slip + 2.0*k3_slip + k4_slip)
    psi  += c * (k1_psi  + 2.0*k2_psi  + 2.0*k3_psi  + k4_psi)
    
    return (k4_slip, k4_psi)   # final-stage rates for diagnostics
```

**Memory note:** Each stage allocates `(nx, ny, nf) ≈ 26×51×5 = 6630` float64 = 53 kB — trivial.
At nx=201, ny=401 (target large), each stage ≈ 201×401×5 = 403k = 3.2 MB — still fine.

---

## 5. `rupture_2d.py` — Runner

Directly mirrors `1d_numba/rupture_1d.py`; key differences:

1. **Import**: `from kernels_2d import rk4_step_2d, FRIC_SW, FRIC_RS`
2. **`-np N` flag**: `numba.set_num_threads(N)` before first kernel call
3. **Warmup**: call `rk4_step_2d` once on a tiny 3×3 grid with 1 step to trigger JIT compilation
4. **Time loop**: same structure as 1d_numba runner:
   ```python
   for it in range(nt):
       ds, dp = rk4_step_2d(F_l, F_r, slip, psi, ..., t, fault_output0)
       FaultOutput[:, it, :] = fault_output0
       if it % iplot == 0:
           # store domain snapshot
           snap_l[..., snap_idx, :] = F_l
           snap_r[..., snap_idx, :] = F_r
           snap_idx += 1
       t += dt
   ```
5. **Output format**: identical to `2d_serial` `.npz` — same keys, same float32 cast at save time.
   Existing `auxiliary/inspect_npz.py`, `output_2d.py`, `notebook_plots.py` work without changes.

### Command-line interface

```
python3 rupture_2d.py input/rupture_2d_SW.in           # 1 thread (fastest for small grids)
python3 rupture_2d.py input/rupture_2d_SW.in -np 4     # 4 threads
python3 rupture_2d.py input/rupture_2d_SW.in -np 8 --no-warmup
                                                        # skip warmup if .nbi cache exists
```

---

## 6. Data Layout Decisions

| Array | 2d_serial shape | 2d_numba shape | Why |
|---|---|---|---|
| `F_l`, `F_r` | `(nx, ny, nf)` | `(nx, ny, nf)` float64 | unchanged — C-contiguous |
| `slip`, `psi` | `(ny, 1)` | `(ny,)` float64 | flat — no column vectors in JIT |
| `dslip`, `dpsi` | `(ny, 1)` | `(ny,)` float64 | flat |
| `friction_params` | `(12, ny)` | `(12, ny)` float64 | unchanged |
| `fault_output0` | `(ny, 6)` via dict | `(ny, 6)` float64 | pre-allocated, no dict |
| `FaultOutput` | `(ny, nt, 6)` float64 | `(ny, nt, 6)` **→ save as float32** | same as 2d_serial |
| `r_l`, `r_r` | Python list `[4]` | `float64[4]` | indexable in JIT |
| `source_params` | Python list `[7]` | `float64[6]` + `int64[5]` M | need to split |

**`source_params` split** (lists cannot enter `@njit`):
```python
source_scalar = np.array([x0, y0, t0, T, M0], dtype=np.float64)
source_moment = np.array(M, dtype=np.float64)   # shape (5,) or (3,)
```

---

## 7. `sbp_operators_2d.py` — Standalone Unit Test Module

A non-JIT copy of the SBP operators using pure NumPy (no prange) — used **only** in unit tests
to validate JIT output against a reference:
```python
def sbp_dx_numpy(u, dx, order):
    """Pure NumPy version for validation. u: (nx, ny, nf). Returns Dxu."""
    ...
```
This is not part of the production path; keep in `src/` as a reference.

---

## 8. Parallelism Strategy

### When `prange` helps for 2D

Unlike 1D (where thread overhead exceeded compute), 2D has abundant parallelism:

| Operation | Inner work per prange task | At nx=26 ny=51 | At nx=101 ny=201 |
|---|---|---|---|
| `sbp_dx_2d` interior | `ny × nf × 6` ≈ 51×5×6 = 1530 ops | 10 tasks | 85 tasks |
| `sbp_dy_2d` interior | `nx × nf × 6` ≈ 26×5×6 = 780 ops | 35 tasks | 185 tasks |
| `_interface_kernel_2d` | ~50 ops per fault point j | 51 tasks | 201 tasks |

**Recommendation by grid size:**

| Grid | Optimal np | Expected speedup |
|---|---|---|
| nx=26, ny=51 | 1–2 | ~1000× over 2d_serial (JIT alone) |
| nx=51, ny=101 | 4 | ~4000× over 2d_serial |
| nx=101, ny=201 | 8 | ~10000× over 2d_serial |
| nx=201, ny=401 | 16 | ~30000× over 2d_serial |

### prange structure

```
sbp_dx_2d:
  prange(8, nx-8)   → each task: Dxu[i, :, :] = stencil   (ny*nf ops)

sbp_dy_2d:
  prange(8, ny-8)   → each task: Dyu[:, j, :] = stencil   (nx*nf ops)

_interface_kernel_2d:
  prange(ny)         → each task: one fault point j         (~50 scalar ops)

_assemble_rates_2d:
  prange(nx)         → each task: D[i, :, :] = A@Dx + B@Dy  (ny*nf^2 ops)
```

Boundary stencils (i=0..7 and i=m-7..m) are applied as **array slices** (no loop), so they
don't need prange — they run as vectorised NumPy inside the JIT.

---

## 9. Validation Checklist

### Unit tests (by module)

- [ ] `sbp_dx_2d` matches `sbp_dx_numpy` to machine precision for random input
- [ ] `sbp_dy_2d` matches `sbp_dy_numpy` to machine precision for random input
- [ ] BC kernels: plane wave reflection coefficient matches analytical value
- [ ] `_regula_falsi` matches 1d_numba version (same scalar problem, same params)
- [ ] `_interface_kernel_2d` with SW: locked fault → zero slip, correct traction
- [ ] `_interface_kernel_2d` with SW: slipping fault → slip rate matches analytical estimate

### Integration tests (full run vs 2d_serial)

- [ ] Mode II SW, nx=26 ny=51: `FaultOutput` matches `2d_serial` to within float32 tolerance
- [ ] Mode II RS, nx=26 ny=51: slip/psi evolution matches `2d_serial` 
- [ ] Mode III SW (if implemented): same check
- [ ] np=1 and np=4 produce bit-identical results (floating-point associativity safe)

### Convergence test

- [ ] Refine grid 2×: error in slip rate decreases by `2^order` (order = SBP order = 6 → 64×)

### Performance benchmark

After validation, run the same benchmark as `1d_numba/benchmark_scaling.py`:
- Reference: `2d_serial` wall time for nx=26, ny=51
- `2d_numba np=1`: expected ~500–2000× speedup
- `2d_numba np=4, np=8`: should show genuine speedup (unlike 1D)
- Save to `output/benchmark_scaling_2d_<run_id>.npz`

---

## 10. Implementation Order (Step-by-Step)

### Step 1 — Directory scaffold  *(15 min)*
```bash
mkdir -p 2d_numba/{src,input,output,plots,auxiliary}
touch 2d_numba/src/__init__.py
cp 2d_serial/input/*.in        2d_numba/input/
cp 2d_serial/auxiliary/*.py    2d_numba/auxiliary/
```

### Step 2 — `kernels_2d.py`: SBP operators  *(1–2 h)*
Port `sbp_dx_2d` and `sbp_dy_2d` from `first_derivative_sbp_operators.dx/dy`.
- Boundary nodes: translate row-wise array expressions to `@njit` array slices
- Interior: add `prange` over i (or j)
- Write unit test: compare against `first_derivative_sbp_operators.dx2d` on random fields

### Step 3 — `kernels_2d.py`: BC kernels  *(1 h)*
Port `boundarycondition.bcm2dx / bcp2dx / bcm2dy / bcp2dy` into 4 `@njit` functions.
- Mode II SAT formulas: decompose into P+S characteristics and apply r-weighted penalty
- Test: single-domain pulse propagation; absorbing BC should produce no reflection

### Step 4 — `kernels_2d.py`: friction helpers  *(30 min)*
Copy `_regula_falsi`, `_tau_strength_sw`, `_slip_weakening_solve`, `_rate_and_state_solve`
verbatim from `1d_numba/src/kernels.py`. No changes needed — they are already scalar `@njit`.

### Step 5 — `kernels_2d.py`: `_interface_kernel_2d`  *(2–3 h)*
This is the most complex step. Port `interfacecondition2d()` from `rate2d.py`:
- Replace Python loop `for j in range(ny)` → `@njit(parallel=True)` with `prange(ny)`
- Normal (x) coupling: locked interface — port `interface_condition()` from `interfacedata.py`
- Shear (y) coupling: friction — call `_sw_solve` / `_regula_falsi` per fault point j
- State evolution: dslip[j], dpsi[j] — already scalar operations
- SAT contribution to D_l[nx-1,j,:] and D_r[0,j,:] — same formula as serial, in-place

### Step 6 — `kernels_2d.py`: `elastic_rate_2d`  *(1 h)*
Assemble steps 2–5 into a single `@njit(parallel=True, cache=True)` function.
Validate: run one rate evaluation and compare D_l, D_r against serial `rate2d.elastic_rate2d`.

### Step 7 — `kernels_2d.py`: `rk4_step_2d`  *(1 h)*
Build `@njit(cache=True) rk4_step_2d` — 4 calls to `elastic_rate_2d`, in-place update.
Validate: run 10 time steps; compare `F_l, F_r, slip, psi` against serial.

### Step 8 — `rupture_2d.py`  *(1 h)*
Copy runner structure from `1d_numba/rupture_1d.py`. Adapt:
- `parse_infile`, `build_params`, `make_run_id`, `save_output` — copy/adapt from `2d_serial/rupture_2d.py`
- Warmup call on tiny grid before main loop
- `-np` flag → `numba.set_num_threads(N)`
- Timing: `timeit.default_timer()` around the time loop

### Step 9 — End-to-end test  *(30 min)*
```bash
cd 2d_numba
source /scratch/aimran/Dev/.venv/bin/activate
python3 rupture_2d.py input/rupture_2d_SW.in -np 1
python3 ../2d_serial/auxiliary/inspect_npz.py output/*.npz --plot
```
Compare `FaultOutput` against `2d_serial/output/rupture_2d_SW_*.npz`.

### Step 10 — Benchmark  *(1 h)*
Write `benchmark_scaling_2d.py` (copy from `1d_numba/benchmark_scaling.py`, adapt for 2D).
Run over np=1,2,4,8; report speedup. Save report to `benchmark_scaling_report_2d.md`.

---

## 11. Estimated Speedup (projected)

| Metric | 2d_serial (nx=26, ny=51) | 2d_numba np=1 (projected) |
|---|---|---|
| Wall time per run | ~170 s | ~0.1–0.3 s |
| Step time | ~490 ms | ~0.3–0.9 ms |
| Speedup | 1× | **~500–1700×** |

At the PINO training scale (e.g. 1000 parameter samples × nx=51 × ny=101 × nt=700):
- 2d_serial: `1000 × ~170s × 4 = 190 hours`
- 2d_numba np=1: `1000 × ~0.5s × 4 = 560 s ≈ 9 minutes`
- 2d_numba np=8: `~2 minutes`

---

## 12. Dependency Notes

```
numba >= 0.57     # np.empty_like inside @njit, parallel=True
numpy >= 1.24
```

Check after install:
```python
import numba
print(numba.__version__)
numba.set_num_threads(4)
```

No changes to `auxiliary/` scripts — they consume `.npz` output directly.
