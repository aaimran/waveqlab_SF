# Plan: Build Parallel Python 1D Seismic Wave Solver (Modeled on waveqlab3d_Dev)

## 1. Code Analysis Summary

### waveqlab3d_Dev (3D Fortran MPI) Architecture

| Layer | File(s) | Role |
|---|---|---|
| **Program entry** | `main.f90` | Reads input file, initializes domain, runs time loop |
| **Domain** | `domain.f90`, `datatypes.f90` | Container for all blocks + interfaces; reads namelists |
| **Block** | `block.f90` | One structured mesh subdomain: grid, material, fields, PML, BCs |
| **Fields & rates** | `fields.f90` | `F` (state) and `DF` (rate ≡ ∂F/∂t) arrays |
| **Spatial operators** | `RHS_Interior.f90`, `JU_xJU_yJU_z6.f90` | SBP interior stencils (traditional, upwind, DRP), orders 2–9 |
| **Boundary conditions** | `BoundaryConditions.f90`, `boundary.f90` | SAT penalty enforcement on each of 6 block faces |
| **Interface** | `iface.f90`, `Interface_Condition.f90` | Fault SAT coupling; slip `S`, state `W`, traction `T` |
| **Friction** | embedded in `Interface_Condition.f90` | Locked, SW, RS (rate-and-state via Regula-Falsi) |
| **Time stepping** | `time_step.f90` | Low-storage RK (RK1/2/3 Williamson, RK4 Kennedy-Carpenter 5-stage) |
| **MPI decomp** | `mpi3dbasic.f90`, `mpi3dcomm.f90`, `mpi3d_interface.f90` | 3D Cartesian process grid, ghost-node halo exchange, split-pair communicators per block |
| **Output** | `seismogram.f90`, `plane_output.f90`, `fault_output.f90` | Per-station, per-plane, per-fault parallel I/O |
| **PML** | `pml.f90` | Auxiliary variables on 6 sponge layers |

**Key MPI patterns in _Dev:**
- Two blocks each owned by a disjoint process sub-communicator; blocks communicate only across the fault interface.
- 3D Cartesian topology (`MPI_Dims_create` + `MPI_Cart_create`); halo exchange via `MPI_Type_vector` derived types.
- Domain decomposition: `decompose1d` gives each rank its local `[m, p]` index range with ghost-node extensions `[mb, pb]`.
- All RK stages involve: ghost exchange → `scale_rates` → `set_rates` (interior SBP) → `exchange_fields_interface` (fault BCs) → `enforce_bound_iface_conditions` (SAT) → `update_fields`.

### waveqlab_SF/1d (1D Python) Architecture

| Layer | File | Role |
|---|---|---|
| **Notebook** | `rupture_1d_ep-science-festival_2025.ipynb` | Driver; sets params, calls time integrator, plots |
| **Time integrator** | `time_integrator.py` | Classic RK4 (4-stage, not low-storage) over two blocks (L and R) |
| **Rate** | `rate.py` | `elastic_rate`: SBP derivative + interior rates + `impose_bcm/bcp` + `interface.couple_friction` |
| **SBP operators** | `first_derivative_sbp_operators.py` | `dx()` for orders 2, 4, 6; pure Python loops (slow) |
| **Boundary** | `boundarycondition.py`, `boundarydata.py` | SAT terms for left/right absorbing/reflecting BCs |
| **Interface** | `interface.py` | `couple_friction`: extracts boundary states, calls friction law, applies SAT penalty |
| **Interface data** | `interfacedata.py` | `friction_law()` for locked, SW, LN, RS; `regula_falsi` for RS |
| **Utils** | `utils.py` | Miscellaneous helpers |

**Key differences from _Dev:**
- Serial only (no parallelism).
- Two blocks hard-coded side by side (left/right); no general block topology.
- Classic 4-stage RK4 (not low-storage Kennedy-Carpenter).
- SBP stencils use Python `for` loops — O(N) per grid point, very slow for large N.
- No output layer, no PML, no anelastic Q.
- All parameters passed as flat function arguments (no domain/block data structure).

---

## 2. Plan: Parallel Python 1D Solver Modeled on _Dev

The goal is to keep the *physics and numerics* identical to the current 1D code while:
1. Restructuring into an object-oriented domain/block hierarchy (like _Dev).
2. Vectorizing all SBP stencils to NumPy array operations.
3. Adding domain decomposition and parallel communication with `mpi4py`.
4. Upgrading the time integrator to the low-storage Kennedy-Carpenter RK4 (5-stage).
5. Adding a clean I/O layer.

---

## 3. Module / File Structure

```
waveqlab_SF/1d_parallel/
├── main.py                       # entry point (like main.f90)
├── config.py                     # dataclasses for all parameters (replaces namelist)
├── domain.py                     # Domain class: owns blocks + interface (like domain.f90)
├── block.py                      # Block class: grid, material, fields, BCs (like block.f90)
├── fields.py                     # FieldState dataclass: F and DF arrays
├── sbp_operators.py              # Vectorized SBP first-derivative operators (dx), orders 2/4/6
├── boundary_conditions.py        # SAT terms for left/right BCs (like BoundaryConditions.f90)
├── interface_condition.py        # Fault SAT coupling (like Interface_Condition.f90)
├── friction.py                   # Friction laws: locked, SW, LN, RS
├── time_integrator.py            # Low-storage RK time-stepping (like time_step.f90)
├── parallel.py                   # MPI decomposition and halo exchange (like mpi3dcomm.f90)
├── output.py                     # Seismogram and field output
└── input/
    └── example.py                # Example parameter setup (replaces *.in namelist)
```

---

## 4. Step-by-Step Implementation Plan

### Step 1 — `config.py`: Parameter Dataclasses

Replace flat function arguments with structured dataclasses inspired by `block_temp_parameters` and `problem_list` in _Dev.

```python
@dataclass
class BlockConfig:
    n: int              # number of grid points
    x_min: float
    x_max: float
    rho: float
    mu: float
    order: int          # SBP order: 2, 4, or 6
    bc_left: str        # 'absorbing' | 'reflecting' | 'free'  (only for outermost boundary)
    bc_right: str
    r_left: float = 0.0     # reflection coefficient
    r_right: float = 0.0

@dataclass
class FrictionConfig:
    law: str            # 'locked' | 'SW' | 'LN' | 'RS'
    Tau_0: float
    sigma_n: float
    # SW params
    alp_s: float = 0.6
    alp_d: float = 0.1
    D_c: float = 0.4
    # RS params
    a: float = 0.008
    b: float = 0.012
    L: float = 0.02
    f0: float = 0.6
    V0: float = 1e-6

@dataclass
class SimConfig:
    t_final: float
    CFL: float = 0.5
    rk_order: int = 4       # 1, 2, 3, or 4 (Kennedy-Carpenter)
    output_stride: int = 1
    n_procs_block1: int = 0  # 0 = auto
    n_procs_block2: int = 0
```

### Step 2 — `sbp_operators.py`: Vectorized SBP Stencils

Replace all `for` loops in `first_derivative_sbp_operators.py` with NumPy array slicing.
This gives O(1) passes over the array instead of O(N) Python loop iterations.

```python
import numpy as np

def dx(u: np.ndarray, dx: float, order: int) -> np.ndarray:
    """
    Vectorized SBP first derivative.
    u: shape (n,) or (n, nf) where nf = number of fields
    Returns ux of same shape.
    """
    n = u.shape[0]
    ux = np.empty_like(u)

    if order == 2:
        ux[0]  = (u[1] - u[0]) / dx
        ux[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        ux[-1] = (u[-1] - u[-2]) / dx

    elif order == 4:
        # boundary stencils (rows 0-3 and mirror at end)
        # ... coefficients identical to current file but applied as array ops
        _apply_order4_boundary(ux, u, dx)
        # interior: 4th-order centered stencil vectorized
        ux[4:-4] = (u[2:-6] - 8*u[3:-5] + 8*u[5:-3] - u[6:-2]) / (12*dx)

    elif order == 6:
        _apply_order6_boundary(ux, u, dx)
        # interior: 6th-order centered
        ux[8:-8] = (-u[5:-11]/60 + 3*u[6:-10]/20 - 3*u[7:-9]/4 +
                     3*u[9:-7]/4 - 3*u[10:-6]/20 + u[11:-5]/60) / dx
    return ux
```

**Benefit:** For N=10,000 points and 6 fields this is ~100× faster than Python loops.

### Step 3 — `fields.py`: FieldState Dataclass

```python
@dataclass
class FieldState:
    v: np.ndarray   # particle velocity,  shape (n,)
    s: np.ndarray   # shear stress,       shape (n,)
    dv: np.ndarray  # rate of v
    ds: np.ndarray  # rate of s
```

Mirroring `block_fields` (`F` and `DF`) in _Dev's `datatypes.f90`.

### Step 4 — `block.py`: Block Class

```python
class Block:
    def __init__(self, cfg: BlockConfig, comm):
        self.cfg = cfg
        self.comm = comm          # MPI sub-communicator for this block
        # domain decomposition
        self.local_slice = decompose1d(cfg.n, comm.Get_size(), comm.Get_rank())
        self.n_local = len(self.local_slice)
        self.x = linspace(cfg.x_min, cfg.x_max, cfg.n)[self.local_slice]
        self.dx = (cfg.x_max - cfg.x_min) / (cfg.n - 1)
        # fields
        self.F = FieldState(...)   # initialize v=0, s=0 + ghost cells
        # ghost cells: 1 extra point on each interior-facing end for halo exchange
```

### Step 5 — `parallel.py`: Domain Decomposition and Halo Exchange

Mirrors `mpi3dcomm.f90` + `mpi3dbasic.f90`, adapted for 1D.

```python
from mpi4py import MPI

def decompose1d(n: int, nprocs: int, rank: int) -> slice:
    """Return the local index slice for one rank."""
    base, rem = divmod(n, nprocs)
    start = rank * base + min(rank, rem)
    length = base + (1 if rank < rem else 0)
    return slice(start, start + length)

class BlockComm:
    """
    Manages a 1D Cartesian sub-communicator for one block.
    Each rank holds local_n interior points plus 1 ghost cell on each side.
    """
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        # neighbor ranks (-1 = boundary)
        self.rank_left  = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.rank_right = self.rank + 1 if self.rank < self.size-1 else MPI.PROC_NULL

    def halo_exchange(self, v: np.ndarray, s: np.ndarray):
        """
        Non-blocking send/recv of boundary ghost cells between neighbors.
        Uses MPI.PROC_NULL for domain boundaries (no-op automatically).
        """
        reqs = []
        # send right boundary → right neighbor; recv from right neighbor into ghost
        reqs.append(self.comm.Isend(v[-2:-1], dest=self.rank_right))
        reqs.append(self.comm.Irecv(v[-1:],   source=self.rank_right))
        # send left boundary → left neighbor; recv from left neighbor into ghost
        reqs.append(self.comm.Isend(v[1:2],  dest=self.rank_left))
        reqs.append(self.comm.Irecv(v[0:1],  source=self.rank_left))
        MPI.Request.Waitall(reqs)
        # repeat for stress s
        ...
```

**Two-block communicator split** (mirrors _Dev's `block_comms`):

```python
# In main.py:
world = MPI.COMM_WORLD
nprocs = world.Get_size()
# Split: first half owns block 1, second half owns block 2
color = 0 if world.Get_rank() < nprocs // 2 else 1
block_comm = world.Split(color, world.Get_rank())
```

The fault interface between the two blocks communicates only across the **pair of ranks at the shared boundary** — exactly one rank per block touches the fault.

### Step 6 — `interface_condition.py`: Parallel Fault SAT

The fault is located at the right end of block 1 (on rank `nprocs//2 - 1`) and the left end of block 2 (on rank `nprocs//2`).  These two ranks exchange fault-face fields using an **inter-block communicator**.

```python
class FaultComm:
    """
    Point-to-point communicator between the two fault-adjacent ranks.
    Rank 0 of this communicator owns block-1 fault face.
    Rank 1 owns block-2 fault face.
    """
    def __init__(self, world_comm, is_fault_rank: bool):
        color = 0 if is_fault_rank else MPI.UNDEFINED
        self.comm = world_comm.Split(color, world_comm.Get_rank())
        self.active = is_fault_rank

    def exchange_fault_fields(self, v_face, s_face):
        """
        Swap the single face value with the opposite block.
        Returns (v_opp, s_opp).
        """
        v_opp = np.empty_like(v_face)
        s_opp = np.empty_like(s_face)
        if self.active:
            peer = 1 - self.comm.Get_rank()
            self.comm.Sendrecv(v_face, dest=peer, recvbuf=v_opp, source=peer)
            self.comm.Sendrecv(s_face, dest=peer, recvbuf=s_opp, source=peer)
        return v_opp, s_opp
```

### Step 7 — `time_integrator.py`: Low-Storage Kennedy-Carpenter RK4 (5-stage)

Matches `time_step.f90` exactly—same coefficients, same stage structure.

```python
# Kennedy-Carpenter (5,4) low-storage coefficients
_A = np.array([0.0,
               -567301805773/1357537059087,
               -2404267990393/2016746695238,
               -3550918686646/2091501179385,
               -1275806237668/842570457699])
_B = np.array([1432997174477/9575080441755,
               5161836677717/13612068292357,
               1720146321549/2090206949498,
               3134564353537/4481467310338,
               2277821191437/14882151754819])
_C = np.array([0.0,
               1432997174477/9575080441755,
               2526269341429/6820363962896,
               2006345519317/3224310063776,
               2802321613138/2924317926251,
               1.0])

def time_step_RK(domain, dt, n):
    """One time step using 5-stage low-storage RK4."""
    # dU_stage accumulates: dU = A*dU + B*dt*L(U)
    for stage, (A, B, C) in enumerate(zip(_A, _B, _C)):
        # halo exchange within each block
        for blk in domain.blocks:
            blk.comm.halo_exchange(blk.F.v, blk.F.s)

        # compute interior SBP rates (non-blocking safe: no cross-block comms yet)
        for blk in domain.blocks:
            compute_interior_rates(blk)

        # fault face exchange and SAT
        domain.fault.exchange_and_apply_sat(domain, stage)

        # apply outer boundary SAT
        for blk in domain.blocks:
            apply_boundary_sat(blk)

        # low-storage update: dU = A*dU + B*dt*L(U); U = U + dU
        for blk in domain.blocks:
            blk.dF.v = A * blk.dF.v + B * dt * blk.F.dv
            blk.dF.s = A * blk.dF.s + B * dt * blk.F.ds
            blk.F.v += blk.dF.v
            blk.F.s += blk.dF.s

        # update interface slip and state
        domain.iface.dS = A * domain.iface.dS + B * dt * domain.iface.DS
        domain.iface.S += domain.iface.dS
        domain.iface.dW = A * domain.iface.dW + B * dt * domain.iface.DW
        domain.iface.W += domain.iface.dW

        domain.t = domain.t0 + C * dt

        # output on first stage of output steps
        if stage == 0 and n % domain.cfg.output_stride == 0:
            domain.output.record(domain, n)
```

**Memory advantage:** Low-storage RK needs only **2 extra arrays** (dF per block) instead of 4 per classic RK4 — important for large grids.

### Step 8 — `friction.py`: All Friction Laws

Keep existing physics but refactor into a clean class hierarchy (Strategy pattern):

```python
class FrictionLaw:
    def solve(self, phi, S, W, eta, params) -> dict: ...

class LockedFriction(FrictionLaw): ...
class SlipWeakeningFriction(FrictionLaw): ...
class LinearFriction(FrictionLaw): ...
class RateAndStateFriction(FrictionLaw):
    def _regula_falsi(self, V0, Phi, eta, sigma_n, psi, V_ref, a): ...
```

### Step 9 — `output.py`: Seismogram and Field Output

```python
class Output:
    def __init__(self, cfg, comm):
        # Only rank 0 of each block writes, or use parallel HDF5 via h5py
        ...
    def record(self, domain, n):
        # gather seismogram stations from distributed ranks
        # write fault slip, slip rate, traction to file
        ...
```

For large simulations, use `h5py` with parallel mode (`h5py.File(..., driver='mpio', comm=comm)`) — mirrors _Dev's `mpi3dio.f90`.

### Step 10 — `main.py`: Entry Point

```python
from mpi4py import MPI
from config import SimConfig, BlockConfig, FrictionConfig
from domain import Domain

def main():
    world = MPI.COMM_WORLD
    cfg_sim, cfg_b1, cfg_b2, cfg_fric = load_config()  # from input file or script

    domain = Domain(cfg_sim, cfg_b1, cfg_b2, cfg_fric, world)

    for n in range(1, domain.nt + 1):
        time_step_RK(domain, domain.dt, n)
        if world.Get_rank() == 0:
            print(f"n={n:6d}  t={domain.t:.6e}")

    domain.close()
    MPI.Finalize()

if __name__ == '__main__':
    main()
```

Run with:
```bash
mpiexec -n 8 python main.py input/example.py
```

---

## 5. Parallelism Strategy (mirroring _Dev)

| _Dev concept | Python equivalent |
|---|---|
| `MPI_COMM_WORLD` | `MPI.COMM_WORLD` |
| Block sub-communicator via `MPI_Comm_split` | `world.Split(color, rank)` |
| `decompose1d` → `[mq, pq]`, ghost `[mbq, pbq]` | `decompose1d(n, nprocs, rank)` with `np.pad` ghost |
| `MPI_Type_vector` halo exchange | `Isend` / `Irecv` on 1-point boundary arrays |
| Two-block fault communicator | `FaultComm` (split to 2 fault-adjacent ranks) |
| Ghost node exchange before every RK stage | `BlockComm.halo_exchange()` each stage |
| Low-storage RK (`scale_rates`, `update_fields`) | In-place `dU = A*dU + B*dt*L(U)` |
| Parallel I/O (`mpi3dio`) | `h5py` with MPI driver |

---

## 6. Performance Considerations

| Issue | Solution |
|---|---|
| Python `for` loops in SBP stencils | NumPy vectorized array ops in `sbp_operators.py` |
| Classic RK4 allocates 4×N temp arrays | Low-storage RK4: 2 extra arrays (dF) |
| Serial Python GIL limits threading | Use `mpi4py` multi-process; no threads |
| Large output: all data hits rank 0 | Parallel HDF5 via `h5py` MPI mode |
| Nonlinear RS friction (Regula-Falsi) | Keep scalar solve at fault point; only 1 rank per block sees it |
| Profile hotspots | `cProfile` + `mpi4py` per-rank timing; identify if SBP or friction dominates |

---

## 7. Suggested Implementation Order

1. **`config.py`** — define all dataclasses; no dependencies.
2. **`sbp_operators.py`** — vectorize `dx`; validate against existing `first_derivative_sbp_operators.py`.
3. **`fields.py`** — trivial dataclass.
4. **`block.py`** (serial, single block first) — integrate SBP + boundary SAT; reproduce 1D pulse propagation.
5. **`friction.py`** — port all friction laws; unit-test RS regula-falsi independently.
6. **`interface_condition.py`** (serial two-block) — reproduce existing 1D rupture notebook output exactly.
7. **`time_integrator.py`** — swap classic RK4 for low-storage KC-RK4; verify convergence.
8. **`parallel.py`** — `decompose1d` + `BlockComm.halo_exchange`; test with 2 and 4 ranks.
9. **`domain.py`** + **`main.py`** — assemble everything; run fault rupture in parallel.
10. **`output.py`** — seismograms + fault output with optional parallel HDF5.

---

## 8. Validation Checklist

- [ ] Single-block pulse propagation: convergence rate = SBP order (2/4/6).
- [ ] Two-block locked interface: energy conserved to machine precision.
- [ ] Slip-weakening: peak slip velocity and rupture time match serial 1D notebook.
- [ ] Rate-and-state: steady-state slip rate matches analytical solution.
- [ ] 2-rank parallel result identical to 1-rank serial result (to floating-point).
- [ ] 8-rank parallel result identical to 2-rank result.
- [ ] Timing: linear strong-scaling up to N_procs = N_points / 100 (communication-to-computation ratio).
