#!/usr/bin/env python3
"""
benchmark_scaling.py — Thread scaling benchmark for 1D SBP-SAT rupture simulation
====================================================================================
Usage:
    python3 benchmark_scaling.py input/rupture_1d_SW.in
    python3 benchmark_scaling.py input/rupture_1d_SW.in --nreps 5
    python3 benchmark_scaling.py input/rupture_1d_SW.in --threads 1 2 4 8 16 32 40

Produces:
    output/benchmark_scaling_{run_id}.npz   — raw timing arrays
    Prints an ASCII table to stdout
"""

import argparse
import json
import os
import sys
import timeit

# ---------------------------------------------------------------------------
# MUST set NUMBA_NUM_THREADS before any numba import.
# Parse --threads from argv early; take the maximum as the thread pool size.
# ---------------------------------------------------------------------------
def _early_max_threads():
    """Quick argv scan for --threads values to determine pool size."""
    max_t = 40          # default ceiling
    try:
        idx = sys.argv.index('--threads')
        vals = []
        for tok in sys.argv[idx + 1:]:
            if tok.startswith('-'):
                break
            try:
                vals.append(int(tok))
            except ValueError:
                break
        if vals:
            max_t = max(vals)
    except ValueError:
        pass
    return max_t

_pool_size = _early_max_threads()
os.environ['NUMBA_NUM_THREADS'] = str(_pool_size)
os.environ['OMP_NUM_THREADS']   = str(_pool_size)

import numpy as np  # noqa: E402 (after env set)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rupture_1d import (parse_infile, build_params, validate,
                        init_fields, build_friction_parameters, make_run_id)

import numba
from kernels import rk4_step

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_THREADS = [1, 2, 4, 8, 16, 32, 40]
DEFAULT_NREPS   = 3


# ---------------------------------------------------------------------------
# Benchmark one thread count
# ---------------------------------------------------------------------------

def _run_once(p, friction_params, nthreads):
    """Reset state, run the full nt-step loop, return (wall_s, step_compute_arr)."""
    nx    = p['nx']
    nt    = p['nt']
    dx    = p['dx']
    dt    = p['dt']
    order = int(p['order'])
    r0_l  = float(p['r0_l'])
    r1_r  = float(p['r1_r'])
    rho   = float(p['rho'])
    mu    = float(p['rho']) * float(p['cs']) ** 2
    Tau_0 = float(p['Tau_0'])

    _, _, v_l, s_l, v_r, s_r, slip, psi = init_fields(p)

    step_compute = np.empty(nt, dtype=np.float64)

    numba.set_num_threads(nthreads)

    t0_wall = timeit.default_timer()
    for n in range(nt):
        t0 = timeit.default_timer()
        rk4_step(v_l, s_l, v_r, s_r, slip, psi,
                 rho, mu, nx, dx, order, r0_l, r1_r, dt, friction_params)
        step_compute[n] = timeit.default_timer() - t0

    wall = timeit.default_timer() - t0_wall
    return wall, step_compute


def benchmark_threads(p, friction_params, thread_counts, nreps):
    """
    For each thread count, run `nreps` full simulations.
    Returns dict: nthreads → {'wall_min', 'wall_mean', 'wall_max',
                               'step_ms_mean', 'step_ms_std', per_rep_walls}
    """
    results = {}
    for nt_val in thread_counts:
        actual = min(nt_val, _pool_size)   # cap to the pre-initialized pool size
        if actual != nt_val:
            print(f"  [WARN] requested {nt_val} threads → capped to {actual}")
        walls = []
        step_arrs = []
        for rep in range(nreps):
            wall, steps = _run_once(p, friction_params, actual)
            walls.append(wall)
            step_arrs.append(steps)
        step_all = np.concatenate(step_arrs)
        results[nt_val] = dict(
            nthreads      = actual,
            wall_min      = min(walls),
            wall_mean     = float(np.mean(walls)),
            wall_max      = max(walls),
            wall_reps     = walls,
            step_ms_mean  = float(step_all.mean() * 1e3),
            step_ms_std   = float(step_all.std()  * 1e3),
            step_ms_min   = float(step_all.min()  * 1e3),
            step_ms_max   = float(step_all.max()  * 1e3),
        )
        print(f"  np={nt_val:>2d}  wall_min={min(walls):.4f}s  "
              f"mean_step={step_all.mean()*1e3:.3f}ms")
    return results


# ---------------------------------------------------------------------------
# Load serial reference timing
# ---------------------------------------------------------------------------

def _find_serial_ref(run_id):
    """Auto-detect 1d_serial timing npz in sibling directory."""
    candidate = os.path.join(_ROOT, '..', '1d_serial', 'output',
                             f'rupture_SW_{run_id}_timing.npz')
    candidate = os.path.normpath(candidate)
    return candidate if os.path.isfile(candidate) else None


def load_serial_timing(run_id, serial_path=None):
    """
    Load serial reference.  Returns dict with wall_time_s, mean_step_ms,
    or None if not found.
    """
    path = serial_path or _find_serial_ref(run_id)
    if path is None or not os.path.isfile(path):
        return None
    try:
        d  = np.load(path, allow_pickle=True)
        tm = json.loads(str(d['metadata']))
        return dict(
            wall_s        = float(tm.get('wall_time_s', 0.0)),
            step_ms_mean  = float(tm.get('mean_step_compute_ms', 0.0)),
            step_ms_std   = float(d['step_time_compute'].std() * 1e3),
            path          = path,
        )
    except Exception as exc:
        print(f"  [WARN] Could not load serial reference: {exc}")
        return None


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

W = 76

def _hline(char='-'):
    print(char * W)


def print_table(results, serial_ref, p, nreps, warmup_s):
    nx   = p['nx']
    nt   = p['nt']

    print()
    _hline('=')
    print(f"  1D SBP-SAT RUPTURE SIMULATION — THREAD SCALING BENCHMARK")
    print(f"  nx={nx}  nt={nt}  order={p['order']}  fric_law={p['fric_law']}")
    print(f"  nreps={nreps} (best-of-{nreps} used for speedup)   "
          f"JIT warmup: {warmup_s:.2f} s")
    _hline('=')

    # Header
    hdr = (f"  {'Config':<14s}  {'Wall min':>9s}  {'Wall mean':>9s}  "
           f"{'Step mean':>9s}  {'Step std':>8s}  {'Speedup':>7s}  {'Eff%':>5s}")
    print(hdr)
    _hline()

    # Serial reference row
    if serial_ref:
        ref_wall  = serial_ref['wall_s']
        ref_step  = serial_ref['step_ms_mean']
        print(f"  {'1d_serial (ref)':<14s}  {ref_wall:>9.3f}s  {'—':>9s}  "
              f"{ref_step:>8.3f}ms  {serial_ref['step_ms_std']:>7.3f}ms  "
              f"  {'1.00x':>6s}  {'100%':>5s}")
        _hline()
    else:
        ref_wall = None
        print(f"  [serial reference not found — run 1d_serial/rupture_1d.py first]")
        _hline()

    # Numba rows
    for nthreads, r in sorted(results.items()):
        best_wall = r['wall_min']
        speedup_str = '—'
        eff_str     = '—'
        if ref_wall:
            speedup = ref_wall / best_wall
            eff     = speedup / nthreads * 100.0
            speedup_str = f'{speedup:.1f}x'
            eff_str     = f'{eff:.0f}%'
        label = f'numba np={nthreads}'
        print(f"  {label:<14s}  {best_wall:>9.4f}s  {r['wall_mean']:>9.4f}s  "
              f"{r['step_ms_mean']:>8.3f}ms  {r['step_ms_std']:>7.3f}ms  "
              f"  {speedup_str:>6s}  {eff_str:>5s}")

    _hline('=')
    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results, serial_ref, p, nreps, warmup_s, run_id):
    out_dir  = os.path.join(_ROOT, 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'benchmark_scaling_{run_id}.npz')

    thread_counts = np.array(sorted(results.keys()), dtype=np.int32)
    wall_min  = np.array([results[n]['wall_min']  for n in sorted(results)], dtype=np.float64)
    wall_mean = np.array([results[n]['wall_mean'] for n in sorted(results)], dtype=np.float64)
    step_mean = np.array([results[n]['step_ms_mean'] for n in sorted(results)], dtype=np.float64)
    step_std  = np.array([results[n]['step_ms_std']  for n in sorted(results)], dtype=np.float64)

    speedup = np.full(len(thread_counts), np.nan)
    if serial_ref:
        speedup = serial_ref['wall_s'] / wall_min

    meta = dict(
        run_id        = run_id,
        nx            = p['nx'],
        nt            = p['nt'],
        order         = p['order'],
        fric_law      = p['fric_law'],
        nreps         = nreps,
        warmup_s      = warmup_s,
        serial_wall_s = serial_ref['wall_s']       if serial_ref else None,
        serial_step_ms= serial_ref['step_ms_mean'] if serial_ref else None,
        serial_path   = serial_ref['path']          if serial_ref else None,
    )

    np.savez_compressed(
        out_file,
        thread_counts = thread_counts,
        wall_min      = wall_min,
        wall_mean     = wall_mean,
        step_mean_ms  = step_mean,
        step_std_ms   = step_std,
        speedup       = speedup,
        metadata      = json.dumps(meta, default=str),
    )
    print(f"Saved: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Thread scaling benchmark for 1D numba rupture simulation')
    parser.add_argument('infile',
                        help='path to .in parameter file')
    parser.add_argument('--threads', type=int, nargs='+',
                        default=DEFAULT_THREADS,
                        help=f'thread counts to benchmark (default: {DEFAULT_THREADS})')
    parser.add_argument('--nreps', type=int, default=DEFAULT_NREPS,
                        help=f'repetitions per thread count (default: {DEFAULT_NREPS})')
    parser.add_argument('--serial-ref', default=None,
                        help='path to 1d_serial _timing.npz file (auto-detected if omitted)')
    args = parser.parse_args()

    if not os.path.isfile(args.infile):
        print(f"ERROR: Input file not found: {args.infile}")
        sys.exit(1)

    # Build params
    p = build_params(parse_infile(args.infile))
    validate(p)
    friction_params = build_friction_parameters(p)
    run_id = make_run_id(p)

    threads = sorted(set(n for n in args.threads if n >= 1))

    print(f"\nBenchmarking: {args.infile}")
    print(f"nx={p['nx']}  nt={p['nt']}  order={p['order']}  "
          f"fric_law={p['fric_law']}  run_id={run_id}")
    print(f"Thread counts : {threads}")
    print(f"Repetitions   : {args.nreps}")

    # JIT warmup — compile at 1 thread (fast), then initialize pool at _pool_size
    print(f"\nJIT warmup (compile at np=1, then set pool to {_pool_size})...")
    t_wu = timeit.default_timer()
    _run_once(p, friction_params, 1)                # triggers compilation cheaply at 1 thread
    warmup_s = timeit.default_timer() - t_wu
    # Now initialize the thread pool at full size before benchmarking
    numba.set_num_threads(_pool_size)
    max_threads = numba.get_num_threads()
    print(f"  done in {warmup_s:.2f} s  "
          f"(threading_layer={numba.threading_layer()}  max_threads={max_threads})\n")

    # Cap thread counts to what numba actually initialized
    if any(n > max_threads for n in threads):
        print(f"[INFO] numba thread pool = {max_threads}.  "
              f"Counts > {max_threads} will be capped to that value.")

    # Benchmark
    print("Running benchmark ...")
    results = benchmark_threads(p, friction_params, threads, args.nreps)

    # Serial reference
    print(f"\nLooking for 1d_serial reference ...")
    serial_ref = load_serial_timing(run_id, args.serial_ref)
    if serial_ref:
        print(f"  Found: {serial_ref['path']}")
        print(f"  Serial wall_time = {serial_ref['wall_s']:.2f} s  "
              f"mean_step = {serial_ref['step_ms_mean']:.3f} ms")
    else:
        print("  Not found.  Speedup column will be blank.")
        print("  (Run: python3 ../1d_serial/rupture_1d.py input/rupture_1d_SW.in)")

    # Print table
    print_table(results, serial_ref, p, args.nreps, warmup_s)

    # Save
    save_results(results, serial_ref, p, args.nreps, warmup_s, run_id)


if __name__ == '__main__':
    main()
