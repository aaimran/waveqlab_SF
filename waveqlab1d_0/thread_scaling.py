#!/usr/bin/env python3
"""
Thread scaling sweep for test-1a at each resolution.

Runs test-1a from coarsest (1m) to finest (80m), sweeping
NUMBA_NUM_THREADS over [1, 2, 4, 8, 16, 20, 32, 40].
Finds the optimal (fastest) thread count per resolution,
prints a markdown table, and writes optimal_threads.json
for use by the full benchmark runner.

Usage (inside idev31 session, env loaded):
    cd /scratch/aimran/FNO/waveqlab_SF/waveqlab1d
    python thread_scaling.py
"""

import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT       = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable
OUTPUT_DIR = os.path.join(ROOT, 'output')
OUT_JSON   = os.path.join(ROOT, 'optimal_threads.json')
LOG        = os.path.join(ROOT, 'thread_scaling.log')

# Resolutions ordered largest grid first (hardest → most benefit from threads)
RESOLUTIONS = ['1m', '5m', '10m', '20m', '40m', '80m']
THREAD_COUNTS = [1, 2, 4, 8, 16, 20, 32, 40]

# Per-resolution thread count overrides
THREAD_COUNTS_OVERRIDE = {
    '1m': [32, 16, 8],
}

TEST = '1a'


def run_once(res, nthreads):
    infile = f'input/test-{TEST}_{res}.in'
    env = os.environ.copy()
    env['NUMBA_NUM_THREADS'] = str(nthreads)
    env['OMP_NUM_THREADS']   = str(nthreads)

    proc = subprocess.run(
        [PYTHON, 'rupture_1d.py', infile, '-np', str(nthreads)],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    stdout = proc.stdout

    # append to log
    with open(LOG, 'a') as f:
        f.write(f'\n=== {res}, threads={nthreads} ===\n')
        f.write(stdout)

    if proc.returncode != 0:
        print(f'    ERROR (exit {proc.returncode})')
        return None

    m = re.search(r'Run ID\s*:\s*(\S+)', stdout)
    if not m:
        print(f'    ERROR: Run ID not found')
        return None
    run_id = m.group(1)

    timing_file = os.path.join(OUTPUT_DIR, f'test-{TEST}_{res}_{run_id}_timing.npz')
    if not os.path.exists(timing_file):
        print(f'    ERROR: timing file missing: {timing_file}')
        return None

    data = np.load(timing_file, allow_pickle=True)
    meta = json.loads(str(data['metadata']))
    return meta


def main():
    # Clear log
    open(LOG, 'w').close()

    results = {}   # res -> list of (nthreads, wall_s, step_ms)
    optimal = {}   # res -> nthreads

    header = [
        '# Thread Scaling Results — test-1a',
        '',
        '| Resolution | Threads | Wall Time (excl. JIT) | Avg. Step Time | Speedup vs 1T |',
        '|:----------:|--------:|----------------------:|---------------:|:-------------:|',
    ]
    rows = []

    for res in RESOLUTIONS:
        print(f'\n--- {res} ---')
        res_results = []

        thread_list = THREAD_COUNTS_OVERRIDE.get(res, THREAD_COUNTS)
        for nt in thread_list:
            print(f'  threads={nt:2d} ...', end=' ', flush=True)
            meta = run_once(res, nt)
            if meta is None:
                res_results.append((nt, None, None))
                continue
            wall = meta['wall_time_s']
            step = meta['mean_step_compute_ms']
            print(f'wall={wall:.2f}s  step={step:.3f}ms')
            res_results.append((nt, wall, step))

        results[res] = res_results

        # find optimal: minimum wall time among successful runs
        valid = [(nt, w, s) for nt, w, s in res_results if w is not None]
        if not valid:
            optimal[res] = 1
            continue

        best_nt, best_wall, best_step = min(valid, key=lambda x: x[1])
        optimal[res] = best_nt
        print(f'  => optimal: {best_nt} threads  (wall={best_wall:.2f}s)')

        # baseline for speedup
        baseline = next((w for nt, w, s in valid if nt == 1), None)

        for nt, wall, step in valid:
            speedup = f'{baseline/wall:.2f}x' if (baseline and wall) else '—'
            marker  = ' ◄ optimal' if nt == best_nt else ''
            rows.append(
                f'| {res} | {nt} | {wall:.2f} s | {step:.3f} ms | {speedup}{marker} |'
            )

    # Write markdown table
    md_path = os.path.join(ROOT, 'thread_scaling.md')
    md = '\n'.join(header + rows) + '\n'
    with open(md_path, 'w') as f:
        f.write(md)

    # Write optimal_threads.json
    with open(OUT_JSON, 'w') as f:
        json.dump(optimal, f, indent=2)

    print('\n' + '='*60)
    print(md)
    print(f'Optimal threads written → {OUT_JSON}')
    print(f'Full table    written  → {md_path}')
    print(f'Full log      written  → {LOG}')
    print('\nOptimal thread counts:')
    for res, nt in optimal.items():
        print(f'  {res:5s}: {nt} threads')


if __name__ == '__main__':
    main()
