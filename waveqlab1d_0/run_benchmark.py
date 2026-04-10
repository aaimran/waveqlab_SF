#!/usr/bin/env python3
"""
Benchmark runner for all waveqlab1d test cases (serial, 1 CPU).

Usage (after sourcing /work/aimran/wql1d/env.sh):
    cd /scratch/aimran/FNO/waveqlab_SF/waveqlab1d
    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_benchmark.py

Generates: benchmark_runtime.md
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable          # use the venv python that launched us
INPUT_DIR  = os.path.join(ROOT, 'input')
OUTPUT_DIR = os.path.join(ROOT, 'output')
OUT_MD     = os.path.join(ROOT, 'benchmark_runtime.md')

RESOLUTIONS = ['80m', '40m', '20m']
TESTS       = ['1a', '1b', '1c',
               '2a', '2b', '2c',
               '3a', '3b', '3c',
               '4a', '4b', '4c',
               '1r', '2r', '3r', '4r']

BC_LABEL = {'a': 'Free surface', 'b': 'Absorbing', 'c': 'PML 5 km', 'r': 'Free surface (ref 2×L)'}

# ---------------------------------------------------------------------------

def load_existing(test, res):
    """Return timing metadata from the most recent matching timing npz, or None."""
    import glob
    prefix  = f'test-{test}_{res}'
    pattern = os.path.join(OUTPUT_DIR, f'{prefix}_*_timing.npz')
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    data = np.load(files[-1], allow_pickle=True)
    meta = json.loads(str(data['metadata']))
    if all(k in meta for k in ('nx', 'nt', 'wall_time_s', 'warmup_time_s', 'mean_step_compute_ms')):
        return meta
    return None


def run_test(test, res):
    """Run one test; return timing dict or None on failure."""
    infile = f'input/test-{test}_{res}.in'
    inpath = os.path.join(ROOT, infile)

    if not os.path.exists(inpath):
        print(f'  SKIP (missing): {infile}')
        return None
    if os.path.getsize(inpath) == 0:
        print(f'  SKIP (empty):   {infile}')
        return None

    env = os.environ.copy()
    env['NUMBA_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS']   = '1'

    t_wall_start = time.monotonic()
    proc = subprocess.run(
        [PYTHON, 'rupture_1d.py', infile, '-np', '1'],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    t_wall_end = time.monotonic()
    elapsed = t_wall_end - t_wall_start

    stdout = proc.stdout
    # print last few lines for progress feedback
    tail = '\n'.join(stdout.strip().splitlines()[-6:])
    print(f'  subprocess wall time: {elapsed:.1f} s\n  {tail}\n')

    if proc.returncode != 0:
        print(f'  ERROR: non-zero exit code {proc.returncode}')
        return None

    # locate Run ID from stdout
    m = re.search(r'Run ID\s*:\s*(\S+)', stdout)
    if not m:
        print(f'  ERROR: Run ID not found in stdout')
        return None
    run_id = m.group(1)

    prefix      = f'test-{test}_{res}'
    timing_file = os.path.join(OUTPUT_DIR, f'{prefix}_{run_id}_timing.npz')
    if not os.path.exists(timing_file):
        print(f'  ERROR: timing file not found: {timing_file}')
        return None

    data = np.load(timing_file, allow_pickle=True)
    meta = json.loads(str(data['metadata']))
    return meta


def write_table(results):
    """Write benchmark_runtime.md from collected results, with aligned columns."""
    # Build raw cell data first
    COL_HEADS = ['Test', 'Resolution', 'BC', 'Total Grid Points', 'Total Time Steps',
                 'Total Work', 'Run Time (excl. JIT warmup)', 'JIT Warm-up Time', 'Avg. Run Time per step']
    COL_ALIGN = ['<', '^', '<', '>', '>', '>', '>', '>', '>']  # l/c/r per column

    data_rows = []
    for r in results:
        test = r['test']
        res  = r['res']
        bc   = BC_LABEL.get(test[-1], '?')
        if r.get('error'):
            data_rows.append([test, res, bc, '—', '—', '—', '—', '—', '—'])
        else:
            total_gp   = 2 * r['nx']
            total_work = total_gp * r['nt']
            data_rows.append([
                test,
                res,
                bc,
                f"{total_gp:,}",
                f"{r['nt']:,}",
                f"{total_work:,}",
                f"{r['wall_time_s']:.2f} s",
                f"{r['warmup_time_s']:.2f} s",
                f"{r['mean_step_compute_ms']:.3f} ms",
            ])

    # Compute column widths
    widths = [len(h) for h in COL_HEADS]
    for row in data_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_cell(text, width, align):
        if align == '>':
            return text.rjust(width)
        elif align == '^':
            return text.center(width)
        else:
            return text.ljust(width)

    def make_row(cells):
        return '| ' + ' | '.join(fmt_cell(c, widths[i], COL_ALIGN[i])
                                  for i, c in enumerate(cells)) + ' |'

    def make_sep():
        parts = []
        for i, w in enumerate(widths):
            if COL_ALIGN[i] == '>':
                parts.append('-' * (w - 1) + ':')
            elif COL_ALIGN[i] == '^':
                parts.append(':' + '-' * (w - 2) + ':')
            else:
                parts.append(':' + '-' * (w - 1))
        return '| ' + ' | '.join(parts) + ' |'

    lines = [
        '# waveqlab1d Runtime Benchmark',
        '',
        '**Platform:** serial (1 CPU), `NUMBA_NUM_THREADS=1`  ',
        '**Solver:** waveqlab1d (Numba, order-6 SBP-SAT)  ',
        '',
        make_row(COL_HEADS),
        make_sep(),
    ]
    for row in data_rows:
        lines.append(make_row(row))

    md = '\n'.join(lines) + '\n'
    with open(OUT_MD, 'w') as f:
        f.write(md)
    print(f'\nBenchmark table written → {OUT_MD}')
    print(md)


# ---------------------------------------------------------------------------

def main():
    results = []
    total = len(TESTS) * len(RESOLUTIONS)
    idx   = 0

    for test in TESTS:
        for res in RESOLUTIONS:
            idx += 1
            label = f'test-{test}_{res}'
            print(f'[{idx:3d}/{total}] {label}')
            meta = load_existing(test, res)
            if meta is not None:
                print(f'  CACHED: reusing existing timing file')
            else:
                meta = run_test(test, res)
            if meta is None:
                results.append({'test': f'test-{test}', 'res': res, 'error': True})
            else:
                results.append({
                    'test':              f'test-{test}',
                    'res':               res,
                    'nx':                meta['nx'],
                    'nt':                meta['nt'],
                    'wall_time_s':       meta['wall_time_s'],
                    'warmup_time_s':     meta['warmup_time_s'],
                    'mean_step_compute_ms': meta['mean_step_compute_ms'],
                    'error':             False,
                })
            # Incrementally update the table after each run
            write_table(results)

    print('\nAll done.')


if __name__ == '__main__':
    main()
