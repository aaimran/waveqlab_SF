#!/usr/bin/env python3
"""
inspect_npz.py — Validate and display contents of a rupture simulation .npz file
==================================================================================
Usage:
    python3 auxiliary/inspect_npz.py output/rupture_SW_8fe1f9ef.npz
    python3 auxiliary/inspect_npz.py output/rupture_SW_8fe1f9ef.npz --verbose
    python3 auxiliary/inspect_npz.py output/rupture_SW_8fe1f9ef.npz --check-only

Exit codes:
    0  — file is valid
    1  — file not found or unreadable
    2  — one or more validation checks failed
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Expected keys and their properties
# ---------------------------------------------------------------------------

# (ndim, dtype_kind)  kind: 'f'=float, 'U'=unicode, 'O'=object
EXPECTED_KEYS = {
    'DomainOutput_l': {'ndim': 3, 'dtype_kind': 'f'},
    'DomainOutput_r': {'ndim': 3, 'dtype_kind': 'f'},
    'y_l':            {'ndim': 1, 'dtype_kind': 'f'},
    'y_r':            {'ndim': 1, 'dtype_kind': 'f'},
    'time':           {'ndim': 1, 'dtype_kind': 'f'},
    'slip':           {'ndim': 1, 'dtype_kind': 'f'},
    'sliprate':       {'ndim': 1, 'dtype_kind': 'f'},
    'traction':       {'ndim': 1, 'dtype_kind': 'f'},
    'Tau_0':          {'ndim': 0, 'dtype_kind': 'f'},
    'metadata':       {'ndim': 0, 'dtype_kind': None},   # JSON string stored as 0-d array
}

# Physics sanity ranges
SANE_RANGES = {
    'cs':      (0.1,   15.0,   'km/s'),
    'rho':     (1.0,   10.0,   'g/cm³'),
    'L':       (0.1,   1000.0, 'km'),
    'nx':      (10,    100000, ''),
    'cfl':     (0.01,  1.0,    ''),
    'tend':    (0.001, 1000.0, 's'),
    'sigma_n': (0.1,   1000.0, 'MPa'),
    'D_c':     (1e-6,  100.0,  'm'),
    'alp_s':   (0.0,   2.0,    ''),
    'alp_d':   (0.0,   2.0,    ''),
    'a':       (0.0,   1.0,    ''),
    'b':       (0.0,   1.0,    ''),
    'V0':      (1e-12, 1.0,    'm/s'),
    'L0':      (1e-6,  100.0,  'm'),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(condition, msg, errors):
    if not condition:
        errors.append(msg)


def _warn(condition, msg, warnings):
    if not condition:
        warnings.append(msg)


def _fmt_shape(arr):
    return 'scalar' if arr.ndim == 0 else str(arr.shape)


def _fmt_size(nbytes):
    for unit, threshold in (('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if nbytes >= threshold:
            return f'{nbytes / threshold:.2f} {unit}'
    return f'{nbytes} B'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(data, errors, warnings):
    """Run all structural and physics checks. Populates errors / warnings lists."""

    # 1. Required keys present
    for key in EXPECTED_KEYS:
        _check(key in data, f"Missing required key: '{key}'", errors)

    if errors:
        return   # no point continuing without keys

    # 2. Array shapes and dtypes
    for key, spec in EXPECTED_KEYS.items():
        arr = data[key]
        if spec['ndim'] is not None:
            _check(arr.ndim == spec['ndim'],
                   f"'{key}': expected ndim={spec['ndim']}, got ndim={arr.ndim}", errors)
        if spec['dtype_kind'] is not None:
            _check(arr.dtype.kind == spec['dtype_kind'],
                   f"'{key}': expected dtype kind '{spec['dtype_kind']}', got '{arr.dtype}'", errors)

    # 3. Metadata JSON parseable
    meta = None
    try:
        meta = json.loads(str(data['metadata']))
    except (json.JSONDecodeError, TypeError) as exc:
        errors.append(f"'metadata' is not valid JSON: {exc}")
        return

    # 4. Dimensional consistency
    nx  = int(meta.get('nx', 0))
    nt  = int(meta.get('nt', 0))

    do_l = data['DomainOutput_l']
    do_r = data['DomainOutput_r']

    if nx > 0:
        _check(do_l.shape[0] == nx,
               f"DomainOutput_l axis-0={do_l.shape[0]} != nx={nx} from metadata", errors)
        _check(do_r.shape[0] == nx,
               f"DomainOutput_r axis-0={do_r.shape[0]} != nx={nx}", errors)
        _check(len(data['y_l']) == nx,
               f"y_l length={len(data['y_l'])} != nx={nx}", errors)
        _check(len(data['y_r']) == nx,
               f"y_r length={len(data['y_r'])} != nx={nx}", errors)

    if nt > 0:
        _check(do_l.shape[1] == nt + 1,
               f"DomainOutput_l axis-1={do_l.shape[1]} != nt+1={nt+1}", errors)
        _check(len(data['time']) == nt,
               f"time length={len(data['time'])} != nt={nt}", errors)
        _check(len(data['slip']) == nt + 1,
               f"slip length={len(data['slip'])} != nt+1={nt+1}", errors)
        _check(len(data['sliprate']) == nt,
               f"sliprate length={len(data['sliprate'])} != nt={nt}", errors)
        _check(len(data['traction']) == nt,
               f"traction length={len(data['traction'])} != nt={nt}", errors)

    _check(do_l.shape[2] == 2,
           f"DomainOutput_l axis-2={do_l.shape[2]} — expected 2 (velocity, stress)", errors)

    # 5. run_id present in metadata
    _check('run_id' in meta, "metadata missing 'run_id'", warnings)

    # 6. Physics sanity ranges
    for key, (lo, hi, unit) in SANE_RANGES.items():
        if key in meta:
            val = meta[key]
            try:
                _warn(lo <= float(val) <= hi,
                      f"metadata['{key}']={val} outside expected range [{lo}, {hi}] {unit}",
                      warnings)
            except (TypeError, ValueError):
                pass

    # 7. SW-specific: alp_d < alp_s
    if meta.get('fric_law') == 'SW':
        if 'alp_s' in meta and 'alp_d' in meta:
            _warn(float(meta['alp_d']) < float(meta['alp_s']),
                  f"SW: alp_d={meta['alp_d']} should be < alp_s={meta['alp_s']}", warnings)

    # 8. RS-specific: b > a
    if meta.get('fric_law') == 'RS':
        if 'a' in meta and 'b' in meta:
            _warn(float(meta['b']) > float(meta['a']),
                  f"RS: b={meta['b']} should be > a={meta['a']}", warnings)

    # 9. No NaN / Inf in field arrays
    for key in ('DomainOutput_l', 'DomainOutput_r', 'slip', 'sliprate', 'traction'):
        arr = data[key]
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        _check(nan_count == 0, f"'{key}' contains {nan_count} NaN values", errors)
        _check(inf_count == 0, f"'{key}' contains {inf_count} Inf values", errors)

    # 10. Time vector is monotonically increasing
    t = data['time']
    if len(t) > 1:
        _warn(bool(np.all(np.diff(t) > 0)),
              "time vector is not strictly monotonically increasing", warnings)

    # 11. Slip non-negative (slip ≥ 0 for positive rupture sense)
    _warn(float(data['slip'].min()) >= -1e-10,
          f"slip has negative values (min={float(data['slip'].min()):.4e})", warnings)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

SECTION_WIDTH = 60

def _section(title):
    print(f"\n{'─' * SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'─' * SECTION_WIDTH}")


def _row(label, value, unit=''):
    suffix = f'  {unit}' if unit else ''
    print(f"  {label:<28s}  {value}{suffix}")


def display(path, data, verbose):
    meta = json.loads(str(data['metadata']))

    # ── File info ────────────────────────────────────────────────────────────
    _section("FILE")
    _row("Path",    os.path.abspath(path))
    _row("Size",    _fmt_size(os.path.getsize(path)))
    _row("Run ID",  meta.get('run_id', 'N/A'))
    _row("Keys",    ', '.join(sorted(data.files)))

    # ── Array inventory ───────────────────────────────────────────────────────
    _section("ARRAYS")
    total_bytes = 0
    for key in sorted(data.files):
        arr = data[key]
        nbytes = arr.nbytes if hasattr(arr, 'nbytes') else 0
        total_bytes += nbytes
        print(f"  {key:<28s}  shape={_fmt_shape(arr):<22s}  "
              f"dtype={str(arr.dtype):<10s}  {_fmt_size(nbytes)}")
    print(f"\n  {'Total uncompressed':<28s}  {_fmt_size(total_bytes)}")

    # ── Simulation parameters ─────────────────────────────────────────────────
    _section("SIMULATION PARAMETERS")

    PARAM_GROUPS = [
        ("Run",         ['run_id', 'output_prefix', 'fric_law', 'wall_time_s']),
        ("Domain",      ['L', 'nx', 'dx']),
        ("Material",    ['cs', 'rho']),
        ("Time",        ['tend', 'dt', 'nt', 'cfl']),
        ("Scheme",      ['order']),
        ("BC (left)",   ['r0_l', 'r1_l', 'tau_11_l', 'tau_12_l', 'tau_21_l', 'tau_22_l']),
        ("BC (right)",  ['r0_r', 'r1_r', 'tau_11_r', 'tau_12_r', 'tau_21_r', 'tau_22_r']),
        ("Friction",    ['Tau_0', 'slip_init']),
        ("SW params",   ['alp_s', 'alp_d', 'D_c', 'sigma_n']),
        ("RS params",   ['f0', 'a', 'b', 'V0', 'L0', 'psi_init']),
    ]

    units = {
        'L': 'km', 'dx': 'km', 'cs': 'km/s', 'rho': 'g/cm³',
        'tend': 's', 'dt': 's', 'Tau_0': 'MPa', 'sigma_n': 'MPa',
        'D_c': 'm', 'V0': 'm/s', 'L0': 'm', 'wall_time_s': 's',
    }

    for group_name, keys in PARAM_GROUPS:
        found = [(k, meta[k]) for k in keys if k in meta]
        if not found:
            continue
        print(f"\n  [{group_name}]")
        for k, v in found:
            unit = units.get(k, '')
            if isinstance(v, float):
                vstr = f'{v:.6g}'
            else:
                vstr = str(v)
            _row(k, vstr, unit)

    if verbose:
        # Print any keys in metadata not covered by PARAM_GROUPS
        shown = {k for _, keys in PARAM_GROUPS for k in keys}
        shown.update({'friction_parameters', 'w_stride',
                      'mean_step_compute_ms', 'mean_step_total_ms'})
        extra = {k: v for k, v in meta.items() if k not in shown}
        if extra:
            print(f"\n  [Other metadata]")
            for k, v in sorted(extra.items()):
                print(f"  {k:<28s}  {v}")

    # ── On-fault statistics ───────────────────────────────────────────────────
    _section("ON-FAULT STATISTICS")
    slip  = data['slip']
    sr    = data['sliprate']
    trac  = data['traction']
    t     = data['time']

    _row("Final slip",     f"{float(slip[-1]):.4f}",          "m")
    _row("Peak slip rate", f"{float(sr.max()):.4f}",           "m/s")
    _row("Traction range", f"{float(trac.min()):.4f} – {float(trac.max()):.4f}", "MPa")
    _row("Tau_0",          f"{float(data['Tau_0']):.4f}",      "MPa")
    if len(t) > 0:
        _row("Time span",  f"0 – {float(t[-1]):.4f}",         "s")
        # Approximate rupture onset: when sliprate first exceeds 1e-4 m/s
        onset_idx = np.argmax(sr > 1e-4)
        if sr[onset_idx] > 1e-4:
            _row("Slip onset (~)",  f"{float(t[onset_idx]):.4f}", "s")

    # ── Domain field statistics ───────────────────────────────────────────────
    if verbose:
        _section("DOMAIN FIELD STATISTICS")
        for label, key in (('Left  (velocity)', 'DomainOutput_l'),
                           ('Right (velocity)', 'DomainOutput_r')):
            arr_v = data[key][:, :, 0]   # velocity channel
            arr_s = data[key][:, :, 1]   # stress channel
            print(f"\n  {label}")
            _row("  velocity  min/max",
                 f"{float(arr_v.min()):.4e} / {float(arr_v.max()):.4e}", "km/s")
            _row("  stress    min/max",
                 f"{float(arr_s.min()):.4e} / {float(arr_s.max()):.4e}", "MPa")

    print()


def display_timing(timing_path):
    """Display the companion _timing.npz if it exists."""
    try:
        td = np.load(timing_path, allow_pickle=True)
    except Exception as exc:
        print(f"  [WARN] Could not load timing file: {exc}")
        return

    tm = json.loads(str(td['metadata']))
    compute = td['step_time_compute']   # seconds per step, RK4 only
    total   = td['step_time_total']     # seconds per step, RK4 + output
    overhead = td['step_time_overhead'] # output-writing cost

    _section("TIMING")
    _row("Timing file",           os.path.basename(timing_path))
    _row("Wall time (total)",     f"{tm.get('wall_time_s', 0):.3f}",          "s")
    print()
    print(f"  {'Metric':<30s}  {'Compute only':>14s}  {'Compute+Output':>14s}  {'Output overhead':>15s}")
    print(f"  {'':─<30s}  {'':─>14s}  {'':─>14s}  {'':─>15s}")
    print(f"  {'Mean per step':<30s}  {compute.mean()*1e3:>13.3f}ms  "
          f"{total.mean()*1e3:>13.3f}ms  {overhead.mean()*1e3:>14.3f}ms")
    print(f"  {'Std per step':<30s}  {compute.std()*1e3:>13.3f}ms  "
          f"{total.std()*1e3:>13.3f}ms  {overhead.std()*1e3:>14.3f}ms")
    print(f"  {'Min per step':<30s}  {compute.min()*1e3:>13.3f}ms  "
          f"{total.min()*1e3:>13.3f}ms  {overhead.min()*1e3:>14.3f}ms")
    print(f"  {'Max per step':<30s}  {compute.max()*1e3:>13.3f}ms  "
          f"{total.max()*1e3:>13.3f}ms  {overhead.max()*1e3:>14.3f}ms")
    nt = len(compute)
    print(f"\n  Output-writing share: "
          f"{100*overhead.sum()/(total.sum() + 1e-300):.1f}% of total loop time  "
          f"({nt} steps)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate and inspect a rupture simulation .npz file')
    parser.add_argument('npzfile', help='path to .npz file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='show domain field statistics and extra metadata')
    parser.add_argument('--check-only', action='store_true',
                        help='run validation only, no display output')
    args = parser.parse_args()

    # Load
    if not os.path.isfile(args.npzfile):
        print(f"ERROR: File not found: {args.npzfile}")
        sys.exit(1)

    try:
        data = np.load(args.npzfile, allow_pickle=True)
    except Exception as exc:
        print(f"ERROR: Could not load {args.npzfile}: {exc}")
        sys.exit(1)

    # Validate
    errors, warnings = [], []
    validate(data, errors, warnings)

    # Report validation results
    ok = len(errors) == 0

    if not args.check_only:
        display(args.npzfile, data, args.verbose)

        # Auto-detect companion timing file  e.g. rupture_SW_8fe1f9ef_timing.npz
        base = args.npzfile
        if base.endswith('.npz') and not base.endswith('_timing.npz'):
            timing_path = base[:-4] + '_timing.npz'
            if os.path.isfile(timing_path):
                display_timing(timing_path)
            else:
                _section("TIMING")
                print(f"  No companion timing file found.")
                print(f"  (Re-run the simulation to generate {os.path.basename(timing_path)})")
                print()

    _section("VALIDATION SUMMARY")
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    [WARN]  {w}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    [FAIL]  {e}")
    if ok:
        print(f"  [PASS]  All {len(EXPECTED_KEYS)} required checks passed."
              + (f"  {len(warnings)} warning(s)." if warnings else ""))
    print()

    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
