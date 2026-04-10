#!/usr/bin/env python3
"""Visual runtime comparison from benchmark_runtime data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Data from benchmark_runtime.md
# (test, res, wall_s, step_ms)
RAW = [
    ('1a','80m',0.07,0.097), ('1a','40m',0.27,0.182), ('1a','20m',1.06,0.353),
    ('1b','80m',0.07,0.097), ('1b','40m',0.27,0.182), ('1b','20m',1.05,0.352),
    ('1c','80m',0.07,0.099), ('1c','40m',0.27,0.182), ('1c','20m',1.06,0.353),
    ('2a','80m',0.10,0.141), ('2a','40m',0.39,0.265), ('2a','20m',1.58,0.525),
    ('2b','80m',0.10,0.141), ('2b','40m',0.39,0.265), ('2b','20m',1.59,0.526),
    ('2c','80m',0.10,0.141), ('2c','40m',0.38,0.264), ('2c','20m',1.60,0.530),
    ('3a','80m',0.11,0.145), ('3a','40m',0.38,0.263), ('3a','20m',1.59,0.531),
    ('3b','80m',0.10,0.141), ('3b','40m',0.38,0.263), ('3b','20m',1.59,0.528),
    ('3c','80m',0.10,0.142), ('3c','40m',0.38,0.264), ('3c','20m',1.60,0.531),
    ('4a','80m',0.15,0.141), ('4a','40m',0.58,0.262), ('4a','20m',2.34,0.518),
    ('4b','80m',0.16,0.143), ('4b','40m',0.57,0.261), ('4b','20m',2.35,0.519),
    ('4c','80m',0.16,0.142), ('4c','40m',0.57,0.262), ('4c','20m',2.34,0.518),
    ('1r','80m',0.14,0.185), ('1r','40m',0.53,0.362), ('1r','20m',2.20,0.710),
    ('2r','80m',0.19,0.269), ('2r','40m',0.83,0.563), ('2r','20m',3.32,1.103),
    ('3r','80m',0.19,0.271), ('3r','40m',0.82,0.559), ('3r','20m',3.33,1.108),
    ('4r','80m',0.29,0.266), ('4r','40m',1.22,0.547), ('4r','20m',4.81,1.069),
]

RESOLUTIONS = ['80m', '40m', '20m']
VARIANTS    = ['a', 'b', 'c', 'r']
VAR_LABELS  = {'a': 'Free surface', 'b': 'Absorbing', 'c': 'PML 5 km', 'r': 'Ref (2×L)'}
VAR_COLORS  = {'a': '#4C72B0', 'b': '#55A868', 'c': '#C44E52', 'r': '#8172B2'}
RES_MARKERS = {'80m': 'o', '40m': 's', '20m': '^'}

# Index: (test_id, res) -> (wall_s, step_ms)
data = {}
for (t, r, w, s) in RAW:
    data[(t, r)] = (w, s)

# ---------------------------------------------------------------------------
# Figure 1: Wall time by test-number — one panel per test (1-4)
# x-axis: BC variant, grouped bars per resolution
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
fig.suptitle('Total Wall Time (excl. JIT warmup) — by Test & BC', fontsize=13, y=1.01)

x = np.arange(len(VARIANTS))
width = 0.22
offsets = np.array([-1, 0, 1]) * width

res_colors = {'80m': '#a8c4e0', '40m': '#4C72B0', '20m': '#1a3a6e'}

TITLES = {
    '1': 'Elastic\n(homogeneous)',
    '2': 'Anelastic\nConstant Q  (γ=0)',
    '3': 'Anelastic\nPower-law Q  (γ=0.6)',
    '4': 'Anelastic, Power-law Q\nLayered Model  (γ=0.6)',
}

for ti, tnum in enumerate(['1', '2', '3', '4']):
    ax = axes[ti]
    for ri, res in enumerate(RESOLUTIONS):
        vals = [data.get((f'{tnum}{v}', res), (0, 0))[0] for v in VARIANTS]
        bars = ax.bar(x + offsets[ri], vals, width, label=res,
                      color=res_colors[res], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_title(TITLES[tnum], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Wall Time (s)' if ti == 0 else '')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    if ti == 0:
        ax.legend(title='Resolution', fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.savefig('plots/wall_time_by_test.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plots/wall_time_by_test.png')

# ---------------------------------------------------------------------------
# Figure 2: Avg step time vs resolution — line plot, one line per test+variant
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: group by test number (colours), variant as linestyle
FOR_LINE = [('1','a'), ('1','b'), ('1','c'), ('1','r'),
            ('2','a'), ('3','a'), ('4','a')]
cmap = plt.get_cmap('tab10')
tnum_colors = {'1': cmap(0), '2': cmap(1), '3': cmap(2), '4': cmap(3)}
var_ls = {'a': '-', 'b': '--', 'c': ':', 'r': '-.'}

ax = axes[0]
for tnum in ['1', '2', '3', '4']:
    for var in VARIANTS:
        key = f'{tnum}{var}'
        ys = [data.get((key, r), (None, None))[1] for r in RESOLUTIONS]
        if any(v is not None for v in ys):
            ax.plot(RESOLUTIONS, ys,
                    color=tnum_colors[tnum], linestyle=var_ls[var],
                    marker=RES_MARKERS.get('80m', 'o'), markersize=5,
                    label=f'test-{key}')
ax.set_title('Avg. Step Time — all variants', fontsize=11)
ax.set_xlabel('Resolution')
ax.set_ylabel('Avg. step time (ms)')
ax.grid(linestyle='--', alpha=0.4)
ax.legend(fontsize=6.5, ncol=2, loc='upper left')

# Right: compare a vs b vs c vs r for each test (cleaner)
ax = axes[1]
for tnum in ['1', '2', '3', '4']:
    for var in VARIANTS:
        key = f'{tnum}{var}'
        ys = [data.get((key, r), (None, None))[1] for r in RESOLUTIONS]
        if any(v is not None for v in ys):
            ax.plot(RESOLUTIONS, ys,
                    color=VAR_COLORS[var], linestyle=['-','--',':','-.'][['1','2','3','4'].index(tnum)],
                    marker='o', markersize=5,
                    label=f'test-{key}' if tnum == '1' else '_nolegend_')

# legend patches for BC type
from matplotlib.lines import Line2D
legend_bc   = [Line2D([0],[0], color=VAR_COLORS[v], lw=2, label=VAR_LABELS[v]) for v in VARIANTS]
legend_test = [Line2D([0],[0], color='k', lw=2, ls=ls, label=f'Test {t}')
               for t, ls in zip(['1','2','3','4'],['-','--',':','-.'])]
ax.legend(handles=legend_bc + legend_test, fontsize=7, ncol=2)
ax.set_title('Avg. Step Time — BC & test comparison', fontsize=11)
ax.set_xlabel('Resolution')
ax.set_ylabel('Avg. step time (ms)')
ax.grid(linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('plots/step_time_lines.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plots/step_time_lines.png')

# ---------------------------------------------------------------------------
# Figure 3: elastic vs anelastic overhead — 1a vs 2a vs 3a vs 4a, wall time
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for var in ['a', 'b', 'c', 'r']:
    for tnum in ['1', '2', '3', '4']:
        key = f'{tnum}{var}'
        xs = np.arange(len(RESOLUTIONS))
        ys = [data.get((key, r), (0, 0))[0] for r in RESOLUTIONS]
        ax.plot(xs, ys,
                color=tnum_colors[tnum], linestyle=var_ls[var],
                marker='o', markersize=5)

from matplotlib.patches import Patch
legend1 = [Patch(color=tnum_colors[t], label=f'Test {t}') for t in ['1','2','3','4']]
legend2 = [Line2D([0],[0], color='k', ls=var_ls[v], lw=2, label=VAR_LABELS[v]) for v in VARIANTS]
ax.legend(handles=legend1+legend2, fontsize=8, ncol=2)
ax.set_xticks(np.arange(len(RESOLUTIONS)))
ax.set_xticklabels(RESOLUTIONS)
ax.set_xlabel('Resolution')
ax.set_ylabel('Wall Time (s)')
ax.set_title('Wall Time — All Tests & BC types', fontsize=12)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('plots/wall_time_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plots/wall_time_overview.png')

print('\nAll plots saved to plots/')

# ---------------------------------------------------------------------------
# Figure 4: Grouped by Resolution — one panel per resolution
# x-axis: BC variant (a/b/c/r), grouped bars = test types
# ---------------------------------------------------------------------------
TEST_LABELS = {
    '1': 'Elastic',
    '2': 'Anelastic\nConst. Q (γ=0)',
    '3': 'Anelastic\nPower-law Q (γ=0.6)',
    '4': 'Anelastic, Layered\nPower-law Q (γ=0.6)',
}
TEST_COLORS = {'1': '#4C72B0', '2': '#DD8452', '3': '#55A868', '4': '#C44E52'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle('Total Wall Time (excl. JIT warmup) — by Resolution & BC', fontsize=13, y=1.01)

x = np.arange(len(VARIANTS))
n_tests = 4
width = 0.18
offsets = (np.arange(n_tests) - (n_tests - 1) / 2) * width

for ri, res in enumerate(RESOLUTIONS):
    ax = axes[ri]
    for ti, tnum in enumerate(['1', '2', '3', '4']):
        vals = [data.get((f'{tnum}{v}', res), (0, 0))[0] for v in VARIANTS]
        bars = ax.bar(x + offsets[ti], vals, width, label=TEST_LABELS[tnum],
                      color=TEST_COLORS[tnum], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_title(f'Resolution: {res}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([VAR_LABELS[v] for v in VARIANTS], fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Wall Time (s)' if ri == 0 else '')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    if ri == 0:
        ax.legend(title='Test type', fontsize=7.5, title_fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('plots/wall_time_by_resolution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plots/wall_time_by_resolution.png')

