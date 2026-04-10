"""
param_space.py — Parameter bounds and sampling for Plans A and B
================================================================

Defines Latin Hypercube parameter ranges for every (fric_law × bc_type)
combination. Used by generate_dataset.py for both plans.

All bounds are (low, high) tuples. Categorical parameters use list of values.
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube

# ─── Shared material ranges ───────────────────────────────────────────────────

MATERIAL_BOUNDS = {
    'cs'         : (2.8,  4.0),       # km/s   shear wave velocity
    'rho'        : (2.4,  3.0),       # g/cm³  density
}

# Q model: c=0 means elastic (no attenuation); c>0 → Q_S = c * V_S
# weight_exp: 0.0 = constant Q, 0.6 = power-law Q
ANELASTIC_BOUNDS = {
    'c'          : (8.0,  25.0),      # s/km   Q_S = c * V_S
}
ANELASTIC_CATEGORICAL = {
    'elastic_frac': 0.30,             # fraction of samples with elastic response
    'weight_exp'  : [0.0, 0.6],       # sampled uniformly
}

# ─── SW parameter ranges ──────────────────────────────────────────────────────

SW_BOUNDS = {
    'Tau_0'      : (78.0, 88.0),      # MPa
    'sigma_n'    : (100., 140.),      # MPa
    'alp_s'      : (0.62, 0.74),
    'alp_d'      : (0.45, 0.55),      # constrained < alp_s during sampling
    'D_c'        : (0.2,  0.8),       # m
}

# ─── RS parameter ranges ──────────────────────────────────────────────────────

RS_BOUNDS = {
    'Tau_0'      : (78.0, 88.0),      # MPa
    'sigma_n'    : (100., 140.),      # MPa
    'f0'         : (0.5,  0.7),
    'a'          : (0.005, 0.015),
    'b'          : (0.015, 0.025),    # constrained > a during sampling
    'V0'         : (-7.0, -5.0),      # log10(V0), mapped: 10^x
    'L0'         : (0.01, 0.05),      # m
    'psi_init'   : (0.35, 0.50),
}

# ─── BC modes for each model ──────────────────────────────────────────────────

# Maps model_key → (r0_l, r1_r, pml, bc_label)
BC_CONFIG = {
    'free'      : dict(r0_l=1,  r1_r=1,  pml=False, npml=0,  pml_alpha=0.0),
    'absorbing' : dict(r0_l=0,  r1_r=0,  pml=False, npml=0,  pml_alpha=0.0),
    'pml'       : dict(r0_l=1,  r1_r=1,  pml=True,  npml=20, pml_alpha=10.0),
}

# ─── Combined model keys ──────────────────────────────────────────────────────

# All 6 Plan-B models
PLAN_B_MODELS = [
    'sw_free', 'sw_absorbing', 'sw_pml',
    'rs_free', 'rs_absorbing', 'rs_pml',
]

def parse_model_key(model_key: str):
    """'sw_free' → (fric_law='SW', bc_mode='free')."""
    parts = model_key.lower().split('_', 1)
    return parts[0].upper(), parts[1]


# ─── Sampling ─────────────────────────────────────────────────────────────────

def _lhs_bounds(bounds_dict: dict, n: int, rng: np.random.Generator) -> dict:
    """Latin-hypercube draw from a dict of (low, high) bounds."""
    keys = list(bounds_dict.keys())
    lows  = np.array([bounds_dict[k][0] for k in keys])
    highs = np.array([bounds_dict[k][1] for k in keys])
    sampler = LatinHypercube(d=len(keys), seed=int(rng.integers(0, 2**31)))
    unit    = sampler.random(n)                         # (n, d) in [0,1]
    samples = lows + unit * (highs - lows)              # (n, d)
    return {k: samples[:, i] for i, k in enumerate(keys)}


def sample_sw(n: int, seed: int = 0) -> list[dict]:
    """
    Generate n SW parameter dictionaries via LHS.
    Enforces alp_d < alp_s (rejects and resamples violating draws).
    """
    rng    = np.random.default_rng(seed)
    result = []
    while len(result) < n:
        need = n - len(result)
        s    = _lhs_bounds(SW_BOUNDS, max(need * 2, 50), rng)
        mat  = _lhs_bounds(MATERIAL_BOUNDS, max(need * 2, 50), rng)
        # Anelastic: fraction elastic, rest sample c uniformly
        c_vals      = _lhs_bounds({'c': ANELASTIC_BOUNDS['c']}, max(need*2, 50), rng)['c']
        elastic_mask = rng.random(len(c_vals)) < ANELASTIC_CATEGORICAL['elastic_frac']
        c_final     = np.where(elastic_mask, 0.0, c_vals)
        w_exp_idx   = rng.integers(0, 2, size=len(c_vals))
        w_exp_vals  = np.array(ANELASTIC_CATEGORICAL['weight_exp'])[w_exp_idx]

        for i in range(s['Tau_0'].shape[0]):
            if s['alp_d'][i] >= s['alp_s'][i]:
                continue
            p = {k: float(s[k][i]) for k in SW_BOUNDS}
            p.update({k: float(mat[k][i]) for k in MATERIAL_BOUNDS})
            p['c']          = float(c_final[i])
            p['weight_exp'] = float(w_exp_vals[i])
            p['response']   = 'elastic' if elastic_mask[i] else 'anelastic'
            result.append(p)
            if len(result) >= n:
                break
    return result[:n]


def sample_rs(n: int, seed: int = 0) -> list[dict]:
    """
    Generate n RS parameter dictionaries via LHS.
    Enforces b > a (rejects violating draws).
    Maps log10(V0) → V0.
    """
    rng    = np.random.default_rng(seed)
    result = []
    while len(result) < n:
        need = n - len(result)
        s    = _lhs_bounds(RS_BOUNDS, max(need * 2, 50), rng)
        mat  = _lhs_bounds(MATERIAL_BOUNDS, max(need * 2, 50), rng)
        c_vals      = _lhs_bounds({'c': ANELASTIC_BOUNDS['c']}, max(need*2, 50), rng)['c']
        elastic_mask = rng.random(len(c_vals)) < ANELASTIC_CATEGORICAL['elastic_frac']
        c_final     = np.where(elastic_mask, 0.0, c_vals)
        w_exp_idx   = rng.integers(0, 2, size=len(c_vals))
        w_exp_vals  = np.array(ANELASTIC_CATEGORICAL['weight_exp'])[w_exp_idx]

        for i in range(s['Tau_0'].shape[0]):
            if s['b'][i] <= s['a'][i]:
                continue
            p = {k: float(s[k][i]) for k in RS_BOUNDS if k != 'V0'}
            p['V0']         = float(10 ** s['V0'][i])
            p.update({k: float(mat[k][i]) for k in MATERIAL_BOUNDS})
            p['c']          = float(c_final[i])
            p['weight_exp'] = float(w_exp_vals[i])
            p['response']   = 'elastic' if elastic_mask[i] else 'anelastic'
            result.append(p)
            if len(result) >= n:
                break
    return result[:n]


def sample_for_model(model_key: str, n: int, seed: int = 0) -> list[dict]:
    """
    Sample n parameter sets for a given Plan-B model key (e.g. 'sw_free').
    Adds BC config to each parameter dict.
    """
    fric_law, bc_mode = parse_model_key(model_key)
    bc_cfg = BC_CONFIG[bc_mode]

    if fric_law == 'SW':
        samples = sample_sw(n, seed=seed)
    else:
        samples = sample_rs(n, seed=seed)

    for p in samples:
        p.update(bc_cfg)
        p['fric_law'] = fric_law
        p['bc_mode']  = bc_mode

    return samples


def sample_unified(n: int, seed: int = 0) -> list[dict]:
    """
    Sample n parameter sets across all 6 conditions (Plan A).
    Balanced: n//6 per condition.
    """
    per_model = max(1, n // len(PLAN_B_MODELS))
    result = []
    for i, key in enumerate(PLAN_B_MODELS):
        result.extend(sample_for_model(key, per_model, seed=seed + i * 1000))
    return result[:n]
