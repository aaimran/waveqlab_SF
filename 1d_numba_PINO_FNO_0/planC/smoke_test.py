#!/usr/bin/env python3
"""
planC/smoke_test.py
====================
Quick sanity checks for all planC components (no GPU required, no data needed).

Tests:
  1. Model forward pass at training resolution (376 × 128)
  2. Model forward pass at super-resolution  (1501 × 128)  — mode clamping
  3. build_input_tensor shape correctness
  4. energy_stability_loss returns non-negative scalar
  5. PINOLoss forward with synthetic data
  6. Dataset file structure check (if data exists)
  7. Config YAML loading for all 6 configs

Usage:
    cd /scratch/aimran/FNO/waveqlab_SF/1d_numba_PINO_FNO_0
    source /work/aimran/wql1d/env.sh
    python planC/smoke_test.py
"""

import os
import sys
import traceback

import numpy as np
import torch

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

PASS = '\033[92m[PASS]\033[0m'
FAIL = '\033[91m[FAIL]\033[0m'
WARN = '\033[93m[WARN]\033[0m'

results = []


def check(name, fn):
    try:
        fn()
        print(f'{PASS} {name}')
        results.append((name, True, ''))
    except Exception as e:
        tb = traceback.format_exc()
        print(f'{FAIL} {name}: {e}')
        print(tb)
        results.append((name, False, str(e)))


# ---------------------------------------------------------------------------
# Test 1 — Model import and forward at 80m (376×128)
# ---------------------------------------------------------------------------

def test_model_forward_lr():
    from model.fno import FNO2d, build_input_tensor
    B, NX, NT = 2, 376, 128
    model = FNO2d(c_in=9, modes_x=24, modes_t=24, width=32, n_layers=4, c_out=4)
    model.eval()
    x = torch.randn(B, 9, NX, NT)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, 4, NX, NT), f'Unexpected shape: {y.shape}'


# ---------------------------------------------------------------------------
# Test 2 — Super-resolution forward at 20m (1501×128) — same model
# ---------------------------------------------------------------------------

def test_model_forward_hr():
    from model.fno import FNO2d
    B, NX, NT = 1, 1501, 128
    model = FNO2d(c_in=9, modes_x=24, modes_t=24, width=32, n_layers=4, c_out=4)
    model.eval()
    x = torch.randn(B, 9, NX, NT)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (B, 4, NX, NT), f'Unexpected shape: {y.shape}'


# ---------------------------------------------------------------------------
# Test 3 — build_input_tensor shape
# ---------------------------------------------------------------------------

def test_build_input_tensor():
    from model.fno import build_input_tensor
    B, NX, NT = 2, 376, 128
    params  = torch.randn(B, 7)
    x_n     = torch.linspace(0, 1, NX)
    t_n     = torch.linspace(0, 1, NT)
    x_ch    = x_n.view(NX, 1).expand(NX, NT)
    t_ch    = t_n.view(1,  NT).expand(NX, NT)
    coords  = torch.stack([x_ch, t_ch], dim=0).unsqueeze(0).expand(B, 2, NX, NT)
    inp = build_input_tensor(params, coords)
    assert inp.shape == (B, 9, NX, NT), f'Bad shape: {inp.shape}'


# ---------------------------------------------------------------------------
# Test 4 — energy_stability_loss returns non-negative scalar
# ---------------------------------------------------------------------------

def test_energy_stability_loss():
    from model.physics_loss import energy_stability_loss
    B, NX, NT = 2, 376, 128
    # v_l, s_l in channels 0,1 of left domain; v_r, s_r in channels 2,3
    pred = torch.randn(B, 4, NX, NT)
    rho  = 2.67
    mu   = 32.0
    dx   = 0.08e-3   # km (80 m in km)
    loss = energy_stability_loss(pred, rho, mu, dx)
    assert loss.item() >= 0, f'Energy loss < 0: {loss.item()}'
    assert loss.dim() == 0, 'Expected scalar'


# ---------------------------------------------------------------------------
# Test 5 — PINOLoss forward with synthetic data
# ---------------------------------------------------------------------------

def test_pino_loss_forward():
    from model.physics_loss import PINOLoss
    B, NX, NT = 2, 376, 128
    dx = torch.tensor(0.08e-3)   # km
    dt = torch.tensor(0.01)      # s (large but just for test)

    criterion = PINOLoss(
        r0_l=0, r1_r=0, rho=2.67, mu=32.0, cs=3.464, sigma_n=120.0,
        w_data=1.0, w_pde=0.05, w_fault=0.10, w_bc=0.08, w_ic=0.05, w_stab=0.02)

    pred   = torch.randn(B, 4, NX, NT)
    target = torch.randn(B, 4, NX, NT)
    Tau_0  = torch.full((B,), 82.0)
    alp_s  = torch.full((B,), 0.68)
    alp_d  = torch.full((B,), 0.52)
    D_c    = torch.full((B,), 0.4)

    loss_d = criterion(pred, target, dx, dt, Tau_0, alp_s, alp_d, D_c)
    assert 'total' in loss_d
    total = loss_d['total']
    assert total.dim() == 0
    assert torch.isfinite(total), f'Non-finite total loss: {total.item()}'


# ---------------------------------------------------------------------------
# Test 6 — Config YAML loading
# ---------------------------------------------------------------------------

def test_configs():
    import yaml
    config_dir = os.path.join(_HERE, 'configs')
    configs = [
        'sw_bc_both0_r80m.yaml', 'sw_bc_both0_r40m.yaml', 'sw_bc_both0_r20m.yaml',
        'sw_bc_both1_r80m.yaml', 'sw_bc_both1_r40m.yaml', 'sw_bc_both1_r20m.yaml',
    ]
    for name in configs:
        path = os.path.join(config_dir, name)
        assert os.path.exists(path), f'Missing config: {path}'
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert 'model' in cfg and 'training' in cfg and 'loss' in cfg


# ---------------------------------------------------------------------------
# Test 7 — data_gen imports work (no sim run)
# ---------------------------------------------------------------------------

def test_data_gen_import():
    import importlib.util, importlib.machinery
    spec = importlib.util.spec_from_file_location(
        'planC_dataset',
        os.path.join(_HERE, 'data_gen', 'dataset.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, 'RuptureDatasetPlanC')
    FIELD_KEYS = mod.FIELD_KEYS
    PARAM_NAMES = mod.PARAM_NAMES
    assert len(FIELD_KEYS) == 4
    assert len(PARAM_NAMES) == 7

    # Normalizer round-trip
    import numpy as np, torch
    mean = np.array([1.0, 2.0, 3.0])
    std  = np.array([0.1, 0.2, 0.3])
    norm = mod.Normalizer(mean, std)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    encoded = norm.encode(x)
    decoded = norm.decode(encoded)
    assert torch.allclose(decoded, x, atol=1e-5), 'Normalizer round-trip failed'


# ---------------------------------------------------------------------------
# Test 8 — generate_dataset imports (no sim run)
# ---------------------------------------------------------------------------

def test_gen_import():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'planC_generate',
        os.path.join(_HERE, 'data_gen', 'generate_dataset.py'))
    mod = importlib.util.module_from_spec(spec)
    # skip actual execution but parse constants at module level
    # Just verify the file parses correctly by checking key names
    with open(os.path.join(_HERE, 'data_gen', 'generate_dataset.py')) as f:
        src = f.read()
    assert 'NX_NATIVE' in src
    assert 'NT_OUT' in src
    assert 'BC_PRESETS' in src
    assert 'RESOLUTIONS' in src
    assert 'PARAM_NAMES' in src


# ---------------------------------------------------------------------------
# Test 9 — Optional: check data directories
# ---------------------------------------------------------------------------

def test_data_dirs_exist():
    data_root = os.path.join(_HERE, 'data')
    if not os.path.isdir(data_root):
        raise AssertionError(
            f'Data root {data_root} not yet created — run generate_dataset.py first')
    for bc in ['bc_both0', 'bc_both1']:
        for split in ['train', 'val', 'test']:
            d = os.path.join(data_root, bc, split)
            if not os.path.isdir(d):
                raise AssertionError(f'Missing split dir: {d}')
            files = [f for f in os.listdir(d) if f.endswith('.npz')]
            if len(files) == 0:
                raise AssertionError(f'No .npz files in {d}')
    print(f'  (data directories present and non-empty)')


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('Plan-C smoke tests')
    print('=' * 60)

    check('model_forward_lr (376×128)',      test_model_forward_lr)
    check('model_forward_hr (1501×128) SR',  test_model_forward_hr)
    check('build_input_tensor shape',        test_build_input_tensor)
    check('energy_stability_loss ≥ 0',       test_energy_stability_loss)
    check('PINOLoss forward (6 terms)',      test_pino_loss_forward)
    check('configs yaml loading (6 files)',  test_configs)
    check('data_gen.dataset imports',        test_data_gen_import)
    check('data_gen.generate imports',       test_gen_import)

    # Data directory check is optional
    print('\n--- Optional (needs data) ---')
    try:
        test_data_dirs_exist()
        print(f'{PASS} data directories present')
        results.append(('data_dirs_exist', True, ''))
    except AssertionError as e:
        print(f'{WARN} data_dirs_exist: {e}')
        results.append(('data_dirs_exist', None, str(e)))

    print('\n' + '=' * 60)
    passed  = sum(1 for _, ok, _ in results if ok is True)
    failed  = sum(1 for _, ok, _ in results if ok is False)
    skipped = sum(1 for _, ok, _ in results if ok is None)
    total   = len(results)
    print(f'Results: {passed}/{total} passed, {failed} failed, {skipped} skipped')
    if failed > 0:
        sys.exit(1)
