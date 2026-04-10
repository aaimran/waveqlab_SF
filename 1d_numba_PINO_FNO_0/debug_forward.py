#!/usr/bin/env python3
"""Quick debug: single forward pass to check shapes and loss computation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from data_gen.dataset import RuptureDataset, fit_normalizers
from model.fno import FNO2d
from model.physics_loss import relative_l2_loss

cfg = yaml.safe_load(open('configs/fno_sw_baseline.yaml'))
param_norm, field_norms = fit_normalizers('data/train')
ds = RuptureDataset('data/train', param_norm, field_norms)

params, fields, coords = ds[0]
print('params:', params.shape, params.dtype)
print('fields:', fields.shape, fields.dtype)
print('coords:', coords.shape, coords.dtype)

model = FNO2d(n_params=4, modes_x=16, modes_t=16, width=64, n_layers=4)

params_b = params.unsqueeze(0)   # (1, 4)
coords_b = coords.unsqueeze(0)   # (1, 2, 64, 64)
print('Running forward pass ...')
pred = model(params_b, coords_b)
print('pred:', pred.shape)       # expected (1, 4, 64, 64)

loss = relative_l2_loss(pred, fields.unsqueeze(0))
print('loss:', loss.item())

# Test one backward
loss.backward()
print('backward OK')
