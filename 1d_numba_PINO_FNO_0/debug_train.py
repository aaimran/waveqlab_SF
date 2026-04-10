#!/usr/bin/env python3
"""Quick benchmark: measure time per epoch to estimate full training duration."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from data_gen.dataset import RuptureDataset, load_normalizers
from model.fno import FNO2d
from model.physics_loss import relative_l2_loss

param_norm, field_norms = load_normalizers('checkpoints/normalizers.pkl')
train_ds = RuptureDataset('data/train', param_norm, field_norms)
val_ds   = RuptureDataset('data/val',   param_norm, field_norms)

loader  = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
vloader = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

model = FNO2d().cpu()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

print('Starting 3 training test epochs ...')
sys.stdout.flush()

for epoch in range(3):
    t0 = time.perf_counter()
    model.train()
    total = 0.0
    for i, (params, fields, coords) in enumerate(loader):
        pred = model(params, coords)
        loss = relative_l2_loss(pred, fields)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        if i == 0:
            print(f'  Ep{epoch+1} batch0 loss={loss.item():.4f}  elapsed={time.perf_counter()-t0:.1f}s')
            sys.stdout.flush()
    elapsed = time.perf_counter() - t0
    print(f'  Epoch {epoch+1}/3  train_loss={total/len(loader):.4f}  {elapsed:.1f}s/epoch')
    sys.stdout.flush()

print('Benchmark done.')
