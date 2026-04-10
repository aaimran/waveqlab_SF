#!/usr/bin/env python3
"""
train.py — Training entry point for PINO-FNO rupture field prediction.

Usage:
    python3 train.py --config configs/fno_sw_baseline.yaml
    python3 train.py --config configs/fno_sw_baseline.yaml --resume checkpoints/epoch_020.pt

Saves:
    checkpoints/best.pt          — best validation loss
    checkpoints/epoch_{N:03d}.pt — periodic saves (save_every)
    checkpoints/normalizers.pkl  — fitted normalizers (from training set)
    checkpoints/config.yaml      — copy of the run config
"""

import argparse
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Local imports ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_gen.dataset import (
    RuptureDataset, Normalizer, fit_normalizers,
    save_normalizers, load_normalizers,
)
from model.fno import FNO2d
from model.physics_loss import relative_l2_loss, pde_loss, fault_bc_loss

# ── Config loading ────────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    raise ImportError("pyyaml is required: pip install pyyaml")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── LR Schedulers ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, n_batches_per_epoch: int):
    sched = cfg['training']['lr_scheduler']
    epochs = cfg['training']['epochs']
    warmup = cfg['training']['warmup_epochs']

    if sched == 'cosine':
        # Linear warmup → cosine decay
        def lr_lambda(step):
            epoch = step / max(n_batches_per_epoch, 1)
            if epoch < warmup:
                return epoch / max(warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    else:
        return None


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, val_loss: float):
    torch.save({
        'epoch'     : epoch,
        'model'     : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict() if scheduler else None,
        'val_loss'  : val_loss,
    }, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and ckpt['scheduler']:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt['val_loss']


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, optimizer, scheduler, cfg, device,
    phase: str = 'train',
    mu: float = 0.0,    # shear modulus (Pa) — used for PDE loss
) -> float:
    """Run one epoch. Returns mean total loss."""
    is_train = (phase == 'train')
    model.train(is_train)
    torch.set_grad_enabled(is_train)

    phys = cfg['physics']
    tr   = cfg['training']
    lambda_data = tr['lambda_data']
    lambda_pde  = tr['lambda_pde']
    lambda_bc   = tr['lambda_bc']

    # Physical grid spacings (used in PDE loss)
    dx = phys['L'] / (cfg['data']['nx_out'] - 1)
    dt = phys['tend'] / (cfg['data']['nt_out'] - 1)

    total_loss = 0.0
    batches = 0

    for params, fields, coords in loader:
        params = params.to(device)    # (B, N_params)
        fields = fields.to(device)    # (B, 4, NX, NT)
        coords = coords.to(device)    # (B, 2, NX, NT)

        pred = model(params, coords)  # (B, 4, NX, NT)

        loss = lambda_data * relative_l2_loss(pred, fields)

        if lambda_pde > 0:
            loss = loss + lambda_pde * pde_loss(
                pred, phys['rho'], mu, dx, dt
            )

        if lambda_bc > 0:
            # Tau_0 is col 0 of params (unnormalized needed for fault BC)
            # For POC, skip BC loss when normalizers are applied to params
            # (raw Tau_0 not available here without inverse-transforming)
            pass

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fno_sw_baseline.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr  = cfg['training']
    md  = cfg['model']
    ph  = cfg['physics']

    # Reproducibility
    torch.manual_seed(tr['seed'])
    np.random.seed(tr['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Normalizers ─────────────────────────────────────────────────────────
    ckpt_dir = tr['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    norm_path = os.path.join(ckpt_dir, 'normalizers.pkl')

    train_dir = cfg['data']['train_dir']
    val_dir   = cfg['data']['val_dir']

    if os.path.exists(norm_path):
        print(f'Loading normalizers from {norm_path}')
        param_norm, field_norms = load_normalizers(norm_path)
    else:
        print('Fitting normalizers from training set ...')
        param_norm, field_norms = fit_normalizers(train_dir)
        save_normalizers(param_norm, field_norms, norm_path)
        print(f'  Saved to {norm_path}')

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds = RuptureDataset(train_dir, param_norm, field_norms)
    val_ds   = RuptureDataset(val_dir,   param_norm, field_norms)

    train_loader = DataLoader(
        train_ds, batch_size=tr['batch_size'], shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr['batch_size'], shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    print(f'Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples')

    # ── Model ────────────────────────────────────────────────────────────────
    model = FNO2d(
        n_params = md['n_params'],
        modes_x  = md['modes_x'],
        modes_t  = md['modes_t'],
        width    = md['width'],
        n_layers = md['n_layers'],
        c_out    = md['c_out'],
    ).to(device)

    n_params_total = sum(p.numel() for p in model.parameters())
    print(f'FNO2d parameters: {n_params_total:,}')

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay']
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Physical constants for PDE loss
    mu = ph['rho'] * ph['cs'] ** 2    # shear modulus (Pa)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.isfile(args.resume):
        print(f'Resuming from {args.resume}')
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1

    # Save a copy of the config
    shutil.copy(args.config, os.path.join(ckpt_dir, 'config.yaml'))

    # ── Training loop ────────────────────────────────────────────────────────
    print(f'\nTraining for {tr["epochs"]} epochs ...\n')
    print(f'{"Epoch":>6}  {"Train Loss":>12}  {"Val Loss":>12}  {"LR":>12}  {"Time":>8}')
    print('-' * 60)

    for epoch in range(start_epoch, tr['epochs']):
        t0 = time.perf_counter()

        train_loss = run_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, 'train', mu
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, scheduler, cfg, device, 'val', mu
        )

        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % tr['log_every'] == 0 or epoch == 0:
            print(f'{epoch+1:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  '
                  f'{current_lr:>12.2e}  {elapsed:>7.1f}s')

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(ckpt_dir, 'best.pt'),
                epoch, model, optimizer, scheduler, val_loss
            )

        # Periodic save
        if (epoch + 1) % tr['save_every'] == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f'epoch_{epoch+1:03d}.pt'),
                epoch, model, optimizer, scheduler, val_loss
            )

    print(f'\nTraining complete. Best val loss: {best_val_loss:.6f}')
    print(f'Best checkpoint: {os.path.join(ckpt_dir, "best.pt")}')


if __name__ == '__main__':
    main()
