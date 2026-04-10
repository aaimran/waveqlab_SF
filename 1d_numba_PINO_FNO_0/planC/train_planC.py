#!/usr/bin/env python3
"""
planC/train_planC.py
=====================
Plan-C training script: PINO-FNO with 6-term loss including energy stability.

Supports:
  - Single-resolution and multi-resolution (batch-swap) training
  - Cosine LR schedule with linear warmup
  - Epoch-20 weight ramp for physics terms
  - Best-model checkpoint + periodic saves
  - Graceful keyboard-interrupt (saves checkpoint)

Usage:
    cd /scratch/aimran/FNO/waveqlab_SF/1d_numba_PINO_FNO_0
    source /work/aimran/wql1d/env.sh

    python planC/train_planC.py \\
        --config planC/configs/sw_bc_both0_r80m.yaml

    python planC/train_planC.py \\
        --config planC/configs/sw_bc_both0_r40m.yaml \\
        --resume planC/checkpoints/sw_bc_both0_r40m/epoch_010.pt
"""

import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # planC/
sys.path.insert(0, _ROOT)   # 1d_numba_PINO_FNO_0/

from model.fno          import FNO2d
from model.physics_loss import PINOLoss
from data_gen.dataset   import (RuptureDatasetPlanC, Normalizer,
                                fit_normalizers, collate_with_meta)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg, *keys, default=None):
    """Safe nested get from config dict."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# LR schedule with linear warmup + cosine annealing
# ---------------------------------------------------------------------------

def _cosine_warmup_schedule(optimizer, epoch, warmup_epochs, total_epochs,
                            lr_max, lr_min):
    if epoch < warmup_epochs:
        lr = lr_max * (epoch + 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ---------------------------------------------------------------------------
# Build datasets and normalizers
# ---------------------------------------------------------------------------

def build_datasets(cfg):
    res      = cfg['data']['res']
    train_dir = cfg['data']['train_dir']
    val_dir   = cfg['data']['val_dir']

    print(f'  Fitting normalizers on {train_dir} [{res}] ...')
    param_norm, field_norm = fit_normalizers(train_dir, res=res)

    norm_dir = os.path.join(cfg['checkpoints']['dir'], 'normalizers')
    os.makedirs(norm_dir, exist_ok=True)
    param_norm.save(os.path.join(norm_dir, 'param_norm.npz'))
    field_norm.save(os.path.join(norm_dir, 'field_norm.npz'))
    print(f'  Normalizers saved to {norm_dir}')

    ds_train = RuptureDatasetPlanC(train_dir, res=res,
                                   param_norm=param_norm,
                                   field_norm=field_norm)
    ds_val   = RuptureDatasetPlanC(val_dir,   res=res,
                                   param_norm=param_norm,
                                   field_norm=field_norm)

    ds_hr = None
    if _get(cfg, 'data', 'multiscale', default=False):
        hr_res   = cfg['data']['multiscale_hr_res']
        hr_dir   = cfg['data']['multiscale_hr_dir']
        hr_p_n, hr_f_n = fit_normalizers(hr_dir, res=hr_res)
        ds_hr = RuptureDatasetPlanC(hr_dir, res=hr_res,
                                    param_norm=hr_p_n,
                                    field_norm=hr_f_n)
        hr_p_n.save(os.path.join(norm_dir, f'param_norm_{hr_res}.npz'))
        hr_f_n.save(os.path.join(norm_dir, f'field_norm_{hr_res}.npz'))

    return ds_train, ds_val, ds_hr


# ---------------------------------------------------------------------------
# Build model and loss
# ---------------------------------------------------------------------------

def build_model_and_loss(cfg, device):
    mc  = cfg['model']
    lc  = cfg['loss']
    bcc = cfg['bc']

    model = FNO2d(
        c_in=mc['c_in'],
        modes_x=mc['modes_x'],
        modes_t=mc['modes_t'],
        width=mc['width'],
        n_layers=mc['n_layers'],
        c_out=mc['c_out'],
    ).to(device)

    pc = cfg['physics']
    criterion = PINOLoss(
        r0_l    = bcc['r0_l'],
        r1_r    = bcc['r1_r'],
        rho     = pc['rho_ref'],
        mu      = pc['mu_ref'],
        cs      = pc['cs_ref'],
        sigma_n = pc['sigma_n_ref'],
        w_data  = lc['w_data'],
        w_pde   = lc['w_pde'],
        w_fault = lc['w_fault'],
        w_bc    = lc['w_bc'],
        w_ic    = lc['w_ic'],
        w_stab  = lc['w_stab'],
    )

    return model, criterion


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device,
              lr_loader=None, lr_frac=0.0,
              train=True, clip_norm=1.0):
    """
    lr_loader : DataLoader for HR data (multiscale)
    lr_frac   : fraction of batches drawn from lr_loader
    """
    model.train(train)
    total_loss  = 0.0
    n_batches   = 0
    loss_detail = {}

    lr_iter = itertools.cycle(lr_loader) if lr_loader is not None else None

    for inp, target, meta in loader:
        inp    = inp.to(device)
        target = target.to(device)
        meta_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in meta.items()}

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(inp)
        dx   = meta_d['dx_km'][0]   # same for whole batch
        dt   = meta_d['dt_s'][0]

        loss_d = criterion(pred, target, dx, dt,
                           meta_d['Tau_0'], meta_d['alp_s'],
                           meta_d['alp_d'], meta_d['D_c'])

        loss = loss_d['total']

        # Optional HR batch
        if lr_iter is not None and torch.rand(1).item() < lr_frac:
            inp_hr, tgt_hr, meta_hr = next(lr_iter)
            inp_hr = inp_hr.to(device)
            tgt_hr = tgt_hr.to(device)
            meta_hr_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in meta_hr.items()}
            pred_hr = model(inp_hr)
            dx_hr   = meta_hr_d['dx_km'][0]
            dt_hr   = meta_hr_d['dt_s'][0]
            loss_hr = criterion(pred_hr, tgt_hr, dx_hr, dt_hr,
                                meta_hr_d['Tau_0'], meta_hr_d['alp_s'],
                                meta_hr_d['alp_d'], meta_hr_d['D_c'])
            loss = loss + loss_hr['total']

        if train:
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        for k, v in loss_d.items():
            loss_detail[k] = loss_detail.get(k, 0.0) + v.item()

    avg = total_loss / max(n_batches, 1)
    avg_detail = {k: v / max(n_batches, 1) for k, v in loss_detail.items()}
    return avg, avg_detail


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_ckpt(path, epoch, model, optimizer, val_loss, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch'     : epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'val_loss'  : val_loss,
        'config'    : cfg,
    }, path)


def _load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    return ckpt['epoch'], ckpt.get('val_loss', float('inf'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device)

    ckpt_dir = cfg['checkpoints']['dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'\n=== Plan-C training: {cfg["experiment"]["name"]} ===')
    print(f'Device: {device}')
    print(f'Config: {args.config}')

    # Datasets
    print('\nBuilding datasets...')
    ds_train, ds_val, ds_hr = build_datasets(cfg)
    batch_size = cfg['training']['batch_size']

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_with_meta, drop_last=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True,
                              collate_fn=collate_with_meta)
    loader_hr    = None
    if ds_hr is not None:
        loader_hr = DataLoader(ds_hr, batch_size=max(1, batch_size // 2),
                               shuffle=True, num_workers=2, pin_memory=True,
                               collate_fn=collate_with_meta, drop_last=True)

    hr_frac = _get(cfg, 'data', 'multiscale_hr_frac', default=0.0)

    print(f'  train: {len(ds_train)}, val: {len(ds_val)}, '
          f'hr: {len(ds_hr) if ds_hr else 0}')

    # Model + loss
    print('\nBuilding model...')
    model, criterion = build_model_and_loss(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')

    # Optimizer
    tc  = cfg['training']
    opt = torch.optim.AdamW(model.parameters(),
                            lr=tc['lr'],
                            weight_decay=tc.get('weight_decay', 1e-4))

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = _load_ckpt(args.resume, model, opt, device)
        print(f'  Resumed from {args.resume} (epoch {start_epoch})')

    epochs      = tc['epochs']
    warmup      = tc.get('warmup_epochs', 5)
    lr_max      = tc['lr']
    lr_min      = tc.get('lr_min', 1e-5)
    clip_norm   = tc.get('clip_grad_norm', 1.0)
    save_every  = tc.get('save_every', 10)
    patience    = tc.get('patience', 25)
    ramp_epoch  = _get(cfg, 'loss', 'ramp_epoch', default=20)

    no_improve_count = 0
    log_every = _get(cfg, 'logging', 'log_every', default=5)

    print(f'\nTraining for {epochs} epochs...\n')

    try:
        for epoch in range(start_epoch, epochs):
            # LR schedule
            lr = _cosine_warmup_schedule(opt, epoch, warmup, epochs, lr_max, lr_min)

            # Physics loss weight ramp
            criterion.update_weights(epoch)

            t0 = time.time()
            train_loss, train_detail = run_epoch(
                model, loader_train, criterion, opt, device,
                lr_loader=loader_hr, lr_frac=hr_frac,
                train=True, clip_norm=clip_norm)

            with torch.no_grad():
                val_loss, val_detail = run_epoch(
                    model, loader_val, criterion, opt, device,
                    train=False, clip_norm=0.0)

            elapsed = time.time() - t0

            if (epoch + 1) % log_every == 0 or epoch == 0:
                print(f'Epoch {epoch+1:4d}/{epochs} '
                      f'| lr={lr:.2e} '
                      f'| train={train_loss:.4f} '
                      f'| val={val_loss:.4f} '
                      f'| data={val_detail.get("data",0):.4f} '
                      f'| pde={val_detail.get("pde",0):.4f} '
                      f'| stab={val_detail.get("stab",0):.4f} '
                      f'| {elapsed:.1f}s')

            # Best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                _save_ckpt(os.path.join(ckpt_dir, cfg['checkpoints']['best']),
                           epoch + 1, model, opt, val_loss, cfg)
            else:
                no_improve_count += 1

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                _save_ckpt(os.path.join(ckpt_dir, f'epoch_{epoch+1:04d}.pt'),
                           epoch + 1, model, opt, val_loss, cfg)

            # Early stopping
            if no_improve_count >= patience:
                print(f'Early stopping at epoch {epoch+1} '
                      f'(no improvement for {patience} epochs)')
                break

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt — saving current state...')
        _save_ckpt(os.path.join(ckpt_dir, f'interrupted_epoch_{epoch+1:04d}.pt'),
                   epoch + 1, model, opt, val_loss, cfg)

    print(f'\nDone. Best val loss: {best_val_loss:.6f}')
    print(f'Best checkpoint: {os.path.join(ckpt_dir, cfg["checkpoints"]["best"])}')


if __name__ == '__main__':
    main()
