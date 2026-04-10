"""
train_separate.py — Plan B: Train one SeparateFNO2d model for a single condition.

Usage:
    python train_separate.py --config configs/sw_absorbing.yaml
    python train_separate.py --config configs/rs_pml.yaml --resume checkpoints/rs_pml/epoch_050.pt

Each invocation trains a single (fric_law × bc_mode) model.
To train all 6 models, run this script six times with the corresponding configs,
optionally in parallel on separate GPUs.
"""

import argparse
import itertools
import os
import random
import sys
import time
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from model import SeparateFNO2d, PINOLoss
from data_gen.dataset import RuptureDataset, Normalizer, collate_with_meta

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cfg_get(cfg, *keys, default=None):
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(cfg):
    data_root  = Path(cfg_get(cfg, "paths", "data_dir", default="data"))
    norm_file  = Path(cfg_get(cfg, "paths", "normalizer_file",
                               default=f"checkpoints/{cfg['model_key']}/normalizers.pkl"))

    fric_law = cfg["fric_law"]      # "SW" or "RS"
    bc_mode  = cfg["bc_mode"]       # "free", "absorbing", "pml"

    if norm_file.exists():
        log.info("Loading normalizers from %s", norm_file)
        with open(norm_file, "rb") as f:
            pack = pickle.load(f)
        in_norm  = pack["in_norm"]
        out_norm = pack["out_norm"]
    else:
        in_norm = out_norm = None

    def make_ds(split, res="lr"):
        return RuptureDataset(
            data_dir=str(data_root / split),
            plan="B",
            fric_law=fric_law,
            bc_mode=bc_mode,
            res=res,
            in_norm=in_norm,
            out_norm=out_norm,
            return_meta=True,
        )

    train_ds = make_ds("train", res="lr")
    val_ds   = make_ds("val",   res="lr")

    if in_norm is None:
        log.info("Fitting normalizers …")
        all_inp, all_out = [], []
        for i in range(len(train_ds)):
            inp, out, *_ = train_ds[i]
            all_inp.append(inp.numpy() if isinstance(inp, torch.Tensor) else inp)
            all_out.append(out.numpy() if isinstance(out, torch.Tensor) else out)
        in_norm  = Normalizer.from_data(np.stack(all_inp, 0))
        out_norm = Normalizer.from_data(np.stack(all_out, 0))
        norm_file.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_file, "wb") as f:
            pickle.dump({"in_norm": in_norm, "out_norm": out_norm}, f)
        log.info("Saved normalizers to %s", norm_file)
        train_ds.in_norm  = in_norm
        train_ds.out_norm = out_norm
        val_ds.in_norm    = in_norm
        val_ds.out_norm   = out_norm

    # HR dataset shares the same normalizers (same physical quantities, diff grid)
    hr_dir = data_root / "train"
    hr_files_exist = any(
        f.endswith('.npz') for f in os.listdir(str(hr_dir))
        if f.startswith('sample_')
    ) if hr_dir.exists() else False
    # check first sample has _hr keys
    _hr_available = False
    if hr_files_exist:
        import numpy as _np
        _probe = _np.load(str(sorted(hr_dir.glob('sample_*.npz'))[0]),
                          allow_pickle=True)
        _hr_available = 'v_l_hr' in _probe
    train_ds_hr = None
    if _hr_available:
        train_ds_hr = make_ds("train", res="hr")
        train_ds_hr.in_norm  = in_norm
        train_ds_hr.out_norm = out_norm
        log.info("  HR dataset available (%d samples)", len(train_ds_hr))
    else:
        log.info("  HR dataset not available — multiscale disabled")

    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))
    return train_ds, train_ds_hr, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg) -> SeparateFNO2d:
    mc = cfg["model"]
    model = SeparateFNO2d(
        c_in=mc["c_in"],
        modes_x=mc["modes_x"],
        modes_t=mc["modes_t"],
        width=mc["width"],
        n_layers=mc["n_layers"],
        c_out=mc.get("c_out", 4),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("SeparateFNO2d [%s]  params = %s", cfg.get("model_key", "?"),
             f"{n_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def run_epoch(model, loader, optimizer, criterion, device, train: bool = True,
              loader_hr=None, hr_frac: float = 0.0):
    """Run one epoch.  If loader_hr is given and hr_frac > 0, each batch is
    drawn from the HR loader with probability hr_frac (LR otherwise)."""
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_loss, total_data, n_batches = 0.0, 0.0, 0
    hr_iter = itertools.cycle(loader_hr) if (loader_hr is not None and hr_frac > 0) else None

    with ctx:
        for batch in loader:
            if hr_iter is not None and random.random() < hr_frac:
                batch = next(hr_iter)
            inp, out, fric_idx, bc_idx, meta = _to_device(batch, device)
            # Plan B: no fric/bc conditioning — model ignores those indices

            pred = model(inp)           # SeparateFNO2d has no index args

            dx = meta.pop("dx", 1.0) if isinstance(meta, dict) else 1.0
            dt = meta.pop("dt", 0.01) if isinstance(meta, dict) else 0.01
            losses = criterion(pred, out, meta, dx, dt)
            data_loss = losses.get("data", losses["total"])
            loss = losses["total"]

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_data += data_loss.item() if isinstance(data_loss, torch.Tensor) \
                          else data_loss
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_data / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg):
    ckpt_dir = Path(cfg_get(cfg, "paths", "checkpoint_dir",
                             default=f"checkpoints/{cfg.get('model_key','model')}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "config": cfg,
    }
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)

    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        torch.save(state, best_path)
    else:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if val_loss < best.get("val_loss", float("inf")):
            torch.save(state, best_path)
            log.info("  ↑ New best val_loss = %.6f", val_loss)
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    log.info("Resumed from %s  (epoch %d  val_loss=%.6f)",
             path, state["epoch"], state.get("val_loss", float("nan")))
    return state["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Separate FNO (Plan B)")
    p.add_argument("--config", required=True,   help="YAML config file")
    p.add_argument("--resume", default=None,    help="Checkpoint to resume from")
    p.add_argument("--device", default=None,    help="cuda / cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    assert cfg.get("plan") == "B", "This script is for Plan B configs only"

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Model: %s", device, cfg.get("model_key", "?"))

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, train_ds_hr, val_ds = build_datasets(cfg)

    tc = cfg.get("training", {})
    multiscale = tc.get("multiscale", False)
    hr_frac    = float(tc.get("multiscale_hr_frac", 0.3)) if multiscale else 0.0

    train_loader = DataLoader(
        train_ds,
        batch_size=tc.get("batch_size", 16),
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_meta,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.get("batch_size", 16) * 2,
        shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_meta,
    )
    # HR loader (used for multiscale — same batch size, full shuffle)
    train_loader_hr = None
    if train_ds_hr is not None and hr_frac > 0:
        train_loader_hr = DataLoader(
            train_ds_hr,
            batch_size=tc.get("batch_size", 16),
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_with_meta,
            drop_last=True,
        )
        log.info("Multiscale training: hr_frac=%.2f  HR loader batches=%d",
                 hr_frac, len(train_loader_hr))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    lc = cfg.get("loss", {})
    criterion = PINOLoss(
        fric_law=cfg["fric_law"],
        bc_mode=cfg["bc_mode"],
        w_data=lc.get("w_data",  1.0),
        w_pde=lc.get("w_pde",   0.05),
        w_fault=lc.get("w_fault", 0.08),
        w_outer=lc.get("w_outer", 0.10),
        w_ic=lc.get("w_ic",    0.05),
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    epochs = tc.get("epochs", 80)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-5),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    save_every = tc.get("save_every", 10)
    log.info("Starting: epochs=%d  batches/epoch=%d", epochs, len(train_loader))

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        criterion.update_weights(epoch)

        train_loss, train_data = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True,
            loader_hr=train_loader_hr, hr_frac=hr_frac,
        )
        val_loss, val_data = run_epoch(
            model, val_loader, None, criterion, device, train=False
        )
        scheduler.step()

        dt = time.perf_counter() - t0
        log.info(
            "E %4d/%d  tr=%.5f (d=%.5f)  val=%.5f (d=%.5f)  lr=%.2e  %.1fs",
            epoch + 1, epochs,
            train_loss, train_data,
            val_loss,   val_data,
            scheduler.get_last_lr()[0], dt,
        )

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, cfg)

    log.info("Done.")


if __name__ == "__main__":
    main()
