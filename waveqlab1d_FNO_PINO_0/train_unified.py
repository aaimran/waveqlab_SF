"""
train_unified.py — Plan A: Train UnifiedFNO2d on all 6 conditions simultaneously.

Usage:
    python train_unified.py --config configs/unified.yaml [--resume checkpoints/unified/epoch_050.pt]

The model takes ALL conditions (sw_free, sw_absorbing, sw_pml, rs_free, rs_absorbing, rs_pml)
via a single joint dataset. FiLM layers handle discrete (fric_law, bc_mode) conditioning.
Multi-scale training alternates between LR and HR samples for super-resolution capability.
"""

import argparse
import os
import sys
import time
import math
import yaml
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── ensure local modules are on the path ──────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from model import UnifiedFNO2d, PINOLoss, build_input_tensor_unified
from data_gen.dataset import RuptureDataset, Normalizer, collate_with_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── map condition keys to integer indices ──────────────────────────────────────
FRIC_IDX  = {"SW": 0, "RS": 1}
BC_IDX    = {"free": 0, "absorbing": 1, "pml": 2}
ALL_CONDITIONS = ["sw_free", "sw_absorbing", "sw_pml",
                  "rs_free", "rs_absorbing", "rs_pml"]


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def cfg_get(cfg: dict, *keys, default=None):
    """Safe nested get: cfg_get(cfg, 'training', 'lr')."""
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(cfg: dict, device: torch.device):
    """
    Build joint train/val datasets across all 6 conditions.

    Each sub-dataset (one per condition) is wrapped in RuptureDataset(plan='A').
    Normalizers are fitted on training data and cached to disk.
    """
    data_root = Path(cfg_get(cfg, "paths", "data_dir", default="data"))
    norm_file = Path(cfg_get(cfg, "paths", "normalizer_file",
                              default="checkpoints/unified/normalizers.pkl"))

    train_sets, val_sets = [], []

    if norm_file.exists():
        log.info("Loading cached normalizers from %s", norm_file)
        with open(norm_file, "rb") as f:
            norm_pack = pickle.load(f)
        in_norm  = norm_pack["in_norm"]
        out_norm = norm_pack["out_norm"]
    else:
        in_norm = out_norm = None

    for cond in ALL_CONDITIONS:
        fric_law, bc_mode = cond.split("_", 1)
        cond_dir = data_root / cond
        if not cond_dir.exists():
            log.warning("Data dir missing for %s: %s — skipping", cond, cond_dir)
            continue

        ds_train = RuptureDataset(
            data_dir=str(cond_dir / "train"),
            plan="A",
            fric_law=fric_law.upper(),
            bc_mode=bc_mode,
            res="lr",
            in_norm=in_norm,
            out_norm=out_norm,
            return_meta=True,
        )
        ds_val = RuptureDataset(
            data_dir=str(cond_dir / "val"),
            plan="A",
            fric_law=fric_law.upper(),
            bc_mode=bc_mode,
            res="lr",
            in_norm=in_norm,
            out_norm=out_norm,
            return_meta=True,
        )
        train_sets.append(ds_train)
        val_sets.append(ds_val)

    if not train_sets:
        raise RuntimeError("No training data found. Run data_gen/generate_dataset.py first.")

    # If no pre-built normalizers, fit them on training data
    if in_norm is None:
        log.info("Fitting normalizers on training data …")
        in_norm, out_norm = _fit_normalizers(train_sets)
        norm_file.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_file, "wb") as f:
            pickle.dump({"in_norm": in_norm, "out_norm": out_norm}, f)
        log.info("Normalizers saved to %s", norm_file)
        # Re-build datasets with fitted normalizers
        for ds in train_sets + val_sets:
            ds.in_norm  = in_norm
            ds.out_norm = out_norm

    train_joint = ConcatDataset(train_sets)
    val_joint   = ConcatDataset(val_sets)
    log.info("Train samples: %d  |  Val samples: %d", len(train_joint), len(val_joint))
    return train_joint, val_joint, in_norm, out_norm


def _fit_normalizers(datasets):
    """Accumulate running mean/var across all datasets to build normalizers."""
    inp_list, out_list = [], []
    for ds in datasets:
        for i in range(len(ds)):
            inp, out, *_ = ds[i]
            inp_list.append(inp.numpy() if isinstance(inp, torch.Tensor) else inp)
            out_list.append(out.numpy() if isinstance(out, torch.Tensor) else out)
    inp_arr = np.stack(inp_list, axis=0)   # (N, C_in, NX, NT)
    out_arr = np.stack(out_list, axis=0)   # (N, C_out, NX, NT)
    in_norm  = Normalizer.from_data(inp_arr)
    out_norm = Normalizer.from_data(out_arr)
    return in_norm, out_norm


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> UnifiedFNO2d:
    mc = cfg["model"]
    model = UnifiedFNO2d(
        c_in=mc["c_in"],
        modes_x=mc["modes_x"],
        modes_t=mc["modes_t"],
        width=mc["width"],
        n_layers=mc["n_layers"],
        c_out=mc.get("c_out", 4),
        n_fric=2,
        n_bc=3,
        embed_dim=mc.get("embed_dim", 16),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("UnifiedFNO2d  params = %s", f"{n_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(batch, device):
    """Move all tensors in a batch to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def run_epoch(model, loader, optimizer, criterion, device, cfg, epoch,
              train: bool = True):
    """
    Run one training or validation epoch.

    The criterion (PINOLoss) is called differently depending on the condition:
    For each batch, fric_idx and bc_idx tell us which condition we are in.
    Since Plan A can mix conditions, we compute the loss sample-by-sample within
    the batch and average.
    """
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_loss, total_data, n_batches = 0.0, 0.0, 0

    with ctx:
        for batch in loader:
            inp, out, fric_idx, bc_idx, meta = _to_device(batch, device)
            # inp:      (B, C_in, NX, NT)
            # out:      (B, C_out, NX, NT)
            # fric_idx: (B,)  int
            # bc_idx:   (B,)  int

            pred = model(inp, fric_idx, bc_idx)  # (B, C_out, NX, NT)

            # Compute PINO loss per sample (conditions may be mixed in a batch)
            loss, data_loss = _pino_loss_mixed(
                pred, out, fric_idx, bc_idx, meta, criterion, cfg
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item()
            total_data  += data_loss.item()
            n_batches   += 1

    return total_loss / max(n_batches, 1), total_data / max(n_batches, 1)


def _pino_loss_mixed(pred, out, fric_idx, bc_idx, meta, criterion_map, cfg):
    """
    Compute PINO loss when the batch may contain multiple conditions.
    criterion_map: dict mapping (fric_id, bc_id) -> PINOLoss instance.
    Returns (total_loss, data_loss) averaged over batch.
    """
    B = pred.shape[0]
    total_loss = pred.new_zeros(1)
    total_data = pred.new_zeros(1)

    for i in range(B):
        f_id = fric_idx[i].item()
        b_id = bc_idx[i].item()
        key = (f_id, b_id)

        if key not in criterion_map:
            # Fallback: pure data loss
            dl = torch.mean((pred[i] - out[i]) ** 2)
            total_loss = total_loss + dl
            total_data = total_data + dl
            continue

        criterion = criterion_map[key]
        fric_law = "SW" if f_id == 0 else "RS"
        bc_mode  = ["free", "absorbing", "pml"][b_id]

        # Build per-sample batch dict from meta
        m = {k: v[i].unsqueeze(0) if isinstance(v[i], torch.Tensor) else v[i]
             for k, v in meta.items()}
        dx = m.pop("dx", 1.0)
        dt = m.pop("dt", 0.01)
        losses = criterion(
            pred[i].unsqueeze(0),
            out[i].unsqueeze(0),
            m, dx, dt,
        )
        total_loss = total_loss + losses["total"]
        total_data = total_data + losses.get("data", losses["total"])

    return total_loss / B, total_data / B


def build_criterion_map(cfg: dict, device: torch.device) -> dict:
    """
    Build a PINOLoss instance for each (fric_law, bc_mode) combination.
    Returns dict keyed by (fric_idx, bc_idx).
    """
    lc = cfg.get("loss", {})

    criterion_map = {}
    for fric_name, fric_id in FRIC_IDX.items():
        for bc_name, bc_id in BC_IDX.items():
            criterion_map[(fric_id, bc_id)] = PINOLoss(
                fric_law=fric_name,
                bc_mode=bc_name,
                w_data=lc.get("w_data", 1.0),
                w_pde=lc.get("w_pde", 0.05),
                w_fault=lc.get("w_fault", 0.08),
                w_outer=lc.get("w_outer", 0.10),
                w_ic=lc.get("w_ic", 0.05),
            ).to(device)
    return criterion_map


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg):
    ckpt_dir = Path(cfg_get(cfg, "paths", "checkpoint_dir",
                             default="checkpoints/unified"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
    }
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)
    # Overwrite 'best' if this is the lowest validation loss
    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        torch.save(state, best_path)
    else:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if val_loss < best.get("val_loss", float("inf")):
            torch.save(state, best_path)
            log.info("  ↑ New best val loss: %.6f", val_loss)
    return path


def load_checkpoint(model, optimizer, scheduler, path: str, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    log.info("Resumed from %s (epoch %d, val_loss=%.6f)",
             path, state["epoch"], state.get("val_loss", float("nan")))
    return state["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Unified FNO (Plan A)")
    p.add_argument("--config", default="configs/unified.yaml",
                   help="YAML config file")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, in_norm, out_norm = build_datasets(cfg, device)

    tc = cfg.get("training", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=tc.get("batch_size", 8),
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_meta,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.get("batch_size", 8) * 2,
        shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_meta,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    epochs = tc.get("epochs", 100)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc.get("lr", 5e-4),
        weight_decay=tc.get("weight_decay", 1e-5),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion_map = build_criterion_map(cfg, device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    save_every = tc.get("save_every", 10)
    best_val   = float("inf")

    log.info("Starting training: epochs=%d  batches/epoch=%d", epochs, len(train_loader))

    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()

        # Update PINO loss weights for curriculum (softer → harder physics enforcement)
        for crit in criterion_map.values():
            crit.update_weights(epoch)

        train_loss, train_data = run_epoch(
            model, train_loader, optimizer, criterion_map, device, cfg, epoch,
            train=True
        )
        val_loss, val_data = run_epoch(
            model, val_loader, None, criterion_map, device, cfg, epoch,
            train=False
        )

        scheduler.step()

        dt = time.perf_counter() - t0
        log.info(
            "Epoch %4d/%d  train=%.5f (data=%.5f)  val=%.5f (data=%.5f)  "
            "lr=%.2e  time=%.1fs",
            epoch + 1, epochs,
            train_loss, train_data,
            val_loss,   val_data,
            scheduler.get_last_lr()[0], dt,
        )

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = save_checkpoint(model, optimizer, scheduler,
                                        epoch + 1, val_loss, cfg)
            log.info("  Checkpoint: %s", ckpt_path)

    log.info("Training complete.")


if __name__ == "__main__":
    main()
