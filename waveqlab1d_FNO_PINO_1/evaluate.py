"""
evaluate.py — Evaluation and super-resolution comparison for both Plan A and Plan B.

Usage:
    # Plan B: evaluate a single model
    python evaluate.py --config configs/sw_absorbing.yaml \
                       --checkpoint checkpoints/sw_absorbing/best.pt \
                       --data_split test --out_dir results/sw_absorbing

    # Plan A: evaluate unified model
    python evaluate.py --config configs/unified.yaml \
                       --checkpoint checkpoints/unified/best.pt \
                       --data_split test --out_dir results/unified

    # Super-resolution: infer at HR grid after training on LR
    python evaluate.py --config configs/sw_absorbing.yaml \
                       --checkpoint checkpoints/sw_absorbing/best.pt \
                       --superres --out_dir results/sw_absorbing_superres
"""

import argparse
import sys
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from model import SeparateFNO2d, UnifiedFNO2d
from data_gen.dataset import RuptureDataset, collate_with_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FRIC_IDX = {"SW": 0, "RS": 1}
BC_IDX   = {"free": 0, "absorbing": 1, "pml": 2}
ALL_CONDITIONS = ["sw_free", "sw_absorbing", "sw_pml",
                  "rs_free", "rs_absorbing", "rs_pml"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cfg_get(cfg, *keys, default=None):
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v


def rel_l2(pred, gt):
    """Relative L2 error: ||pred - gt|| / ||gt||, averaged over batch."""
    diff = (pred - gt).reshape(pred.shape[0], -1).norm(dim=1)
    norm = gt.reshape(gt.shape[0], -1).norm(dim=1).clamp(min=1e-8)
    return (diff / norm).mean().item()


def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path, device):
    plan = cfg.get("plan", "B")
    mc   = cfg["model"]

    if plan == "A":
        model = UnifiedFNO2d(
            c_in=mc["c_in"],
            modes_x=mc["modes_x"],
            modes_t=mc["modes_t"],
            width=mc["width"],
            n_layers=mc["n_layers"],
            c_out=mc.get("c_out", 4),
            n_fric=2, n_bc=3,
            embed_dim=mc.get("embed_dim", 16),
        )
    else:
        model = SeparateFNO2d(
            c_in=mc["c_in"],
            modes_x=mc["modes_x"],
            modes_t=mc["modes_t"],
            width=mc["width"],
            n_layers=mc["n_layers"],
            c_out=mc.get("c_out", 4),
        )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded %s  params=%s  from %s",
             model.__class__.__name__, f"{n_params:,}", checkpoint_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, plan, out_norm=None):
    """
    Run full inference over loader.
    Returns:
        preds  : (N, C_out, NX, NT)  — denormalized predictions
        truths : (N, C_out, NX, NT)  — denormalized ground truth
        errors : dict  {channel_name: rel_l2}
    """
    pred_list, gt_list = [], []
    for batch in loader:
        inp, out, fric_idx, bc_idx, meta = _to_device(batch, device)

        if plan == "A":
            pred = model(inp, fric_idx, bc_idx)
        else:
            pred = model(inp)

        pred_list.append(pred.cpu())
        gt_list.append(out.cpu())

    preds  = torch.cat(pred_list,  dim=0)    # (N, 4, NX, NT)
    truths = torch.cat(gt_list,    dim=0)

    # Denormalize
    if out_norm is not None:
        preds  = torch.tensor(out_norm.inverse_transform(preds.numpy()))
        truths = torch.tensor(out_norm.inverse_transform(truths.numpy()))

    channel_names = ["v_l", "s_l", "v_r", "s_r"]
    errors = {
        name: rel_l2(preds[:, i], truths[:, i])
        for i, name in enumerate(channel_names)
    }
    errors["mean"] = float(np.mean(list(errors.values())))
    return preds.numpy(), truths.numpy(), errors


# ─────────────────────────────────────────────────────────────────────────────
# Super-resolution inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_superres_inference(model, dataset_lr, dataset_hr, device, plan,
                           out_norm=None):
    """
    Evaluate super-resolution capability:
    - Predict at LR (training resolution)
    - Predict at HR by feeding HR input directly (FNO is resolution-invariant)
    - Compare both against HR ground truth via bilinear upsampling baseline

    The FNO handles arbitrary resolution because Fourier modes are fixed;
    increasing NX/NT only adds more spatial/temporal sample points.
    """
    preds_lr, preds_hr, truths_hr = [], [], []

    loader_lr = DataLoader(dataset_lr, batch_size=4, shuffle=False,
                           collate_fn=collate_with_meta)
    loader_hr = DataLoader(dataset_hr, batch_size=4, shuffle=False,
                           collate_fn=collate_with_meta)

    for (batch_lr, batch_hr) in zip(loader_lr, loader_hr):
        inp_lr, out_lr, fric_i, bc_i, meta = _to_device(batch_lr, device)
        inp_hr, out_hr, fric_i2, bc_i2, _  = _to_device(batch_hr, device)

        if plan == "A":
            pred_lr = model(inp_lr, fric_i, bc_i)
            pred_hr = model(inp_hr, fric_i2, bc_i2)
        else:
            pred_lr = model(inp_lr)
            pred_hr = model(inp_hr)

        preds_lr.append(pred_lr.cpu())
        preds_hr.append(pred_hr.cpu())
        truths_hr.append(out_hr.cpu())

    preds_lr  = torch.cat(preds_lr,  0).numpy()
    preds_hr  = torch.cat(preds_hr,  0).numpy()
    truths_hr = torch.cat(truths_hr, 0).numpy()

    if out_norm is not None:
        preds_lr  = out_norm.inverse_transform(preds_lr)
        preds_hr  = out_norm.inverse_transform(preds_hr)
        truths_hr = out_norm.inverse_transform(truths_hr)

    # Bilinear baseline: upsample LR predictions to HR shape
    nx_hr, nt_hr = truths_hr.shape[2], truths_hr.shape[3]
    preds_lr_t = torch.tensor(preds_lr)
    preds_lr_up = F.interpolate(
        preds_lr_t, size=(nx_hr, nt_hr), mode="bilinear", align_corners=False
    ).numpy()

    def _rel_l2(a, b):
        return float(np.mean(
            np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1)
            / (np.linalg.norm(b.reshape(b.shape[0], -1), axis=1) + 1e-8)
        ))

    errs = {
        "lr→lr (training res)":     _rel_l2(preds_lr, _downsample(truths_hr, preds_lr.shape[2:])),
        "lr→hr (bilinear upsample)": _rel_l2(preds_lr_up, truths_hr),
        "hr→hr (FNO super-res)":     _rel_l2(preds_hr, truths_hr),
    }
    return preds_lr, preds_lr_up, preds_hr, truths_hr, errs


def _downsample(arr, shape_2d):
    """Downsample (N,C,NX,NT) numpy array to (N,C,nx,nt) via F.interpolate."""
    t = torch.tensor(arr)
    return F.interpolate(t, size=shape_2d, mode="bilinear",
                         align_corners=False).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sample(pred, truth, idx, out_dir, tag="", dt=1.0, dx=1.0):
    """
    Space-time panel comparison for a single sample.
    Shows v_l, s_l, v_r, s_r (4 channels) as 2D images plus relative error.
    """
    channels = ["v_l", "s_l", "v_r", "s_r"]
    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, 3, figsize=(14, 3.5 * n_ch))

    for ci, name in enumerate(channels):
        p = pred[ci]    # (NX, NT)
        g = truth[ci]
        e = np.abs(p - g)

        vmax = max(np.abs(g).max(), 1e-12)
        emax = e.max() or 1e-12

        ax_p, ax_g, ax_e = axes[ci]
        ax_p.imshow(p, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax,
                    cmap="RdBu_r")
        ax_g.imshow(g, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax,
                    cmap="RdBu_r")
        im = ax_e.imshow(e, aspect="auto", origin="lower", vmin=0, vmax=emax,
                         cmap="hot_r")

        ax_p.set_title(f"{name} — FNO pred")
        ax_g.set_title(f"{name} — Ground truth")
        ax_e.set_title(f"{name} — |error|  max={emax:.2e}")
        plt.colorbar(im, ax=ax_e)

    fig.suptitle(f"Sample {idx}  {tag}", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{idx:04d}{tag}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_distribution(errors_per_sample, out_dir: Path, tag=""):
    """Histogram of per-sample relative L2 errors."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors_per_sample, bins=40, edgecolor="k", alpha=0.75)
    ax.axvline(float(np.median(errors_per_sample)), color="r", lw=1.5,
               label=f"median={float(np.median(errors_per_sample)):.3f}")
    ax.axvline(float(np.mean(errors_per_sample)),   color="b", lw=1.5,
               linestyle="--",
               label=f"mean={float(np.mean(errors_per_sample)):.3f}")
    ax.set_xlabel("Relative L2 error");  ax.set_ylabel("Count")
    ax.set_title(f"Error distribution  {tag}")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"error_hist{tag}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_superres_comparison(preds_lr_up, preds_hr, truths_hr, idx,
                              out_dir: Path, tag=""):
    """
    Side-by-side: bilinear upsample vs FNO super-res vs ground truth.
    Shows v_l field (channel 0) as representative.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ch = 0   # v_l
    panels = [
        (preds_lr_up[idx, ch], "Bilinear upsample (LR→HR)"),
        (preds_hr[idx, ch],    "FNO super-resolution (HR)"),
        (truths_hr[idx, ch],   "Ground truth (HR)"),
    ]
    vmax = max(np.abs(truths_hr[idx, ch]).max(), 1e-12)

    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(data, aspect="auto", origin="lower",
                       vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Super-resolution  sample {idx}  {tag}", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"superres_{idx:04d}{tag}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition summary (Plan A)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_conditions_unified(model, cfg, data_root, device,
                                    out_norm, split, out_dir):
    """Evaluate unified model separately on each condition."""
    results = {}
    for cond in ALL_CONDITIONS:
        fric_law, bc_mode = cond.split("_", 1)
        cond_dir = data_root / cond / split
        if not cond_dir.exists():
            log.warning("Skip missing: %s", cond_dir)
            continue

        ds = RuptureDataset(
            data_dir=str(cond_dir),
            plan="A",
            fric_law=fric_law.upper(),
            bc_mode=bc_mode,
            res="lr",
            out_norm=out_norm,
            return_meta=True,
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_with_meta)
        preds, truths, errs = run_inference(model, loader, device, "A", out_norm)
        results[cond] = errs
        log.info("[%s]  mean_rel_L2=%.4f", cond, errs["mean"])

    # Summary table
    log.info("\n=== Per-condition errors ===")
    log.info("%-20s  v_l      s_l      v_r      s_r      mean", "condition")
    for cond, e in results.items():
        log.info("%-20s  %.4f   %.4f   %.4f   %.4f   %.4f",
                 cond, e["v_l"], e["s_l"], e["v_r"], e["s_r"], e["mean"])

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    conds  = list(results.keys())
    means  = [results[c]["mean"] for c in conds]
    ax.bar(range(len(conds)), means, tick_label=conds)
    ax.set_ylabel("Mean relative L2 error")
    ax.set_title("Unified FNO — per-condition accuracy")
    plt.xticks(rotation=30, ha="right")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "per_condition_errors.png", dpi=120)
    plt.close(fig)
    log.info("Saved: %s", out_dir / "per_condition_errors.png")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PINO-FNO")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_split", default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--superres",   action="store_true",
                   help="Also run super-resolution comparison")
    p.add_argument("--out_dir",    default="results")
    p.add_argument("--device",     default=None)
    p.add_argument("--n_plot",     type=int, default=6,
                   help="Number of sample plots to generate")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    out_dir = Path(args.out_dir)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(cfg, args.checkpoint, device)

    # ── Load normalizers ──────────────────────────────────────────────────────
    norm_file = Path(cfg_get(cfg, "paths", "normalizer_file",
                              default=f"checkpoints/{cfg.get('model_key','model')}/normalizers.pkl"))
    out_norm = None
    if norm_file.exists():
        with open(norm_file, "rb") as f:
            pack = pickle.load(f)
        out_norm = pack.get("out_norm")

    plan       = cfg.get("plan", "B")
    data_root  = Path(cfg_get(cfg, "paths", "data_dir", default="data"))
    split      = args.data_split

    # ── Plan-A: per-condition evaluation ──────────────────────────────────────
    if plan == "A":
        results = evaluate_all_conditions_unified(
            model, cfg, data_root, device, out_norm, split, out_dir
        )
        # Sample plots for each condition
        for cond in list(results.keys())[:3]:
            fric_law, bc_mode = cond.split("_", 1)
            cond_dir = data_root / cond / split
            if not cond_dir.exists():
                continue
            ds = RuptureDataset(
                data_dir=str(cond_dir), plan="A",
                fric_law=fric_law.upper(), bc_mode=bc_mode,
                res="lr", out_norm=out_norm, return_meta=True,
            )
            loader = DataLoader(ds, batch_size=4, collate_fn=collate_with_meta)
            preds, truths, _ = run_inference(model, loader, device, "A", out_norm)
            for k in range(min(args.n_plot // 3 + 1, len(preds))):
                _plot_sample(preds[k], truths[k], k,
                             out_dir / cond, tag=f"_{cond}")
        return

    # ── Plan-B: single condition ──────────────────────────────────────────────
    fric_law = cfg["fric_law"]
    bc_mode  = cfg["bc_mode"]

    ds_lr = RuptureDataset(
        data_dir=str(data_root / split),
        plan="B",
        fric_law=fric_law,
        bc_mode=bc_mode,
        res="lr",
        out_norm=out_norm,
        return_meta=True,
    )

    loader_lr = DataLoader(ds_lr, batch_size=8, shuffle=False,
                            collate_fn=collate_with_meta)
    preds, truths, errors = run_inference(model, loader_lr, device, "B", out_norm)

    log.info(
        "Results [%s_%s]  v_l=%.4f  s_l=%.4f  v_r=%.4f  s_r=%.4f  mean=%.4f",
        fric_law, bc_mode,
        errors["v_l"], errors["s_l"], errors["v_r"], errors["s_r"], errors["mean"],
    )

    # Per-sample error
    per_sample = np.array([
        float(np.linalg.norm((preds[i] - truths[i]).ravel()) /
              (np.linalg.norm(truths[i].ravel()) + 1e-8))
        for i in range(len(preds))
    ])
    plot_error_distribution(per_sample, out_dir, tag=f"_{fric_law}_{bc_mode}")

    # Sample plots
    for k in range(min(args.n_plot, len(preds))):
        _plot_sample(preds[k], truths[k], k, out_dir,
                     tag=f"_{fric_law}_{bc_mode}")

    # ── Super-resolution ─────────────────────────────────────────────────────
    if args.superres:
        dc = cfg.get("data", {})
        nx_hr = dc.get("nx_hr", 256)
        nt_hr = dc.get("nt_hr", 256)
        ds_hr_path = data_root / f"{split}_hr"

        if not ds_hr_path.exists():
            log.warning("HR data dir not found: %s  (run generate_dataset with --nx_hr %d)",
                        ds_hr_path, nx_hr)
        else:
            ds_hr = RuptureDataset(
                data_dir=str(ds_hr_path),
                plan="B",
                fric_law=fric_law,
                bc_mode=bc_mode,
                res="hr",
                out_norm=out_norm,
                return_meta=True,
            )
            (preds_lr, preds_lr_up, preds_hr, truths_hr, sr_errs) = \
                run_superres_inference(model, ds_lr, ds_hr, device, "B", out_norm)

            log.info("Super-resolution:")
            for k, v in sr_errs.items():
                log.info("  %-35s  %.4f", k, v)

            for k in range(min(args.n_plot, len(preds_hr))):
                plot_superres_comparison(preds_lr_up, preds_hr, truths_hr,
                                         k, out_dir, tag=f"_{fric_law}_{bc_mode}")

    log.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
