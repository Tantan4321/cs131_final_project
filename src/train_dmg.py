"""Stage 2: train the Siamese U-Net + 4-class damage head.

Warm-starts both encoders from the Stage-1 checkpoint. Uses weighted CE + focal
loss with inverse-frequency class weights derived from outputs/dataset_stats.csv.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.xbd_dataset import XBDDataset, list_pairs, split_pairs
from src.models.siamese_unet import SiameseUNet
from src.paths import (
    CHECKPOINTS,
    DAMAGE_NAMES,
    FIGS,
    HOLDOUT_DISASTER,
    LOGS,
    OUTPUTS,
    ensure_dirs,
)


class FocalCE(nn.Module):
    def __init__(self, weight, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-6)
        focal = (1 - pt).pow(self.gamma) * ce
        return focal.mean()


def macro_f1(pred, target, num_classes=5, eps=1e-7):
    f1s = []
    for c in range(1, num_classes):  # ignore background
        p = (pred == c).float()
        t = (target == c).float()
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        f1s.append((2 * tp / (2 * tp + fp + fn + eps)).item())
    return float(np.mean(f1s)), f1s


def compute_class_weights():
    csv = OUTPUTS / "dataset_stats.csv"
    weights = torch.ones(5)
    if not csv.exists():
        print(f"[warn] {csv} not found -> using uniform class weights. "
              f"Run `python -m src.tools.dataset_stats` first for inverse-freq weights.")
        return weights
    df = pd.read_csv(csv)
    totals = df.groupby("subtype")["pixel_area"].sum().to_dict()
    # background pixels are ~everything; assign weight 1.0 for bg, inverse-freq for damage classes
    name_to_cls = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
    freqs = np.array([totals.get(n, 1.0) for n in name_to_cls])
    inv = freqs.max() / np.clip(freqs, 1.0, None)
    weights = torch.tensor([1.0, *inv.tolist()], dtype=torch.float32)
    print(f"Class weights: {dict(zip(DAMAGE_NAMES, weights.tolist()))}")
    return weights


def run_epoch(model, loader, device, optimizer, loss_fn, train):
    model.train(train)
    total_loss = 0.0
    total_f1 = 0.0
    per_class = np.zeros(4)
    n = 0
    pbar = tqdm(loader, desc="train" if train else "val ", leave=False)
    for batch in pbar:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        target = batch["dmg_mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(pre, post)
            loss = loss_fn(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            f1, f1s = macro_f1(pred, target)
        bs = pre.size(0)
        total_loss += loss.item() * bs
        total_f1 += f1 * bs
        per_class += np.array(f1s) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.3f}", macroF1=f"{f1:.3f}")
    return {
        "loss": total_loss / n,
        "macro_f1": total_f1 / n,
        "per_class_f1": (per_class / n).tolist(),
    }


def plot_curves(history, out_path):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(history["train"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["loss"] for h in history["train"]], "-o", label="train")
    axes[0].plot(epochs, [h["loss"] for h in history["val"]], "-s", label="val")
    axes[0].set_title("Damage loss (weighted CE + focal)")
    axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(epochs, [h["macro_f1"] for h in history["train"]], "-o", label="train")
    axes[1].plot(epochs, [h["macro_f1"] for h in history["val"]], "-s", label="val")
    axes[1].set_title("Macro F1 (4 damage classes)")
    axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"Stage 2: Siamese U-Net damage head (val = held-out {HOLDOUT_DISASTER})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--stage1-ckpt", type=str, default=str(CHECKPOINTS / "loc.pt"))
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="resume from outputs/checkpoints/dmg_last.pt if present")
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_pairs = list_pairs()
    train_pairs, val_pairs = split_pairs(all_pairs)
    if args.limit_train:
        train_pairs = train_pairs[: args.limit_train]
    print(f"train={len(train_pairs)} pairs, val={len(val_pairs)} pairs (holdout={HOLDOUT_DISASTER})")

    ds_train = XBDDataset(train_pairs, stage="dmg", crop=args.crop, train=True)
    ds_val = XBDDataset(val_pairs, stage="dmg", crop=args.crop, train=False)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = SiameseUNet(classes=5).to(device)
    weights = compute_class_weights().to(device)
    loss_fn = FocalCE(weight=weights, gamma=args.gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train": [], "val": []}
    best_f1 = -1.0
    start_epoch = 1
    log_path = LOGS / "train_dmg.json"
    best_ckpt = CHECKPOINTS / "dmg.pt"
    last_ckpt = CHECKPOINTS / "dmg_last.pt"

    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_f1 = state["best_f1"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {last_ckpt}: completed epoch {state['epoch']}, best val macroF1 so far={best_f1:.4f}")
        if start_epoch > args.epochs:
            print(f"Nothing to do: already trained {state['epoch']}/{args.epochs} epochs.")
            plot_curves(history, FIGS / "dmg_training_curves.png")
            return
    else:
        if args.resume:
            print(f"--resume requested but {last_ckpt} not found; starting from scratch.")
        if Path(args.stage1_ckpt).exists():
            model.load_stage1_encoder(args.stage1_ckpt)
        else:
            print(f"[warn] Stage-1 checkpoint not found at {args.stage1_ckpt}; using ImageNet-only init.")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        m_train = run_epoch(model, dl_train, device, optimizer, loss_fn, train=True)
        m_val = run_epoch(model, dl_val, device, optimizer, loss_fn, train=False)
        history["train"].append(m_train)
        history["val"].append(m_val)
        per = ", ".join(f"{n}={f:.3f}" for n, f in zip(DAMAGE_NAMES[1:], m_val["per_class_f1"]))
        print(f"  train  loss={m_train['loss']:.4f}  macroF1={m_train['macro_f1']:.4f}")
        print(f"  val    loss={m_val['loss']:.4f}  macroF1={m_val['macro_f1']:.4f}  [{per}]")
        log_path.write_text(json.dumps({"args": vars(args), "history": history}, indent=2))

        if m_val["macro_f1"] > best_f1:
            best_f1 = m_val["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": best_f1, "args": vars(args)}, best_ckpt)
            print(f"  -> new best, saved to {best_ckpt} (val_macro_f1={best_f1:.4f})")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_f1": best_f1,
            "args": vars(args),
        }, last_ckpt)

    print(f"\nTotal training time: {(time.time() - t0)/60:.1f} min  |  best val macro F1 = {best_f1:.4f}")
    plot_curves(history, FIGS / "dmg_training_curves.png")


if __name__ == "__main__":
    main()
