"""Stage 1: train a 3-class U-Net (bg / interior / boundary) on pre-disaster tiles.

Holds out one full disaster as validation so val metrics reflect cross-disaster
generalization. Saves the best checkpoint by val foreground-F1, a training-log
JSON, and Fig 4 (loss + F1 curves).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.xbd_dataset import XBDDataset, list_pairs, split_pairs
from src.paths import CHECKPOINTS, FIGS, HOLDOUT_DISASTER, LOGS, ensure_dirs


def make_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )


def foreground_f1(pred_argmax, target, eps=1e-7):
    """F1 of building-foreground (interior + boundary collapsed) vs ground truth."""
    p = (pred_argmax >= 1).float()
    t = (target >= 1).float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return (2 * tp / (2 * tp + fp + fn + eps)).item()


def run_epoch(model, loader, device, optimizer, ce_loss, dice_loss, train):
    model.train(train)
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_f1 = 0.0
    n = 0
    pbar = tqdm(loader, desc="train" if train else "val ", leave=False)
    for batch in pbar:
        pre = batch["pre"].to(device, non_blocking=True)
        target = batch["loc_mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(pre)
            l_ce = ce_loss(logits, target)
            l_dice = dice_loss(logits, target)
            loss = l_ce + l_dice
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            f1 = foreground_f1(pred, target)
        bs = pre.size(0)
        total_loss += loss.item() * bs
        total_ce += l_ce.item() * bs
        total_dice += l_dice.item() * bs
        total_f1 += f1 * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.3f}", f1=f"{f1:.3f}")
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "dice": total_dice / n,
        "f1": total_f1 / n,
    }


def plot_curves(history, out_path):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(history["train"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes, ["loss", "dice", "f1"], ["Total loss (CE + Dice)", "Dice loss", "Foreground F1"]):
        ax.plot(epochs, [h[key] for h in history["train"]], "-o", label="train")
        ax.plot(epochs, [h[key] for h in history["val"]], "-s", label="val")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Stage 1: U-Net localization (val = held-out {HOLDOUT_DISASTER})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--boundary-weight", type=float, default=3.0)
    ap.add_argument("--limit-train", type=int, default=None, help="for fast smoke tests")
    ap.add_argument("--resume", action="store_true",
                    help="resume from outputs/checkpoints/loc_last.pt if present")
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_pairs = list_pairs()
    train_pairs, val_pairs = split_pairs(all_pairs)
    if args.limit_train:
        train_pairs = train_pairs[: args.limit_train]
    print(f"train={len(train_pairs)} pairs, val={len(val_pairs)} pairs (holdout={HOLDOUT_DISASTER})")

    ds_train = XBDDataset(train_pairs, stage="loc", crop=args.crop, train=True)
    ds_val = XBDDataset(val_pairs, stage="loc", crop=args.crop, train=False)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = make_model().to(device)
    ce_weight = torch.tensor([1.0, 1.0, args.boundary_weight], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train": [], "val": []}
    best_f1 = -1.0
    start_epoch = 1
    log_path = LOGS / "train_loc.json"
    best_ckpt = CHECKPOINTS / "loc.pt"
    last_ckpt = CHECKPOINTS / "loc_last.pt"

    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_f1 = state["best_f1"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {last_ckpt}: completed epoch {state['epoch']}, best val F1 so far={best_f1:.4f}")
        if start_epoch > args.epochs:
            print(f"Nothing to do: already trained {state['epoch']}/{args.epochs} epochs.")
            plot_curves(history, FIGS / "loc_training_curves.png")
            return
    elif args.resume:
        print(f"--resume requested but {last_ckpt} not found; starting from scratch.")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        m_train = run_epoch(model, dl_train, device, optimizer, ce_loss, dice_loss, train=True)
        m_val = run_epoch(model, dl_val, device, optimizer, ce_loss, dice_loss, train=False)
        history["train"].append(m_train)
        history["val"].append(m_val)
        print(f"  train  loss={m_train['loss']:.4f}  dice={m_train['dice']:.4f}  f1={m_train['f1']:.4f}")
        print(f"  val    loss={m_val['loss']:.4f}  dice={m_val['dice']:.4f}  f1={m_val['f1']:.4f}")
        log_path.write_text(json.dumps({"args": vars(args), "history": history}, indent=2))

        if m_val["f1"] > best_f1:
            best_f1 = m_val["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": best_f1, "args": vars(args)}, best_ckpt)
            print(f"  -> new best, saved to {best_ckpt} (val_f1={best_f1:.4f})")

        # Save full state every epoch for resume
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_f1": best_f1,
            "args": vars(args),
        }, last_ckpt)

    print(f"\nTotal training time: {(time.time() - t0)/60:.1f} min  |  best val F1 = {best_f1:.4f}")
    plot_curves(history, FIGS / "loc_training_curves.png")


if __name__ == "__main__":
    main()
