"""Render Stage-1 U-Net predictions on held-out tiles next to ground truth.

Loads outputs/checkpoints/loc.pt, runs inference on N tiles from the val disaster,
and writes an N-row grid: input | GT mask | predicted mask | overlay.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.xbd_dataset import XBDDataset, list_pairs, split_pairs
from src.paths import CHECKPOINTS, FIGS, HOLDOUT_DISASTER, ensure_dirs
from src.train_loc import make_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm(t):
    img = t.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def colorize(mask_np):
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    rgb[mask_np == 1] = (40, 200, 40)
    rgb[mask_np == 2] = (230, 30, 30)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CHECKPOINTS / "loc.pt"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt} (epoch={ckpt.get('epoch')} val_f1={ckpt.get('val_f1'):.4f})")

    _, val_pairs = split_pairs(list_pairs())
    val_pairs = val_pairs[: args.n]
    print(f"Inference on {len(val_pairs)} tiles from held-out disaster: {HOLDOUT_DISASTER}")

    ds = XBDDataset(val_pairs, stage="loc", crop=args.crop, train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for i, batch in enumerate(dl):
        pre = batch["pre"].to(device)
        gt = batch["loc_mask"][0].cpu().numpy()
        with torch.no_grad():
            logits = model(pre)
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        img = denorm(pre[0])
        gt_rgb = colorize(gt)
        pred_rgb = colorize(pred)

        axes[i, 0].imshow(img); axes[i, 0].set_title(f"{batch['disaster'][0]}/{batch['image_id'][0]}", fontsize=9); axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_rgb); axes[i, 1].set_title("ground truth (3-class)", fontsize=9); axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_rgb); axes[i, 2].set_title("prediction (argmax)", fontsize=9); axes[i, 2].axis("off")
        axes[i, 3].imshow(img); axes[i, 3].imshow(pred_rgb, alpha=0.55); axes[i, 3].set_title("overlay", fontsize=9); axes[i, 3].axis("off")

    fig.suptitle(f"Stage-1 U-Net predictions on held-out {HOLDOUT_DISASTER} (val F1={ckpt.get('val_f1', 0):.3f})", fontsize=12)
    fig.tight_layout()
    out = FIGS / "loc_predictions.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
