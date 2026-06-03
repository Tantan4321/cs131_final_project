"""Render Stage-2 Siamese U-Net damage predictions on held-out tiles.

Loads outputs/checkpoints/dmg.pt, runs inference on N (pre, post) tile pairs
from the held-out val disaster, and writes an N-row grid:
    pre | post | GT damage mask | predicted damage mask | overlay on post
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.xbd_dataset import XBDDataset, list_test_pairs
from src.models.siamese_unet import SiameseUNet
from src.paths import CHECKPOINTS, DAMAGE_COLORS, DAMAGE_NAMES, FIGS, TEST_ROOT, ensure_dirs

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm(t):
    img = t.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def colorize(mask_np):
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(DAMAGE_COLORS):
        if cls == 0:
            continue  # background stays black
        rgb[mask_np == cls] = color
    return rgb


def add_legend(fig):
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=np.array(DAMAGE_COLORS[c]) / 255.0, label=DAMAGE_NAMES[c])
        for c in range(1, len(DAMAGE_NAMES))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CHECKPOINTS / "dmg.pt"))
    ap.add_argument("--n", type=int, default=4,
                    help="number of tiles to show (ignored when --indices is set)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--skip", type=int, nargs="*", default=[],
                    help="global test-list indices to exclude (when not using --indices)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit global test-list indices to visualise (overrides --n/--skip)")
    ap.add_argument("--out", default="dmg_predictions.png",
                    help="output filename written under outputs/figs/")
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseUNet(classes=5).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_f1 = ckpt.get("val_macro_f1", 0.0)
    print(f"Loaded {args.ckpt} (epoch={ckpt.get('epoch')} val_macro_f1={val_f1:.4f})")

    all_test = list_test_pairs()
    if args.indices is not None:
        selected = [all_test[i] for i in args.indices]
        print(f"Using explicit indices: {args.indices}")
    else:
        skipped = set(args.skip)
        selected = [p for i, p in enumerate(all_test) if i not in skipped][: args.n]
        if skipped:
            print(f"Skipped test indices: {sorted(skipped)}")
    print(f"Inference on {len(selected)} tile pairs from xBD test split")

    ds = XBDDataset(selected, stage="dmg", crop=args.crop, train=False, data_root=TEST_ROOT)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    fig, axes = plt.subplots(len(selected), 5, figsize=(20, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    for i, batch in enumerate(dl):
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        gt = batch["dmg_mask"][0].cpu().numpy()
        with torch.no_grad():
            logits = model(pre, post)
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        pre_img = denorm(pre[0])
        post_img = denorm(post[0])
        gt_rgb = colorize(gt)
        pred_rgb = colorize(pred)

        axes[i, 0].imshow(pre_img); axes[i, 0].set_title(f"pre  ({batch['disaster'][0]}/{batch['image_id'][0]})", fontsize=9); axes[i, 0].axis("off")
        axes[i, 1].imshow(post_img); axes[i, 1].set_title("post", fontsize=9); axes[i, 1].axis("off")
        axes[i, 2].imshow(gt_rgb); axes[i, 2].set_title("ground truth (5-class)", fontsize=9); axes[i, 2].axis("off")
        axes[i, 3].imshow(pred_rgb); axes[i, 3].set_title("prediction (argmax)", fontsize=9); axes[i, 3].axis("off")
        axes[i, 4].imshow(post_img); axes[i, 4].imshow(pred_rgb, alpha=0.6); axes[i, 4].set_title("prediction on post", fontsize=9); axes[i, 4].axis("off")

    fig.suptitle(f"Stage-2 Siamese U-Net damage predictions, xBD test split "
                 f"(val macro-F1={val_f1:.3f})", fontsize=12)
    add_legend(fig)
    fig.tight_layout()
    out = FIGS / args.out
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
