"""Full-val evaluation: confusion matrix + per-class F1 for Stage-2 model.

Runs inference on all held-out val tiles (santa-rosa-wildfire) and writes:
  outputs/figs/dmg_confusion.png      row-normalized 4x4 confusion matrix
  outputs/logs/eval_dmg.json          raw numbers

Usage:
  python -m src.tools.eval_dmg                       # uses dmg_final.pt
  python -m src.tools.eval_dmg --ckpt dmg_enc_resnet34.pt
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.xbd_dataset import XBDDataset, list_test_pairs
from src.models.siamese_unet import SiameseUNet
from src.paths import CHECKPOINTS, DAMAGE_NAMES, FIGS, LOGS, TEST_ROOT, ensure_dirs

EPS = 1e-7
NUM_CLASSES = 5


def build_confusion_fast(model, dl, device):
    """Vectorized confusion matrix accumulation."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    model.eval()
    for batch in tqdm(dl, desc="eval"):
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        gt = batch["dmg_mask"].numpy().ravel()
        with torch.no_grad():
            pred = model(pre, post).argmax(dim=1).cpu().numpy().ravel()
        np.add.at(cm, (gt, pred), 1)
    return cm


def per_class_f1_from_cm(cm):
    f1s = []
    for c in range(1, NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        f1s.append(float(2 * tp / (2 * tp + fp + fn + EPS)))
    return f1s


def plot_confusion(cm, val_f1, out):
    labels = DAMAGE_NAMES[1:]
    cm_sub = cm[1:, 1:].astype(float)
    row_sums = cm_sub.sum(axis=1, keepdims=True)
    cm_norm = cm_sub / (row_sums + EPS)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(
        f"Damage classification — test confusion matrix\n"
        f"xBD test split (933 tiles)   macro-F1 = {val_f1:.3f}",
        fontsize=11,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", fontsize=11,
                color="white" if cm_norm[i, j] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CHECKPOINTS / "dmg_final.pt"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder = state.get("args", {}).get("encoder", "resnet34")
    model = SiameseUNet(encoder_name=encoder, classes=5).to(device)
    model.load_state_dict(state["model"])
    ckpt_f1 = state.get("val_macro_f1", 0.0)
    print(f"Loaded {args.ckpt}  encoder={encoder}  epoch={state.get('epoch')}  saved_val_f1={ckpt_f1:.4f}")

    val_pairs = list_test_pairs()
    print(f"Test tiles: {len(val_pairs)}")
    ds = XBDDataset(val_pairs, stage="dmg", crop=args.crop, train=False, data_root=TEST_ROOT)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    cm = build_confusion_fast(model, dl, device)

    f1s = per_class_f1_from_cm(cm)
    macro_f1 = float(np.mean(f1s))
    print(f"\nPer-class F1:")
    for name, f1 in zip(DAMAGE_NAMES[1:], f1s):
        print(f"  {name:15s}: {f1:.4f}")
    print(f"  {'macro':15s}: {macro_f1:.4f}")

    plot_confusion(cm, macro_f1, FIGS / "dmg_confusion.png")

    results = {
        "ckpt": args.ckpt,
        "encoder": encoder,
        "saved_val_macro_f1": ckpt_f1,
        "eval_macro_f1": macro_f1,
        "per_class_f1": dict(zip(DAMAGE_NAMES[1:], f1s)),
        "confusion_matrix": cm.tolist(),
    }
    log_path = LOGS / "eval_dmg.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
