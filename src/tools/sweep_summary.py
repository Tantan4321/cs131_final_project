"""Aggregate outputs/logs/train_dmg*.json into a sortable comparison table.

Prints to stdout and (with --csv) writes outputs/sweep_summary.csv. Optionally
plots a bar chart of best val macro-F1 per run with --plot.

Usage:
    python -m src.tools.sweep_summary
    python -m src.tools.sweep_summary --csv --plot
"""
import argparse
import json

import numpy as np

from src.paths import FIGS, LOGS, OUTPUTS, ensure_dirs


def _load_runs():
    runs = []
    for f in sorted(LOGS.glob("train_dmg*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        val = data.get("history", {}).get("val", [])
        if not val:
            continue
        f1s = [v["macro_f1"] for v in val]
        best_idx = int(np.argmax(f1s))
        a = data.get("args", {})
        name = f.stem.replace("train_dmg_", "").replace("train_dmg", "default")
        per = val[best_idx].get("per_class_f1", [])
        runs.append({
            "name": name,
            "lr": a.get("lr"),
            "gamma": a.get("gamma"),
            "weights": a.get("weights", "inv-freq"),
            "encoder": a.get("encoder", "resnet34"),
            "batch": a.get("batch_size"),
            "epochs": len(val),
            "best_epoch": best_idx + 1,
            "best_f1": f1s[best_idx],
            "per_class": per,
        })
    return runs


def _fmt(v, w):
    return f"{v:>{w}}" if v is not None else " " * w


def print_table(runs):
    if not runs:
        print("No runs found in outputs/logs/. Run `python -m src.train_dmg ...` first.")
        return
    runs.sort(key=lambda r: r["best_f1"], reverse=True)
    header = f"{'Run':<22} {'LR':>8} {'γ':>4} {'weights':<14} {'encoder':<14} {'bs':>3} {'best F1':>8} {'best ep':>7} {'/N':<4}"
    print(header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r['name']:<22} "
            f"{r['lr']:>8.0e} "
            f"{r['gamma']:>4.1f} "
            f"{r['weights']:<14} "
            f"{r['encoder']:<14} "
            f"{_fmt(r['batch'], 3)} "
            f"{r['best_f1']:>8.4f} "
            f"{r['best_epoch']:>7} "
            f"/{r['epochs']:<3}"
        )

    print()
    print("Per-class F1 at the best epoch (no-damage / minor / major / destroyed):")
    for r in runs:
        per = "  ".join(f"{x:.3f}" for x in r["per_class"])
        print(f"  {r['name']:<22} {per}")


def write_csv(runs, out_path):
    import csv
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "lr", "gamma", "weights", "encoder", "batch", "epochs",
                    "best_epoch", "best_f1", "f1_no_damage", "f1_minor", "f1_major", "f1_destroyed"])
        for r in runs:
            per = list(r["per_class"]) + [None] * (4 - len(r["per_class"]))
            w.writerow([r["name"], r["lr"], r["gamma"], r["weights"], r["encoder"],
                        r["batch"], r["epochs"], r["best_epoch"], r["best_f1"], *per])
    print(f"Wrote {out_path}")


def plot_bars(runs, out_path):
    import matplotlib.pyplot as plt
    runs = sorted(runs, key=lambda r: r["best_f1"])
    names = [r["name"] for r in runs]
    f1s = [r["best_f1"] for r in runs]
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.45), 4))
    bars = ax.barh(names, f1s, color="steelblue")
    for b, v in zip(bars, f1s):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlim(0, max(f1s) * 1.15)
    ax.set_xlabel("Best val macro-F1")
    ax.set_title("Stage 2 hyperparameter sweep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write outputs/sweep_summary.csv")
    ap.add_argument("--plot", action="store_true", help="also write outputs/figs/sweep_summary.png")
    args = ap.parse_args()

    ensure_dirs()
    runs = _load_runs()
    print_table(runs)
    if args.csv:
        write_csv(runs, OUTPUTS / "sweep_summary.csv")
    if args.plot and runs:
        plot_bars(runs, FIGS / "sweep_summary.png")


if __name__ == "__main__":
    main()
