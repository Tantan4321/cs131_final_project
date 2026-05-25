"""Compute per-disaster, per-damage-class statistics from the post-disaster JSONs.

Outputs:
  outputs/dataset_stats.csv               long-format CSV
  outputs/figs/class_distribution.png     stacked bar chart of building count
                                          and pixel area per disaster x class
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from tqdm import tqdm

from src.paths import DAMAGE_MAP, FIGS, OUTPUTS, XBD_REF, ensure_dirs

CLASS_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
CLASS_COLORS = {
    "no-damage": "#2ca02c",
    "minor-damage": "#ffd54f",
    "major-damage": "#ff7f0e",
    "destroyed": "#d62728",
    "un-classified": "#777777",
}


def main():
    ensure_dirs()

    if not XBD_REF.exists() or not any(XBD_REF.iterdir()):
        raise SystemExit(f"{XBD_REF} is empty. Run `python -m src.preprocess.group_by_disaster` first.")

    rows = []
    counts = defaultdict(lambda: defaultdict(int))   # [disaster][subtype] -> count
    areas = defaultdict(lambda: defaultdict(float))  # [disaster][subtype] -> pixel area

    disasters = sorted(d.name for d in XBD_REF.iterdir() if d.is_dir())
    print(f"Scanning {len(disasters)} disasters")

    for disaster in disasters:
        lbl_dir = XBD_REF / disaster / "labels"
        for j in tqdm(sorted(lbl_dir.glob("*_post_disaster.json")), desc=disaster, leave=False):
            data = json.loads(j.read_text(encoding="utf-8"))
            for feat in data.get("features", {}).get("xy", []):
                try:
                    g = wkt.loads(feat["wkt"])
                except (GEOSException, ValueError):
                    continue
                if not isinstance(g, Polygon) or g.is_empty:
                    continue
                subtype = feat.get("properties", {}).get("subtype") or "un-classified"
                counts[disaster][subtype] += 1
                areas[disaster][subtype] += float(g.area)

    for disaster in disasters:
        for subtype in CLASS_ORDER:
            rows.append({
                "disaster": disaster,
                "subtype": subtype,
                "n_buildings": counts[disaster][subtype],
                "pixel_area": areas[disaster][subtype],
            })
    df = pd.DataFrame(rows)
    csv_path = OUTPUTS / "dataset_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    totals_count = df.groupby("subtype")["n_buildings"].sum().reindex(CLASS_ORDER).fillna(0)
    totals_area = df.groupby("subtype")["pixel_area"].sum().reindex(CLASS_ORDER).fillna(0)
    print("\nGlobal class distribution:")
    for c in CLASS_ORDER:
        n = int(totals_count[c])
        pct = 100 * totals_area[c] / totals_area.sum() if totals_area.sum() else 0
        print(f"  {c:15s}  count={n:7d}   pixel-area={int(totals_area[c]):>11d} ({pct:5.2f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    pivot_n = df.pivot(index="disaster", columns="subtype", values="n_buildings").reindex(columns=CLASS_ORDER).fillna(0)
    pivot_a = df.pivot(index="disaster", columns="subtype", values="pixel_area").reindex(columns=CLASS_ORDER).fillna(0)

    bottoms = np.zeros(len(pivot_n))
    for c in CLASS_ORDER:
        ax1.bar(pivot_n.index, pivot_n[c], bottom=bottoms, color=CLASS_COLORS[c], label=c, edgecolor="white", linewidth=0.4)
        bottoms = bottoms + pivot_n[c].values
    ax1.set_title("Buildings per disaster, by damage class")
    ax1.set_ylabel("Building count")
    ax1.tick_params(axis="x", rotation=45)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")
    ax1.legend(loc="upper right", fontsize=8)

    bottoms = np.zeros(len(pivot_a))
    for c in CLASS_ORDER:
        ax2.bar(pivot_a.index, pivot_a[c] / 1e6, bottom=bottoms / 1e6, color=CLASS_COLORS[c], label=c, edgecolor="white", linewidth=0.4)
        bottoms = bottoms + pivot_a[c].values
    ax2.set_title("Pixel area per disaster (millions of px)")
    ax2.set_ylabel("Mpx")
    ax2.tick_params(axis="x", rotation=45)
    for lbl in ax2.get_xticklabels():
        lbl.set_ha("right")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("xBD train: severe class imbalance motivates focal/weighted loss", fontsize=12)
    fig.tight_layout()
    out = FIGS / "class_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
