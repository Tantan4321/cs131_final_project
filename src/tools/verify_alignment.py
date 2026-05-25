"""Render 4 random pre/post pairs with polygon overlays and the cached 3-class mask.

A sanity check that the polygon -> raster pipeline is aligned end-to-end:
  - JSON polygons land on the right pixels
  - Damage colors line up between pre and post views
  - The cached 3-class mask matches the input image
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from src.paths import CACHE_LOC, DAMAGE_COLORS, DAMAGE_MAP, FIGS, XBD, XBD_REF, ensure_dirs


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_polygons(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = []
    for feat in data.get("features", {}).get("xy", []):
        try:
            geom = wkt.loads(feat["wkt"])
        except (GEOSException, ValueError):
            continue
        if not isinstance(geom, Polygon) or geom.is_empty:
            continue
        subtype = feat.get("properties", {}).get("subtype")
        out.append((geom, subtype))
    return out


def overlay_polys(ax, polys, color_by_subtype=False):
    for geom, subtype in polys:
        xs, ys = geom.exterior.xy
        if color_by_subtype and subtype in DAMAGE_MAP:
            c = np.array(DAMAGE_COLORS[DAMAGE_MAP[subtype]]) / 255.0
        else:
            c = np.array([1.0, 0.2, 0.2])
        ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True, fill=False,
                                edgecolor=c, linewidth=1.2))


def encoded_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = (40, 200, 40)   # interior
    rgb[mask == 2] = (230, 30, 30)   # boundary band
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of pairs to sample")
    ap.add_argument("--seed", type=int, default=131)
    args = ap.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)

    candidates = []
    for disaster_dir in sorted(XBD.iterdir()):
        if not disaster_dir.is_dir():
            continue
        for f in (disaster_dir / "images").glob("*_pre_disaster.png"):
            candidates.append((disaster_dir.name, f.stem.replace("_pre_disaster", "")))
    if not candidates:
        raise SystemExit("No images found. Run group_by_disaster first.")

    samples = rng.sample(candidates, args.n)
    print(f"Sampled {len(samples)} pairs across {len({s[0] for s in samples})} disasters")

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for i, (disaster, image_id) in enumerate(samples):
        pre = load_image(XBD / disaster / "images" / f"{image_id}_pre_disaster.png")
        post = load_image(XBD / disaster / "images" / f"{image_id}_post_disaster.png")
        pre_polys = load_polygons(XBD_REF / disaster / "labels" / f"{image_id}_pre_disaster.json")
        post_polys = load_polygons(XBD_REF / disaster / "labels" / f"{image_id}_post_disaster.json")
        mask_path = CACHE_LOC / disaster / f"{image_id}_pre_disaster.npy"
        mask = np.load(mask_path) if mask_path.exists() else np.zeros(pre.shape[:2], np.uint8)

        ax = axes[i, 0]; ax.imshow(pre); ax.set_title(f"PRE  ({disaster}/{image_id})", fontsize=9); ax.axis("off")
        ax = axes[i, 1]; ax.imshow(pre); overlay_polys(ax, pre_polys); ax.set_title(f"PRE + polygons ({len(pre_polys)} bldgs)", fontsize=9); ax.axis("off")
        ax = axes[i, 2]; ax.imshow(encoded_to_rgb(mask)); ax.set_title("3-class encoded mask\n(green=interior, red=boundary band)", fontsize=9); ax.axis("off")
        ax = axes[i, 3]; ax.imshow(post); overlay_polys(ax, post_polys, color_by_subtype=True); ax.set_title("POST + damage-colored polygons", fontsize=9); ax.axis("off")

    fig.suptitle("Polygon -> mask alignment across 4 random pre/post pairs", fontsize=12)
    fig.tight_layout()
    out = FIGS / "polygon_alignment.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
