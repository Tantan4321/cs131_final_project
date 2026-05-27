"""Side-by-side: our 3-class boundary-aware encoding vs. the baseline shrink-N-pixels mask.

The script automatically picks the densest urban tile in the train set (most polygons),
generates both encodings, and renders a comparison with a zoomed inset highlighting
how adjacent buildings are separated.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from src.paths import FIGS, XBD, XBD_REF, ensure_dirs
from src.preprocess.encode_targets import encode_loc

H = W = 1024


def find_densest_tile(min_polys=80):
    best = None
    best_n = 0
    for disaster_dir in sorted(XBD_REF.iterdir()):
        if not disaster_dir.is_dir():
            continue
        for j in (disaster_dir / "labels").glob("*_pre_disaster.json"):
            data = json.loads(j.read_text(encoding="utf-8"))
            n = len(data.get("features", {}).get("xy", []))
            if n > best_n:
                best_n = n
                best = (disaster_dir.name, j.stem.replace("_pre_disaster", ""))
    print(f"Densest training tile: {best} with {best_n} polygons")
    return best


def baseline_mask_shrink(polys, border=2):
    """Replicates the baseline mask_polygons_together_with_border logic.

    Shrinks every polygon by `border` pixels toward its centroid, then fills with
    cv2.fillPoly and zeroes out overlapping pixels.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polys:
        cx, cy = poly.centroid.coords[0]
        verts = []
        for x, y in poly.exterior.coords:
            x = x + border if x < cx else (x - border if x > cx else x)
            y = y + border if y < cy else (y - border if y > cy else y)
            verts.append([x, y])
        verts = np.array(verts, np.int32)
        blank = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(blank, [verts], 1)
        mask = mask + blank
    mask[mask > 1] = 0  # zero overlaps
    return mask


def load_polys(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    polys = []
    for feat in data.get("features", {}).get("xy", []):
        try:
            g = wkt.loads(feat["wkt"])
        except (GEOSException, ValueError):
            continue
        if isinstance(g, Polygon) and not g.is_empty:
            polys.append(g)
    return polys


def colorize(mask, palette):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for v, c in palette.items():
        rgb[mask == v] = c
    return rgb


def draw_inset(ax_src, ax_dst, image, x, y, w):
    ax_dst.imshow(image[y:y + w, x:x + w])
    ax_dst.set_xticks([]); ax_dst.set_yticks([])
    ax_src.add_patch(Rectangle((x, y), w, w, fill=False, edgecolor="yellow", linewidth=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disaster", default=None)
    ap.add_argument("--image-id", default=None)
    ap.add_argument("--inset-xy", nargs=2, type=int, default=None, help="top-left of inset")
    ap.add_argument("--inset-size", type=int, default=256)
    args = ap.parse_args()

    ensure_dirs()

    if args.disaster and args.image_id:
        disaster, image_id = args.disaster, args.image_id
    else:
        picked = find_densest_tile()
        if picked is None:
            raise SystemExit("No labeled tiles found.")
        disaster, image_id = picked

    img_path = XBD / disaster / "images" / f"{image_id}_pre_disaster.png"
    json_path = XBD_REF / disaster / "labels" / f"{image_id}_pre_disaster.json"
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    polys = load_polys(json_path)
    print(f"Tile has {len(polys)} polygons")

    base_mask = baseline_mask_shrink(polys, border=2)
    our_mask = encode_loc(json_path)

    base_rgb = colorize(base_mask, {1: (40, 200, 40)})
    our_rgb = colorize(our_mask, {1: (40, 200, 40), 2: (230, 30, 30)})

    # Pick an inset region that has many buildings: by default ~ center of mass of polygons
    if args.inset_xy:
        ix, iy = args.inset_xy
    else:
        ys, xs = np.where(base_mask > 0)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            iw = args.inset_size
            ix = max(0, min(W - iw, cx - iw // 2))
            iy = max(0, min(H - iw, cy - iw // 2))
        else:
            ix, iy = 400, 400
    iw = args.inset_size

    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.3], hspace=0.18, wspace=0.04)

    title_kw = dict(fontsize=12, pad=4)
    zoom_kw = dict(fontsize=11, pad=4)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img)
    ax0.set_title("Pre-disaster", **title_kw); ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(img); ax1.imshow(base_rgb, alpha=0.55)
    ax1.set_title("Shrink-2px mask", **title_kw); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(img); ax2.imshow(our_rgb, alpha=0.55)
    ax2.set_title("3-class encoding", **title_kw); ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 3]); ax3.imshow(our_rgb)
    ax3.set_title("3-class target", **title_kw); ax3.axis("off")

    for ax in (ax1, ax2):
        ax.add_patch(Rectangle((ix, iy), iw, iw, fill=False, edgecolor="yellow", linewidth=2))

    axz0 = fig.add_subplot(gs[1, 0]); axz0.imshow(img[iy:iy+iw, ix:ix+iw])
    axz0.set_title("inset: pre", **zoom_kw); axz0.set_xticks([]); axz0.set_yticks([])
    axz1 = fig.add_subplot(gs[1, 1])
    axz1.imshow(img[iy:iy+iw, ix:ix+iw]); axz1.imshow(base_rgb[iy:iy+iw, ix:ix+iw], alpha=0.55)
    axz1.set_title("inset: shrink-2px", **zoom_kw); axz1.set_xticks([]); axz1.set_yticks([])
    axz2 = fig.add_subplot(gs[1, 2])
    axz2.imshow(img[iy:iy+iw, ix:ix+iw]); axz2.imshow(our_rgb[iy:iy+iw, ix:ix+iw], alpha=0.55)
    axz2.set_title("inset: 3-class", **zoom_kw); axz2.set_xticks([]); axz2.set_yticks([])
    axz3 = fig.add_subplot(gs[1, 3]); axz3.imshow(our_rgb[iy:iy+iw, ix:ix+iw])
    axz3.set_title("inset: target", **zoom_kw); axz3.set_xticks([]); axz3.set_yticks([])

    base_fg = int((base_mask > 0).sum())
    our_fg = int((our_mask >= 1).sum())
    gain_pct = 100 * (our_fg - base_fg) / max(base_fg, 1)
    fig.suptitle(
        f"Tile: {disaster}/{image_id}    ·    "
        f"foreground px: shrink-2px {base_fg:,}  vs.  3-class {our_fg:,}  "
        f"({gain_pct:+.0f}%)",
        fontsize=12, y=0.995,
    )
    out = FIGS / "encoding_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
