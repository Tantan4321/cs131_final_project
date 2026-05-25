"""Vectorized polygon -> raster encoding.

Two modes:
  --mode loc : reads PRE-disaster JSONs -> 3-class {bg=0, interior=1, boundary=2} uint8 mask
  --mode dmg : reads POST-disaster JSONs -> 5-class {bg, no-dmg, minor, major, destroyed} uint8 mask

Outputs cached as .npy under dataset/cache/{loc|dmg}/{disaster}/{stem}.npy.
Re-runs are no-ops on existing files unless --force.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon
from tqdm import tqdm

from src.paths import CACHE_DMG, CACHE_LOC, DAMAGE_MAP, XBD_REF, ensure_dirs

H = W = 1024
DILATE_ITERS = 2  # 2 iters of 3x3 cross -> ~2-pixel boundary band on each side


def _load_polygons(json_path):
    """Returns list[(Polygon, subtype_str_or_None)]."""
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


def encode_loc(json_path):
    """3-class {bg, interior, boundary} mask from a PRE-disaster JSON."""
    polys = _load_polygons(json_path)
    mask = np.zeros((H, W), dtype=np.uint8)
    if not polys:
        return mask
    interior = rasterize(
        ((p, 1) for p, _ in polys),
        out_shape=(H, W),
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    # Polygon exteriors as LineStrings -> 1px boundary, then dilate
    boundaries = [LineString(list(p.exterior.coords)) for p, _ in polys if p.exterior]
    edge = rasterize(
        ((b, 1) for b in boundaries),
        out_shape=(H, W),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    edge = binary_dilation(edge.astype(bool), iterations=DILATE_ITERS)
    mask[interior > 0] = 1
    mask[edge] = 2  # boundary class overrides interior at the edges
    return mask


def encode_dmg(json_path):
    """5-class damage mask from a POST-disaster JSON."""
    polys = _load_polygons(json_path)
    mask = np.zeros((H, W), dtype=np.uint8)
    if not polys:
        return mask
    # Sort by subtype so destroyed paints last (defensive against overlapping polygons)
    order = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3, None: -1}
    polys_sorted = sorted(polys, key=lambda x: order.get(x[1], -1))
    shapes = []
    for poly, subtype in polys_sorted:
        cls = DAMAGE_MAP.get(subtype)
        if cls is None:
            continue
        shapes.append((poly, cls))
    if not shapes:
        return mask
    return rasterize(shapes, out_shape=(H, W), fill=0, dtype=np.uint8, all_touched=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loc", "dmg"], required=True)
    ap.add_argument("--force", action="store_true", help="rewrite existing cached masks")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (debug)")
    args = ap.parse_args()

    ensure_dirs()

    if args.mode == "loc":
        json_glob = "*_pre_disaster.json"
        cache_root = CACHE_LOC
        encode = encode_loc
    else:
        json_glob = "*_post_disaster.json"
        cache_root = CACHE_DMG
        encode = encode_dmg

    if not XBD_REF.exists() or not any(XBD_REF.iterdir()):
        raise SystemExit(f"{XBD_REF} is empty. Run `python -m src.preprocess.group_by_disaster` first.")

    todo = []
    for disaster_dir in sorted(XBD_REF.iterdir()):
        if not disaster_dir.is_dir():
            continue
        lbl_dir = disaster_dir / "labels"
        if not lbl_dir.exists():
            continue
        out_dir = cache_root / disaster_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for j in sorted(lbl_dir.glob(json_glob)):
            out_path = out_dir / (j.stem + ".npy")
            if out_path.exists() and not args.force:
                continue
            todo.append((j, out_path))

    if args.limit:
        todo = todo[: args.limit]
    print(f"[encode_targets:{args.mode}] {len(todo)} files to process")

    n_empty = 0
    for j, out_path in tqdm(todo, unit="img"):
        mask = encode(j)
        if not mask.any():
            n_empty += 1
        np.save(out_path, mask)

    print(f"Done. {n_empty} masks had zero buildings.")


if __name__ == "__main__":
    main()
