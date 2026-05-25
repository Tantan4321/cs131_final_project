"""Group the flat xBD train/{images,labels,targets} into per-disaster subfolders via hardlinks.

Hardlinks have zero disk overhead and work on Windows NTFS for files on the same volume.
Skips files already linked. Safe to re-run.
"""
import argparse
import os
import re
from collections import defaultdict

from tqdm import tqdm

from src.paths import TRAIN_IMG, TRAIN_LBL, TRAIN_TGT, XBD, XBD_REF, ensure_dirs

NAME_RE = re.compile(r"^([a-z\-]+?)_(\d+)_(pre|post)_disaster\.")


def _link(src, dst):
    if dst.exists():
        return False
    try:
        os.link(src, dst)
    except OSError:
        # fallback (different volume / permission) -> copy
        import shutil
        shutil.copy2(src, dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_dirs()

    files_by_disaster = defaultdict(lambda: {"images": [], "labels": [], "targets": []})
    for f in TRAIN_IMG.iterdir():
        m = NAME_RE.match(f.name)
        if not m:
            continue
        files_by_disaster[m.group(1)]["images"].append(f.name)
    for f in TRAIN_LBL.iterdir():
        m = NAME_RE.match(f.name)
        if not m:
            continue
        files_by_disaster[m.group(1)]["labels"].append(f.name)
    if TRAIN_TGT.exists():
        for f in TRAIN_TGT.iterdir():
            m = NAME_RE.match(f.name)
            if not m:
                continue
            files_by_disaster[m.group(1)]["targets"].append(f.name)

    print(f"Found {len(files_by_disaster)} disasters in {TRAIN_IMG}")
    for d in sorted(files_by_disaster):
        c = files_by_disaster[d]
        print(f"  {d:25s} images={len(c['images']):4d} labels={len(c['labels']):4d} targets={len(c['targets']):4d}")

    if args.dry_run:
        return

    # images -> train/xBD/{disaster}/images/   (training essential)
    # labels, targets -> dataset/train/xBD/{disaster}/{sub}/   (reference only)
    routes = {
        "images":  (TRAIN_IMG, XBD),
        "labels":  (TRAIN_LBL, XBD_REF),
        "targets": (TRAIN_TGT, XBD_REF),
    }

    created = 0
    for disaster in tqdm(sorted(files_by_disaster), desc="linking"):
        for sub, (src_dir, dest_root) in routes.items():
            out_dir = dest_root / disaster / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in files_by_disaster[disaster][sub]:
                if _link(src_dir / name, out_dir / name):
                    created += 1

    print(f"Created {created} new hardlinks ({XBD} for images, {XBD_REF} for labels/targets)")


if __name__ == "__main__":
    main()
