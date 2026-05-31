"""PyTorch Dataset over the per-disaster xBD layout, using cached .npy targets."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.paths import (
    CACHE_DMG, CACHE_DMG_TEST, CACHE_LOC,
    DATASET, DISASTERS, OUTPUTS, TEST_ROOT, XBD, XBD_TEST,
)

DMG_CLASSES = (1, 2, 3, 4)   # no-damage, minor, major, destroyed (0 = bg)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_pairs(xbd_root=None):
    """Return (disaster, image_id) tuples from the per-disaster xBD layout.

    Defaults to the training set (train/xBD/). Pass xbd_root=XBD_TEST for test.
    """
    root = Path(xbd_root) if xbd_root else XBD
    pairs = []
    for disaster_dir in sorted(root.iterdir()):
        if not disaster_dir.is_dir():
            continue
        img_dir = disaster_dir / "images"
        for f in sorted(img_dir.glob("*_pre_disaster.png")):
            image_id = f.stem.replace("_pre_disaster", "")
            pairs.append((disaster_dir.name, image_id))
    return pairs


def list_test_pairs():
    """Return (disaster, image_id) tuples for the official xBD test split."""
    return list_pairs(xbd_root=XBD_TEST)


def split_pairs(pairs, holdout="santa-rosa-wildfire"):
    """Split pairs into (train, val) by held-out disaster. Kept for backward compat."""
    train = [p for p in pairs if p[0] != holdout]
    val = [p for p in pairs if p[0] == holdout]
    return train, val


def _augment(crop, train):
    if train:
        return A.Compose(
            [
                A.RandomCrop(crop, crop),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={
                "image_post": "image",
                "loc_mask": "mask",
                "dmg_mask": "mask",
            },
        )
    return A.Compose(
        [
            A.CenterCrop(crop, crop),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={
            "image_post": "image",
            "loc_mask": "mask",
            "dmg_mask": "mask",
        },
    )


def compute_tile_weights(pairs, dmg_classes=DMG_CLASSES, cache_file=None,
                         verbose=True, cache_dmg_root=None):
    """Per-tile sampling weights for balanced damage-class exposure.

    For each tile, weight = sum_c (pixels_of_class_c_in_tile / total_pixels_of_class_c).
    A tile that contains a large fraction of all destroyed pixels in the
    training set therefore gets a much higher weight than a tile that's
    almost all background. When used with WeightedRandomSampler the expected
    per-class pixel count in a batch becomes roughly equal across all
    damage classes -- no file duplication required.
    """
    cache_root = Path(cache_dmg_root) if cache_dmg_root else CACHE_DMG

    if cache_file and Path(cache_file).exists():
        cached = torch.load(cache_file, weights_only=False)
        if len(cached) == len(pairs):
            return cached
        print(f"[tile weights] cache size {len(cached)} != {len(pairs)} pairs, recomputing")

    n_classes = max(dmg_classes) + 1
    per_tile = np.zeros((len(pairs), n_classes), dtype=np.float64)
    iterator = tqdm(pairs, desc="tile weights", disable=not verbose)
    for i, (disaster, image_id) in enumerate(iterator):
        mask_path = cache_root / disaster / f"{image_id}_post_disaster.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path, allow_pickle=True)
        for c in dmg_classes:
            per_tile[i, c] = int((mask == c).sum())

    class_totals = per_tile.sum(axis=0)
    weights = np.zeros(len(pairs), dtype=np.float64)
    for c in dmg_classes:
        if class_totals[c] > 0:
            weights += per_tile[:, c] / class_totals[c]

    # Tiny baseline so tiles with no damage pixels still get sampled occasionally.
    baseline = max(weights.max(), 1.0) * 1e-4
    weights = np.maximum(weights, baseline)

    weights_t = torch.from_numpy(weights).double()
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights_t, cache_file)

    if verbose:
        wn = weights / weights.sum()
        top10 = np.argsort(wn)[-10:][::-1]
        bot_share = wn[wn.argsort()[: len(wn) // 2]].sum()
        print(f"  top 10 tiles account for {wn[top10].sum()*100:.1f}% of total sampling weight")
        print(f"  bottom half of tiles account for {bot_share*100:.2f}%")
    return weights_t


def make_balanced_sampler(pairs, generator=None, cache_file=None):
    """Drop-in WeightedRandomSampler for damage-class balancing."""
    weights = compute_tile_weights(pairs, cache_file=cache_file)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(pairs), replacement=True, generator=generator
    )


class XBDDataset(Dataset):
    def __init__(self, pairs, stage="loc", crop=512, train=True, data_root=None):
        """
        data_root: root of the processed split (defaults to train/).
                   Use TEST_ROOT for test-set inference.
        """
        assert stage in ("loc", "dmg", "both")
        self.pairs = pairs
        self.stage = stage
        self.train = train
        self.tf = _augment(crop, train)
        root = Path(data_root) if data_root else DATASET
        self.xbd = root / "xBD"
        self.cache_loc = root / "cache" / "loc"
        self.cache_dmg = root / "cache" / "dmg"

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, root, disaster, image_id, suffix):
        p = root / disaster / f"{image_id}_{suffix}.npy"
        if not p.exists():
            return np.zeros((1024, 1024), dtype=np.uint8)
        return np.load(p, allow_pickle=True).astype(np.uint8)

    def __getitem__(self, idx):
        disaster, image_id = self.pairs[idx]
        pre = self._load_img(self.xbd / disaster / "images" / f"{image_id}_pre_disaster.png")

        kwargs = {"image": pre}
        if self.stage in ("dmg", "both"):
            post = self._load_img(self.xbd / disaster / "images" / f"{image_id}_post_disaster.png")
            kwargs["image_post"] = post
        if self.stage in ("loc", "both"):
            kwargs["loc_mask"] = self._load_mask(self.cache_loc, disaster, image_id, "pre_disaster")
        if self.stage in ("dmg", "both"):
            kwargs["dmg_mask"] = self._load_mask(self.cache_dmg, disaster, image_id, "post_disaster")

        out = self.tf(**kwargs)
        sample = {"pre": out["image"], "disaster": disaster, "image_id": image_id}
        if "image_post" in out:
            sample["post"] = out["image_post"]
        if "loc_mask" in out:
            sample["loc_mask"] = out["loc_mask"].long()
        if "dmg_mask" in out:
            sample["dmg_mask"] = out["dmg_mask"].long()
        return sample
