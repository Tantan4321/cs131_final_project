"""PyTorch Dataset over the per-disaster xBD layout, using cached .npy targets."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.paths import CACHE_DMG, CACHE_LOC, HOLDOUT_DISASTER, XBD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_pairs():
    """Return list of (disaster, image_id) tuples for the train set."""
    pairs = []
    for disaster_dir in sorted(XBD.iterdir()):
        if not disaster_dir.is_dir():
            continue
        img_dir = disaster_dir / "images"
        for f in sorted(img_dir.glob("*_pre_disaster.png")):
            image_id = f.stem.replace("_pre_disaster", "")
            pairs.append((disaster_dir.name, image_id))
    return pairs


def split_pairs(pairs, holdout=HOLDOUT_DISASTER):
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


class XBDDataset(Dataset):
    def __init__(self, pairs, stage="loc", crop=512, train=True):
        assert stage in ("loc", "dmg", "both")
        self.pairs = pairs
        self.stage = stage
        self.train = train
        self.tf = _augment(crop, train)

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
        return np.load(p).astype(np.uint8)

    def __getitem__(self, idx):
        disaster, image_id = self.pairs[idx]
        pre = self._load_img(XBD / disaster / "images" / f"{image_id}_pre_disaster.png")

        kwargs = {"image": pre}
        if self.stage in ("dmg", "both"):
            post = self._load_img(XBD / disaster / "images" / f"{image_id}_post_disaster.png")
            kwargs["image_post"] = post
        if self.stage in ("loc", "both"):
            kwargs["loc_mask"] = self._load_mask(CACHE_LOC, disaster, image_id, "pre_disaster")
        if self.stage in ("dmg", "both"):
            kwargs["dmg_mask"] = self._load_mask(CACHE_DMG, disaster, image_id, "post_disaster")

        out = self.tf(**kwargs)
        sample = {"pre": out["image"], "disaster": disaster, "image_id": image_id}
        if "image_post" in out:
            sample["post"] = out["image_post"]
        if "loc_mask" in out:
            sample["loc_mask"] = out["loc_mask"].long()
        if "dmg_mask" in out:
            sample["dmg_mask"] = out["dmg_mask"].long()
        return sample
