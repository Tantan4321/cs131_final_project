from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Training-essential subset (committed to git, uploaded to Colab):
#   train/xBD/{disaster}/images/*.png   pre + post images
#   train/cache/loc/{disaster}/*.npy    3-class boundary-aware targets
#   train/cache/dmg/{disaster}/*.npy    5-class damage targets
DATASET = ROOT / "train"
XBD = DATASET / "xBD"
CACHE_LOC = DATASET / "cache" / "loc"
CACHE_DMG = DATASET / "cache" / "dmg"

# Reference data (originals + per-disaster labels/targets), used by one-time
# preprocessing and visualization. Lives in dataset/train/ so train/ stays
# minimal for upload. Mirrors the same internal shape as train/.
REF = ROOT / "dataset" / "train"
TRAIN_IMG = REF / "images"          # flat original xBD download
TRAIN_LBL = REF / "labels"
TRAIN_TGT = REF / "targets"
XBD_REF = REF / "xBD"               # per-disaster labels/, targets/
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
LOGS = OUTPUTS / "logs"
FIGS = OUTPUTS / "figs"

DAMAGE_MAP = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
DAMAGE_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
DAMAGE_COLORS = [(0, 0, 0), (0, 200, 0), (255, 215, 0), (255, 120, 0), (220, 20, 20)]

DISASTERS = [
    "guatemala-volcano", "hurricane-florence", "hurricane-harvey",
    "hurricane-matthew", "hurricane-michael", "mexico-earthquake",
    "midwest-flooding", "palu-tsunami", "santa-rosa-wildfire", "socal-fire",
]

HOLDOUT_DISASTER = "santa-rosa-wildfire"


def ensure_dirs():
    for d in (XBD, CACHE_LOC, CACHE_DMG, OUTPUTS, CHECKPOINTS, LOGS, FIGS):
        d.mkdir(parents=True, exist_ok=True)
