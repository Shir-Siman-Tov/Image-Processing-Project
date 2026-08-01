"""KITTI Flow 2015 loading: consecutive frame pairs + GT flow.

Kept separate from kitti.py's object-detection/semantics loaders: classical
optical flow (Lucas-Kanade/Farneback) needs true consecutive video frames,
which only this split provides - the object-detection and semantics splits
are independent, non-consecutive images and must never be substituted in.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ipproj import config
from ipproj.datasets.download_utils import download_and_extract
from ipproj.datasets.split_utils import train_val_test_split


@dataclass
class FlowSample:
    frame1_path: Path
    frame2_path: Path
    flow_gt_path: Path  # 16-bit 3-channel PNG, KITTI devkit encoding


def ensure_scene_flow() -> Path:
    root = config.KITTI_ROOT / "scene_flow"
    download_and_extract(config.KITTI_URLS["scene_flow"], root, "training/image_2")
    return root


def load_optical_flow_subset(n: int = config.OPTICAL_FLOW_SUBSET_SIZE) -> dict:
    """Returns {"train": [...], "val": [...], "test": [...]} of FlowSample.

    Uses flow_occ (GT over all pixels, including occluded regions) as the
    reference - the standard KITTI evaluation protocol. flow_noc (non-occluded
    only) is available under the same root if a stricter comparison is needed.
    """
    root = ensure_scene_flow()
    image_dir = root / "training" / "image_2"
    flow_dir = root / "training" / "flow_occ"

    frame1_paths = sorted(image_dir.glob("*_10.png"))[:n]
    samples = [
        FlowSample(
            frame1_path=p,
            frame2_path=p.with_name(p.name.replace("_10.png", "_11.png")),
            flow_gt_path=flow_dir / p.name,
        )
        for p in frame1_paths
    ]
    return train_val_test_split(samples)


def read_kitti_flow_png(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode a KITTI devkit flow PNG into (flow[h,w,2] float32, valid[h,w] bool)."""
    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"Could not read KITTI flow GT at {path}")
    rgb = raw[:, :, ::-1].astype(np.float32)  # BGR -> RGB: channels become [u_raw, v_raw, valid_raw]
    flow_raw, valid_raw = rgb[:, :, :2], rgb[:, :, 2]
    flow = (flow_raw - 2**15) / 64.0
    return flow, valid_raw > 0
