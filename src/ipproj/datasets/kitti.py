"""KITTI object-detection and semantic-segmentation loading (clean images + native GT).

Optical flow is handled separately in kitti_flow.py - it needs consecutive
frame pairs, a different loading concern from these single-image splits.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ipproj import config
from ipproj.datasets.download_utils import download_and_extract
from ipproj.datasets.split_utils import train_val_test_split


@dataclass
class DetectionSample:
    image_path: Path
    boxes: np.ndarray  # (N, 4) xyxy pixel coords
    classes: list  # len N, values from config.KITTI_DETECTION_CLASSES
    ignore_boxes: np.ndarray  # (M, 4) xyxy "DontCare" boxes - excluded from training, kept for eval


@dataclass
class SegmentationSample:
    image_path: Path
    mask_path: Path


def ensure_object_detection() -> Path:
    root = config.KITTI_ROOT / "object_detection"
    download_and_extract(config.KITTI_URLS["object_images"], root, "training/image_2")
    download_and_extract(config.KITTI_URLS["object_labels"], root, "training/label_2")
    return root


def ensure_semantics() -> Path:
    root = config.KITTI_ROOT / "semantics"
    download_and_extract(config.KITTI_URLS["semantics"], root, "training/image_2")
    return root


def parse_kitti_label_file(label_path: Path) -> DetectionSample:
    boxes, classes, ignore_boxes = [], [], []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        obj_type = fields[0]
        bbox = np.array(fields[4:8], dtype=np.float32)
        if obj_type == config.KITTI_IGNORE_LABEL:
            ignore_boxes.append(bbox)
        elif obj_type in config.KITTI_DETECTION_CLASSES:
            boxes.append(bbox)
            classes.append(obj_type)
        # Any other raw KITTI type is silently skipped: some KITTI releases
        # use a slightly larger raw label set than config.KITTI_DETECTION_CLASSES.

    image_path = label_path.parents[1] / "image_2" / f"{label_path.stem}.png"
    return DetectionSample(
        image_path=image_path,
        boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        classes=classes,
        ignore_boxes=np.array(ignore_boxes, dtype=np.float32).reshape(-1, 4),
    )


def load_object_detection_subset(n: int = config.OBJECT_DETECTION_SUBSET_SIZE) -> dict:
    """Returns {"train": [...], "val": [...], "test": [...]} of DetectionSample."""
    root = ensure_object_detection()
    label_paths = sorted((root / "training" / "label_2").glob("*.txt"))[:n]
    samples = [parse_kitti_label_file(p) for p in label_paths]
    return train_val_test_split(samples)


def load_semantic_segmentation_subset(n: int = config.SEMANTIC_SEGMENTATION_SUBSET_SIZE) -> dict:
    """Returns {"train": [...], "val": [...], "test": [...]} of SegmentationSample."""
    root = ensure_semantics()
    image_dir = root / "training" / "image_2"
    mask_dir = root / "training" / "semantic"
    image_paths = sorted(image_dir.glob("*.png"))[:n]
    samples = [SegmentationSample(image_path=p, mask_path=mask_dir / p.name) for p in image_paths]
    return train_val_test_split(samples)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_segmentation_mask(path: Path) -> np.ndarray:
    """Reads a KITTI Semantics GT mask and remaps Cityscapes' raw 34-class
    label IDs to the 19-class "trainId" taxonomy (see
    config.CITYSCAPES_ID_TO_TRAINID) that tasks.semantic_segmentation's model
    and metric operate in - the raw ids can't be compared against trainId
    predictions directly."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read segmentation mask at {path}")
    if raw.ndim == 3:
        # cv2.imread(..., IMREAD_UNCHANGED) is inconsistent across OpenCV
        # versions about squeezing single-channel PNGs: some builds return
        # (H, W), others (H, W, 1) - opencv-python is unpinned in
        # requirements.txt, so which one you get depends on the machine.
        # KITTI's semantic masks are single-channel label-id images, so
        # squeeze the trailing dim back to 2D rather than trusting either
        # behavior blindly.
        if raw.shape[2] != 1:
            raise ValueError(f"Expected a single-channel label mask at {path}, got shape {raw.shape}")
        raw = raw[:, :, 0]
    lut = np.full(256, config.SEGFORMER_IGNORE_INDEX, dtype=np.uint8)
    for raw_id, train_id in config.CITYSCAPES_ID_TO_TRAINID.items():
        lut[raw_id] = train_id
    return lut[raw]
