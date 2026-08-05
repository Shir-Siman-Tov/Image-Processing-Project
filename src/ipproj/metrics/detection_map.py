"""Object-detection mAP/IoU, wrapping torchmetrics to avoid hand-rolled
IoU/mAP accumulation bugs (per the assignment PDF's own advice: "don't
write/generate long code, use AI and libraries")."""
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from ipproj import config


def new_metric() -> MeanAveragePrecision:
    return MeanAveragePrecision(class_metrics=True)


def suppress_dontcare_predictions(
    boxes: np.ndarray,
    classes: list,
    scores: np.ndarray,
    ignore_boxes: np.ndarray,
    iou_threshold: float = config.KITTI_IGNORE_IOU_THRESHOLD,
) -> tuple:
    """Drops predicted boxes that overlap a KITTI DontCare/ignore region above
    `iou_threshold`. Matches KITTI's own eval protocol: predictions on a
    DontCare region are excluded from scoring entirely, not counted as false
    positives."""
    if len(ignore_boxes) == 0 or len(boxes) == 0:
        return boxes, classes, scores

    ious = box_iou(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(ignore_boxes, dtype=torch.float32))
    keep = (ious.max(dim=1).values < iou_threshold).numpy()
    return boxes[keep], [c for c, k in zip(classes, keep) if k], scores[keep]


def to_torchmetrics_target(boxes: np.ndarray, classes: list) -> dict:
    class_ids = [config.KITTI_DETECTION_CLASSES.index(c) for c in classes]
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(class_ids, dtype=torch.int64),
    }


def to_torchmetrics_prediction(boxes: np.ndarray, classes: list, scores: np.ndarray) -> dict:
    class_ids = [config.KITTI_DETECTION_CLASSES.index(c) for c in classes]
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(class_ids, dtype=torch.int64),
        "scores": torch.as_tensor(scores, dtype=torch.float32),
    }


def per_class_ap_dict(metrics: dict, class_names: list = config.KITTI_DETECTION_CLASSES) -> dict:
    """Zips a `new_metric()`-produced `compute()` result's `map_per_class`/
    `classes` tensors into `{class_name: AP}`. torchmetrics only reports
    classes present in that call's ground truth, so callers comparing two
    results (e.g. distorted vs. restored) must align by class name rather
    than assuming identical order/coverage between calls."""
    return {
        class_names[i]: v
        for i, v in zip(metrics["classes"].tolist(), metrics["map_per_class"].tolist())
    }
