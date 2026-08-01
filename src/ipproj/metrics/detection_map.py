"""Object-detection mAP/IoU, wrapping torchmetrics to avoid hand-rolled
IoU/mAP accumulation bugs (per the assignment PDF's own advice: "don't
write/generate long code, use AI and libraries")."""
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ipproj import config


def new_metric() -> MeanAveragePrecision:
    return MeanAveragePrecision(class_metrics=True)


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
