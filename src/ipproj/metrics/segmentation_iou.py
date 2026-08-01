"""Semantic-segmentation IoU, wrapping torchmetrics (MulticlassJaccardIndex)
to avoid hand-rolled per-class IoU accumulation bugs."""
from torchmetrics.classification import MulticlassJaccardIndex


def new_metric(num_classes: int) -> MulticlassJaccardIndex:
    return MulticlassJaccardIndex(num_classes=num_classes, average=None)  # per-class IoU
