"""YOLOv8 object-detection task (high-level, DL).

Course grounding: 2_DeepDetection/4021_BasicsOfObjectDetection, 4022_AdvancedObjectDetection.

A COCO-pretrained YOLO can't be evaluated against KITTI's own class taxonomy
directly - COCO has no Pedestrian/Cyclist/Van/Tram classes. The head must be
fine-tuned on KITTI's own classes (config.KITTI_DETECTION_CLASSES) first;
that fine-tuned checkpoint, not the raw COCO one, is what "baseline" means
for this task in 02_clean_baseline and everything downstream.
"""
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ipproj import config
from ipproj.datasets.kitti import read_image
from ipproj.metrics.detection_map import new_metric, to_torchmetrics_prediction, to_torchmetrics_target


def load_pretrained() -> YOLO:
    return YOLO(config.YOLO_BASE_WEIGHTS)


def checkpoint_path_for(run_name: str) -> Path:
    """Where fine_tune() writes/looks for `run_name`'s best.pt.

    Exposed so callers can check `.exists()` before doing any of the (also
    expensive) prep work fine_tune() needs, e.g. building the YOLO-format
    dataset - no point building it just to have fine_tune() throw it away.
    """
    return config.CHECKPOINT_ROOT / "yolo" / run_name / "weights" / "best.pt"


def fine_tune(data_yaml: Path, run_name: str, epochs: int = config.YOLO_FINE_TUNE_EPOCHS) -> Path:
    """Fine-tunes YOLO's head on config.KITTI_DETECTION_CLASSES.
    `data_yaml` comes from datasets.kitti_yolo_format.build_yolo_dataset().
    Returns the path to the resulting best.pt checkpoint.

    Skips training and returns the existing checkpoint if `run_name` was
    already fine-tuned before. CHECKPOINT_ROOT lives on Drive on Colab, so a
    runtime restart (which wipes the kernel but not Drive) would otherwise
    silently redo an expensive training run on every rerun.
    """
    checkpoint_path = checkpoint_path_for(run_name)
    if checkpoint_path.exists():
        return checkpoint_path

    model = load_pretrained()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        seed=config.RANDOM_SEED,
        project=str(config.CHECKPOINT_ROOT / "yolo"),
        name=run_name,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def predict(model: YOLO, image: np.ndarray, confidence: float = 0.25):
    """Returns (boxes[N,4] xyxy, classes: list[str], scores[N])."""
    result = model.predict(image, conf=confidence, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    classes = [model.names[i] for i in class_ids]
    return boxes, classes, scores


def evaluate(model: YOLO, samples: list) -> dict:
    """`samples`: list of kitti.DetectionSample. Returns per-class + overall mAP/IoU."""
    metric = new_metric()
    for sample in samples:
        image = read_image(sample.image_path)
        boxes, classes, scores = predict(model, image)
        metric.update(
            [to_torchmetrics_prediction(boxes, classes, scores)],
            [to_torchmetrics_target(sample.boxes, sample.classes)],
        )
    return metric.compute()
