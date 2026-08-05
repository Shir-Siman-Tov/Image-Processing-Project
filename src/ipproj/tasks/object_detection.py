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
import pandas as pd
from ultralytics import YOLO

from ipproj import config
from ipproj.metrics.detection_map import (
    new_metric,
    suppress_dontcare_predictions,
    to_torchmetrics_prediction,
    to_torchmetrics_target,
)


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


def load_training_curve(checkpoint_path: Path) -> dict:
    """Reads the per-epoch train/val loss history ultralytics wrote next to
    `checkpoint_path` (weights/best.pt) during model.train(). Each loss is
    the sum of that split's box/cls/dfl loss columns from results.csv.
    """
    results_csv = checkpoint_path.parent.parent / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"No results.csv next to {checkpoint_path} - was this checkpoint produced by model.train()?"
        )
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    train_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    val_cols = [c for c in df.columns if c.startswith("val/") and c.endswith("_loss")]
    return {
        "epoch": df["epoch"].astype(int).tolist(),
        "train_loss": df[train_cols].sum(axis=1).tolist(),
        "val_loss": df[val_cols].sum(axis=1).tolist(),
    }


def predict(model: YOLO, image: np.ndarray, confidence: float = config.YOLO_CONFIDENCE_THRESHOLD):
    """Returns (boxes[N,4] xyxy, classes: list[str], scores[N])."""
    result = model.predict(image, conf=confidence, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    classes = [model.names[i] for i in class_ids]
    return boxes, classes, scores


def evaluate(model: YOLO, samples: list, batch_size: int = config.YOLO_EVAL_BATCH_SIZE) -> dict:
    """`samples`: list of kitti.DetectionSample. Returns per-class + overall mAP/IoU.

    Runs inference through Ultralytics' own batched path (image paths, not
    pre-loaded arrays) rather than one image at a time - YOLO handles its own
    I/O/preprocessing per image and batches the GPU forward pass.
    """
    metric = new_metric()
    image_paths = [str(sample.image_path) for sample in samples]
    results = model.predict(image_paths, batch=batch_size, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    for sample, result in zip(samples, results):
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        classes = [model.names[i] for i in class_ids]
        boxes, classes, scores = suppress_dontcare_predictions(boxes, classes, scores, sample.ignore_boxes)

        metric.update(
            [to_torchmetrics_prediction(boxes, classes, scores)],
            [to_torchmetrics_target(sample.boxes, sample.classes)],
        )
    return metric.compute()
