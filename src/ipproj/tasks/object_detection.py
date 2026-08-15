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


def _last_completed_epoch(run_dir: Path) -> int:
    """Highest epoch number appearing in `run_dir`/results.csv, or 0 if the
    run hasn't produced one yet. Ultralytics appends a row after every epoch
    (not just at the end), so this reflects true progress even mid-run."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return 0
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    return int(df["epoch"].max()) if len(df) else 0


def fine_tune(
    data_yaml: Path, run_name: str, epochs: int = config.YOLO_FINE_TUNE_EPOCHS, start_weights: Path | str | None = None
) -> Path:
    """Fine-tunes YOLO's head on config.KITTI_DETECTION_CLASSES, starting from
    `start_weights` (defaults to config.YOLO_BASE_WEIGHTS via load_pretrained()
    - e.g. 02_clean_baseline adapting raw COCO weights). Callers continuing an
    already-adapted checkpoint (e.g. 05_finetuning_distorted continuing
    02's clean-adapted checkpoint) pass that checkpoint's path explicitly.
    `data_yaml` comes from datasets.kitti_yolo_format.build_yolo_dataset().
    Returns the path to the resulting best.pt checkpoint.

    Skips training and returns the existing checkpoint if `run_name` already
    finished fine-tuning - checked via results.csv reaching `epochs`, not just
    best.pt existing, since Ultralytics writes best.pt on every fitness
    improvement, not only at completion (a run interrupted at epoch 45/50
    already has a best.pt from some earlier epoch - treating that alone as
    "done" would silently hand back an under-trained checkpoint and never
    train epochs 46-50). If `run_name` was interrupted mid-run (e.g. a Colab
    disconnect), resumes from `last.pt` (which Ultralytics writes every
    epoch) instead of restarting at epoch 1. CHECKPOINT_ROOT lives on Drive
    on Colab, so both checkpoints survive a runtime restart.
    """
    checkpoint_path = checkpoint_path_for(run_name)
    run_dir = checkpoint_path.parent.parent
    if checkpoint_path.exists() and _last_completed_epoch(run_dir) >= epochs:
        return checkpoint_path

    last_checkpoint = checkpoint_path.parent / "last.pt"
    if last_checkpoint.exists():
        model = YOLO(last_checkpoint)
        # resume=True makes ultralytics restore data/epochs/optimizer state
        # from the interrupted run's own saved args - re-passing them here
        # would be ignored (and risks drifting from what that run actually used).
        results = model.train(resume=True)
    else:
        model = YOLO(start_weights) if start_weights is not None else load_pretrained()
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            seed=config.RANDOM_SEED,
            project=str(config.CHECKPOINT_ROOT / "yolo"),
            name=run_name,
            # Without this, Ultralytics silently renames the run to
            # "{run_name}-2", "-3", ... whenever a directory named `run_name`
            # already exists (e.g. a prior interrupted attempt) instead of
            # reusing it - and every check above only ever looks at the fixed
            # checkpoint_path_for(run_name) path, so a renamed run's progress
            # would become permanently invisible to future calls, silently
            # restarting from `start_weights` every time instead of resuming.
            exist_ok=True,
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
