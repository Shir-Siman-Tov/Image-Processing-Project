"""YOLOv8 object-detection task (high-level, DL).

Course grounding: 2_DeepDetection/4021_BasicsOfObjectDetection, 4022_AdvancedObjectDetection.

A COCO-pretrained YOLO can't be evaluated against KITTI's own class taxonomy
directly - COCO has no Pedestrian/Cyclist/Van/Tram classes. The head must be
fine-tuned on KITTI's own classes (config.KITTI_DETECTION_CLASSES) first;
that fine-tuned checkpoint, not the raw COCO one, is what "baseline" means
for this task in 02_clean_baseline and everything downstream.
"""
import shutil
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


def _reconcile(canonical_dir: Path, other_dir: Path) -> None:
    """Folds `other_dir`'s progress into `canonical_dir` - every check in
    fine_tune() only ever looks at canonical_dir, but `model.train(resume=True)`
    writes its output to whatever run name got baked into the checkpoint's own
    saved args back when that run was first created (e.g. "finetuned_blended-3",
    if "finetuned_blended"/"-2" happened to be taken at that moment), not
    necessarily canonical_dir - and renaming a directory on disk doesn't change
    that baked-in name, so every future resume of the same checkpoint keeps
    landing in that same original spot.

    Merges results.csv (deduped by epoch, so calling this twice is a no-op),
    copies weights/best.pt over only if other_dir has one (a resumed tail may
    legitimately have none, if nothing in it beat the fitness already baked
    into the checkpoint it resumed from - canonical's existing best.pt, from
    before this merge, is then still the right one), and always copies
    weights/last.pt (that should track the most recent state regardless of
    which directory it happened to land in).
    """
    if other_dir == canonical_dir:
        return

    other_results = other_dir / "results.csv"
    if other_results.exists():
        other_df = pd.read_csv(other_results)
        other_df.columns = other_df.columns.str.strip()
        canonical_results = canonical_dir / "results.csv"
        if canonical_results.exists():
            canonical_df = pd.read_csv(canonical_results)
            canonical_df.columns = canonical_df.columns.str.strip()
            merged = pd.concat([canonical_df[~canonical_df["epoch"].isin(other_df["epoch"])], other_df])
            merged = merged.sort_values("epoch")
        else:
            merged = other_df
        canonical_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(canonical_results, index=False)

    (canonical_dir / "weights").mkdir(parents=True, exist_ok=True)
    other_best = other_dir / "weights" / "best.pt"
    if other_best.exists():
        shutil.copy2(other_best, canonical_dir / "weights" / "best.pt")
    other_last = other_dir / "weights" / "last.pt"
    if other_last.exists():
        shutil.copy2(other_last, canonical_dir / "weights" / "last.pt")


def _reconcile_furthest_sibling(run_name: str, canonical_dir: Path) -> None:
    """Self-heals a canonical_dir left stale by a crash between a resumed
    run finishing and fine_tune() reconciling it (see _reconcile) - scans for
    any "{run_name}*" directory further along than canonical_dir and folds it
    in, so a rerun doesn't redo already-completed epochs. No-op (and cheap)
    when nothing is ahead, e.g. every call for a run_name that never hit this."""
    canonical_epoch = _last_completed_epoch(canonical_dir)
    for sibling in config.CHECKPOINT_ROOT.joinpath("yolo").glob(f"{run_name}*"):
        if sibling == canonical_dir or not sibling.is_dir():
            continue
        sibling_epoch = _last_completed_epoch(sibling)
        if sibling_epoch > canonical_epoch:
            _reconcile(canonical_dir, sibling)
            canonical_epoch = sibling_epoch


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
    _reconcile_furthest_sibling(run_name, run_dir)

    if checkpoint_path.exists() and _last_completed_epoch(run_dir) >= epochs:
        return checkpoint_path

    last_checkpoint = checkpoint_path.parent / "last.pt"
    if last_checkpoint.exists():
        model = YOLO(last_checkpoint)
        # resume=True makes ultralytics restore data/epochs/optimizer state
        # from the interrupted run's own saved args - re-passing them here
        # would be ignored (and risks drifting from what that run actually used).
        # It may also write output under a different directory than run_dir
        # (see _reconcile's docstring), hence folding it back in below rather
        # than trusting results.save_dir directly.
        results = model.train(resume=True)
        _reconcile(run_dir, Path(results.save_dir))
        return checkpoint_path
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
