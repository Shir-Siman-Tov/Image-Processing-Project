"""SegFormer semantic-segmentation task (high-level, DL).

Course grounding: 3_DeepSegmentation/4032_ImageSegmentation.

KITTI Semantics reuses Cityscapes' 19-class "trainId" taxonomy, and
config.SEGFORMER_CHECKPOINT is a Cityscapes-pretrained checkpoint - but the
checkpoint's logit-index order is never assumed to match KITTI's GT mask IDs
without checking. verify_class_alignment() must be run once during
02_clean_baseline; if it returns False, config.SEGFORMER_CLASS_ID_REMAP must
be populated before evaluate()'s results are trusted.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from ipproj import config
from ipproj.datasets.kitti import read_image
from ipproj.metrics.segmentation_iou import new_metric

NUM_CLASSES = len(config.CITYSCAPES_TRAINID_LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained():
    model = SegformerForSemanticSegmentation.from_pretrained(config.SEGFORMER_CHECKPOINT).to(DEVICE)
    model.config.semantic_loss_ignore_index = config.SEGFORMER_IGNORE_INDEX
    processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_CHECKPOINT)
    return model, processor


def verify_class_alignment(model) -> bool:
    """True if the checkpoint's id2label order matches config.CITYSCAPES_TRAINID_LABELS."""
    checkpoint_labels = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]
    expected = [label.lower() for label in config.CITYSCAPES_TRAINID_LABELS]
    return checkpoint_labels == expected


def predict(model, processor, image: np.ndarray) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    pred_ids = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
    if config.SEGFORMER_CLASS_ID_REMAP:
        remap = config.SEGFORMER_CLASS_ID_REMAP
        lut = np.arange(256)
        for src_id, dst_id in remap.items():
            lut[src_id] = dst_id
        pred_ids = lut[pred_ids]
    return pred_ids


def evaluate(model, processor, samples: list) -> torch.Tensor:
    """`samples`: list of kitti.SegmentationSample. Returns per-class IoU tensor."""
    metric = new_metric(NUM_CLASSES, ignore_index=config.SEGFORMER_IGNORE_INDEX)
    for sample in samples:
        image = read_image(sample.image_path)
        pred = predict(model, processor, image)
        gt = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        metric.update(torch.as_tensor(pred), torch.as_tensor(gt.astype(np.int64)))
    return metric.compute()


def checkpoint_path_for(run_name: str) -> Path:
    """Where fine_tune() writes/looks for `run_name`'s fine-tuned weights.

    Mirrors tasks.object_detection.checkpoint_path_for() - exposed so callers
    can check `.exists()` before doing expensive prep work.
    """
    return config.CHECKPOINT_ROOT / "segformer" / run_name


def _forward_loss(model, processor, sample) -> torch.Tensor:
    image = read_image(sample.image_path)
    gt = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    labels = torch.as_tensor(gt, dtype=torch.long, device=DEVICE)[None]
    return model(**inputs, labels=labels).loss


def fine_tune(model, processor, train_samples: list, val_samples: list, run_name: str,
              epochs: int = config.SEGFORMER_FINE_TUNE_EPOCHS,
              lr: float = config.SEGFORMER_FINE_TUNE_LR) -> tuple[Path, pd.DataFrame]:
    """Fine-tunes `model` in-place on `train_samples` (a distorted split whose
    labels are reused from clean GT, per the PDF's Part 4 "create labels from
    clean"). A plain manual loop, not transformers.Trainer: at ~200 images
    this is simpler than standing up a Dataset/TrainingArguments for a
    one-off run.

    Tracks per-epoch validation loss on `val_samples` and keeps the
    best-val-loss state dict (rather than just the last epoch's), then
    persists it under checkpoint_path_for(run_name) via
    model.save_pretrained()/processor.save_pretrained() - CHECKPOINT_ROOT
    lives on Drive on Colab, so this survives a runtime restart the same way
    object_detection.fine_tune()'s YOLO checkpoints already do.

    Skips training and reloads the existing checkpoint's weights into `model`
    if `run_name` was already fine-tuned before (same idempotency guard as
    object_detection.fine_tune()). Returns (checkpoint_path, history_df) with
    one row per epoch: epoch, train_loss, val_loss. On the skip path,
    history_df is read back from the checkpoint's saved history.csv.
    """
    checkpoint_path = checkpoint_path_for(run_name)
    history_path = checkpoint_path / "history.csv"

    if checkpoint_path.exists():
        loaded = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path).to(DEVICE)
        model.load_state_dict(loaded.state_dict())
        model.eval()
        return checkpoint_path, pd.read_csv(history_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.manual_seed(config.RANDOM_SEED)

    best_val_loss = float("inf")
    best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for sample in train_samples:
            loss = _forward_loss(model, processor, sample)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / max(len(train_samples), 1)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for sample in val_samples:
                val_loss_sum += _forward_loss(model, processor, sample).item()
        val_loss = val_loss_sum / max(len(val_samples), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    model.eval()

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    processor.save_pretrained(checkpoint_path)
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    return checkpoint_path, history_df
