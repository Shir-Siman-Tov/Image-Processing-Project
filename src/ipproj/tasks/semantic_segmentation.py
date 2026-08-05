"""SegFormer semantic-segmentation task (high-level, DL).

Course grounding: 3_DeepSegmentation/4032_ImageSegmentation.

KITTI Semantics reuses Cityscapes' 19-class "trainId" taxonomy, and
config.SEGFORMER_CHECKPOINT is a Cityscapes-pretrained checkpoint - but the
checkpoint's logit-index order is never assumed to match KITTI's GT mask IDs
without checking. verify_class_alignment() must be run once during
02_clean_baseline; if it returns False, config.SEGFORMER_CLASS_ID_REMAP must
be populated before evaluate()'s results are trusted.
"""
import cv2
import numpy as np
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


def fine_tune(model, processor, samples: list, epochs: int = config.SEGFORMER_FINE_TUNE_EPOCHS,
              lr: float = config.SEGFORMER_FINE_TUNE_LR) -> None:
    """Fine-tunes `model` in-place on `samples` (a distorted split whose labels
    are reused from clean GT, per the PDF's Part 4 "create labels from clean").
    A plain manual loop, not transformers.Trainer: at ~200 images this is
    simpler than standing up a Dataset/TrainingArguments for a one-off run."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.manual_seed(config.RANDOM_SEED)

    for _ in range(epochs):
        for sample in samples:
            image = read_image(sample.image_path)
            gt = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            labels = torch.as_tensor(gt, dtype=torch.long, device=DEVICE)[None]

            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
