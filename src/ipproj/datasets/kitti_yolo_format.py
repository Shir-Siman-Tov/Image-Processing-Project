"""Convert KITTI's native object-detection label format to YOLO's, for
fine-tuning YOLOv8's head on KITTI's own classes (see config.KITTI_DETECTION_CLASSES).

"DontCare" boxes are intentionally never written here - they're KITTI's
ignore-region marker, not a trainable class (see kitti.DetectionSample.ignore_boxes).
"""
import shutil
from pathlib import Path

import cv2
import yaml

from ipproj import config
from ipproj.datasets.kitti import DetectionSample


def convert_split_to_yolo(samples: list[DetectionSample], output_dir: Path, split_name: str) -> None:
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at {sample.image_path}")
        height, width = image.shape[:2]

        dest_image = images_dir / sample.image_path.name
        if not dest_image.exists():
            shutil.copy2(sample.image_path, dest_image)

        lines = []
        for box, cls in zip(sample.boxes, sample.classes):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            class_id = config.KITTI_DETECTION_CLASSES.index(cls)
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        label_path = labels_dir / f"{sample.image_path.stem}.txt"
        label_path.write_text("\n".join(lines))


def write_yolo_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "data.yaml"
    content = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(config.KITTI_DETECTION_CLASSES)},
    }
    yaml_path.write_text(yaml.safe_dump(content, sort_keys=False))
    return yaml_path


def build_yolo_dataset(splits: dict, output_dir: Path = config.KITTI_ROOT / "yolo_format") -> Path:
    """`splits` is the {"train": [...], "val": [...], "test": [...]} dict returned by
    kitti.load_object_detection_subset(). Returns the path to the written data.yaml,
    ready to pass as `data=` into ultralytics.YOLO().train()."""
    for split_name, samples in splits.items():
        convert_split_to_yolo(samples, output_dir, split_name)
    return write_yolo_dataset_yaml(output_dir)
