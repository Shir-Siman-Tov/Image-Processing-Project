"""Materializes a transformed (distorted/restored) copy of a dataset split to
disk, producing new sample objects that point at the transformed images while
keeping their original GT untouched (a distortion/restoration changes pixels,
not content) - lets tasks.*.evaluate() be reused unchanged in 03/04/05, since
those always read images from a sample's `.image_path`.
"""
import dataclasses
from pathlib import Path

import cv2

from ipproj.datasets.kitti import read_image


def materialize_transformed(samples: list, transform, output_dir: Path) -> list:
    """`transform(image: np.ndarray) -> np.ndarray`. Returns new sample objects
    (same dataclass type and GT as the input) whose image_path points into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    new_samples = []
    for sample in samples:
        image = read_image(sample.image_path)
        transformed = transform(image)
        out_path = output_dir / sample.image_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
        new_samples.append(dataclasses.replace(sample, image_path=out_path))
    return new_samples
