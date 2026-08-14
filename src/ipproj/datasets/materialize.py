"""Materializes a transformed (distorted/restored) copy of a dataset split to
disk, producing new sample objects that point at the transformed images while
keeping their original GT untouched (a distortion/restoration changes pixels,
not content) - lets tasks.*.evaluate() be reused unchanged in 03/04/05, since
those always read images from a sample's `.image_path`.
"""
import dataclasses
from pathlib import Path

import cv2
from tqdm.auto import tqdm

from ipproj.datasets.kitti import read_image


def materialize_transformed(samples: list, transform, output_dir: Path, name_prefix: str = "") -> list:
    """`transform(image: np.ndarray) -> np.ndarray`. Returns new sample objects
    (same dataclass type and GT as the input) whose image_path points into `output_dir`.

    `name_prefix` disambiguates filenames when multiple transformed copies of the
    same source image are later combined into one directory (e.g. blended
    multi-distortion/multi-severity fine-tuning, notebook 05) - without it, two
    variants of the same source image would share a filename and downstream
    dedup-by-name logic (kitti_yolo_format.convert_split_to_yolo) would silently
    keep only one. Default "" preserves single-variant call sites unchanged.

    Reports progress via tqdm - this loop round-trips through disk (and, on
    Colab, often a Drive mount) once per sample, so silently sitting through
    it on a large split is easy to mistake for a hang."""
    output_dir.mkdir(parents=True, exist_ok=True)
    new_samples = []
    for sample in tqdm(samples, desc=f"materializing {output_dir.name}", unit="img", leave=False):
        image = read_image(sample.image_path)
        transformed = transform(image)
        out_path = output_dir / f"{name_prefix}{sample.image_path.name}"
        cv2.imwrite(str(out_path), cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
        new_samples.append(dataclasses.replace(sample, image_path=out_path))
    return new_samples
