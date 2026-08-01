"""JPEG compression-artifact distortion.

Not tied to a dedicated course lecture (this course has no dedicated
DCT/compression lecture); grounded only in the assignment PDF's own worked
example, which pairs it with bilateral filtering + interpolation for
restoration (see restoration/bilateral_interp.py). Flagged and user-approved
as a course-alignment gap when the distortion set was locked in.
"""
import cv2
import numpy as np

from ipproj import config

NAME = "jpeg_compression"
LEVELS = config.JPEG_QUALITY_LEVELS  # JPEG quality, per intensity level (lower = more artifacts)


def distort(image: np.ndarray, level: int) -> np.ndarray:
    """`level` indexes into LEVELS (higher level = lower quality = more artifacts)."""
    quality = LEVELS[level]
    success, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise RuntimeError(f"JPEG encoding failed at quality={quality}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
