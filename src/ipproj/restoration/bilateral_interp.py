"""Bilateral filtering + interpolation restoration - the assignment PDF's own
worked example pairing for JPEG compression artifacts (no dedicated course
lecture; see distortions/jpeg_compression.py for the course-alignment note)."""
import cv2
import numpy as np

from ipproj import config

NAME = "jpeg_compression"  # matches the distortion this restores (distortions.jpeg_compression.NAME)


def restore(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = config.INTERPOLATION_SCALE
    upsampled = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    filtered = cv2.bilateralFilter(upsampled, **config.BILATERAL_FILTER_PARAMS)
    return cv2.resize(filtered, (w, h), interpolation=cv2.INTER_AREA)
