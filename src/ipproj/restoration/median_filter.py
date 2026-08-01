"""Median filter restoration - the course-taught counterpart to salt & pepper
(impulse) noise (3046_IP_NoiseSmoothing: Non-Local Means is explicitly
"not suitable for impulse noise"; median filtering is)."""
import cv2
import numpy as np

from ipproj import config

NAME = "salt_pepper"  # matches the distortion this restores (distortions.salt_pepper.NAME)


def restore(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, config.MEDIAN_FILTER_KERNEL_SIZE)
