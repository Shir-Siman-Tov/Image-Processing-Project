"""Linear motion blur distortion.

Course grounding: 3043_IP_Freq_Filtering, dedicated "Enhancing Motion Blur"
section - motion blur is modeled as convolution with a linear kernel in the
direction of motion; restoration undoes it via frequency-domain deconvolution
with that same (known) kernel (see restoration/deconvolution.py).
"""
import cv2
import numpy as np

from ipproj import config

NAME = "motion_blur"
LEVELS = config.MOTION_BLUR_KERNEL_SIZES  # kernel length in px, per intensity level


def motion_kernel(kernel_size: int, angle_deg: float = 0.0) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (kernel_size / 2, kernel_size / 2)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (kernel_size, kernel_size))
    return kernel / kernel.sum()


def distort(image: np.ndarray, level: int, angle_deg: float = 0.0) -> np.ndarray:
    """`level` indexes into LEVELS (higher level = longer blur kernel)."""
    kernel = motion_kernel(LEVELS[level], angle_deg)
    return cv2.filter2D(image, -1, kernel)
