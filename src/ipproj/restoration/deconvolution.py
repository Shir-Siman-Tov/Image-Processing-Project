"""Frequency-domain (Wiener) deconvolution - the course-taught counterpart to
motion blur (3043_IP_Freq_Filtering, "Enhancing Motion Blur" section: known
degradation kernel -> deconvolution in the frequency domain).

Unlike median_filter/bilateral_interp, restore() needs the degradation
kernel size that produced the distortion - deconvolution is only well-posed
when the blur kernel is known, which is the assumption this course lecture
teaches. This makes its signature intentionally asymmetric with the other
restoration modules; forcing a fake-uniform signature would hide that assumption.
"""
import cv2
import numpy as np

from ipproj import config
from ipproj.distortions.motion_blur import motion_kernel

NAME = "motion_blur"  # matches the distortion this restores (distortions.motion_blur.NAME)


def _wiener_deconvolve_channel(channel: np.ndarray, kernel: np.ndarray, nsr: float) -> np.ndarray:
    h, w = channel.shape
    kh, kw = kernel.shape
    kernel_padded = np.zeros((h, w), dtype=np.float32)
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, (-kh // 2, -kw // 2), axis=(0, 1))

    channel_fft = np.fft.fft2(channel)
    kernel_fft = np.fft.fft2(kernel_padded)
    wiener_filter = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + nsr)
    restored = np.real(np.fft.ifft2(channel_fft * wiener_filter))
    return np.clip(restored, 0, 255)


def restore(image: np.ndarray, kernel_size: int, angle_deg: float = 0.0) -> np.ndarray:
    """`kernel_size` must match the motion_blur.distort() level that produced `image`."""
    kernel = motion_kernel(kernel_size, angle_deg)
    channels = cv2.split(image.astype(np.float32)) if image.ndim == 3 else [image.astype(np.float32)]
    restored_channels = [
        _wiener_deconvolve_channel(c, kernel, config.DECONVOLUTION_SNR_ESTIMATE) for c in channels
    ]
    restored = cv2.merge(restored_channels) if len(restored_channels) > 1 else restored_channels[0]
    return restored.astype(np.uint8)
