"""Classical dense optical flow (Farneback) task (low-level).

Course grounding: B_OpticalFlow/3821__OpticalFlow.
"""
import cv2
import numpy as np

from ipproj import config
from ipproj.metrics.epe import compute_epe, compute_fl_error


def compute_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if frame1.ndim == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if frame2.ndim == 3 else frame2
    return cv2.calcOpticalFlowFarneback(gray1, gray2, None, **config.FARNEBACK_PARAMS)


def evaluate(frame1: np.ndarray, frame2: np.ndarray, gt_flow: np.ndarray, valid: np.ndarray) -> dict:
    pred_flow = compute_flow(frame1, frame2)
    return {
        "epe": compute_epe(pred_flow, gt_flow, valid),
        "fl_error": compute_fl_error(pred_flow, gt_flow, valid),
    }
