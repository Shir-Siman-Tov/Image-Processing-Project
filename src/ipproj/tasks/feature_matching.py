"""ORB feature detection & matching task (low-level).

Course grounding: F_Features4ML/3060_IP_ImageFeatures, 3061_IP_ImageDescriptors.
"""
import cv2

from ipproj import config
from ipproj.metrics.match_accuracy import compute_match_accuracy


def detect_and_describe(image, n_features: int = config.ORB_N_FEATURES):
    orb = cv2.ORB_create(nfeatures=n_features)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return orb.detectAndCompute(gray, None)


def match(clean_image, other_image):
    """Detects ORB features in both images and matches descriptors.
    Returns (clean_keypoints, other_keypoints, matches, accuracy)."""
    clean_kp, clean_desc = detect_and_describe(clean_image)
    other_kp, other_desc = detect_and_describe(other_image)
    if clean_desc is None or other_desc is None:
        return clean_kp, other_kp, [], 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(clean_desc, other_desc)
    accuracy = compute_match_accuracy(clean_kp, other_kp, matches)
    return clean_kp, other_kp, matches, accuracy
