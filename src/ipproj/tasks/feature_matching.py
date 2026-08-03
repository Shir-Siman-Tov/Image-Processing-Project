"""ORB feature detection & matching task (low-level).

Course grounding: F_Features4ML/3060_IP_ImageFeatures, 3061_IP_ImageDescriptors.
Matching strategy is a user-approved deviation from those slides: 3060/3061
only describe matching as "compute descriptor similarity, keep highest
similarity matches", and the closest lecture that names a concrete ORB
matching algorithm (3803_CV_ImageRegistration, a different course module)
describes brute-force greedy Hamming matching, not Lowe's ratio test. Ratio
test is used here anyway, by explicit user request, filtering knn candidate
pairs rather than requiring mutual nearest-neighbor agreement.
"""
import cv2

from ipproj import config
from ipproj.metrics.match_accuracy import compute_match_accuracy

_DEFAULT_ORB = cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)


def detect_and_describe(image, n_features: int = config.ORB_N_FEATURES):
    orb = _DEFAULT_ORB if n_features == config.ORB_N_FEATURES else cv2.ORB_create(nfeatures=n_features)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return orb.detectAndCompute(gray, None)


def match(clean_image, other_image):
    """Detects ORB features in both images and matches descriptors via Lowe's ratio test.

    Returns (clean_keypoints, other_keypoints, good_matches, accuracy, good_match_ratio).
    `good_match_ratio` is a filter-selectivity diagnostic (good_matches / raw candidate
    pairs), not a correctness metric - `accuracy` (from compute_match_accuracy) remains
    the metric of record for this task.
    """
    clean_kp, clean_desc = detect_and_describe(clean_image)
    other_kp, other_desc = detect_and_describe(other_image)
    if clean_desc is None or other_desc is None or len(clean_desc) < 2 or len(other_desc) < 2:
        return clean_kp, other_kp, [], 0.0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(clean_desc, other_desc, k=2)

    good_matches = [
        m for m, n in (pair for pair in raw_matches if len(pair) == 2)
        if m.distance < config.ORB_RATIO_TEST_THRESHOLD * n.distance
    ]
    good_match_ratio = len(good_matches) / len(raw_matches) if raw_matches else 0.0

    accuracy = compute_match_accuracy(clean_kp, other_kp, good_matches)
    return clean_kp, other_kp, good_matches, accuracy, good_match_ratio
