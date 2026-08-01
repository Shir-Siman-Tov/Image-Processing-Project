"""ORB feature-matching accuracy under distortion.

Our 3 distortions (salt & pepper, motion blur, JPEG compression) are all
purely pixel-domain - no geometric warp - so a "correct" match is simply one
where the matched keypoint's pixel location in the distorted image is within
a small tolerance of its location in the clean image; no homography needed.
"""
import numpy as np

PIXEL_TOLERANCE = 3.0


def compute_match_accuracy(clean_keypoints, distorted_keypoints, matches, tolerance: float = PIXEL_TOLERANCE) -> float:
    """`matches`: list of cv2.DMatch, clean keypoints as queryIdx, distorted as trainIdx."""
    if not matches:
        return 0.0
    correct = 0
    for m in matches:
        clean_pt = np.array(clean_keypoints[m.queryIdx].pt)
        distorted_pt = np.array(distorted_keypoints[m.trainIdx].pt)
        if np.linalg.norm(clean_pt - distorted_pt) <= tolerance:
            correct += 1
    return correct / len(matches)
