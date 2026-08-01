"""Salt & pepper (impulse) noise distortion.

Course grounding: 3046_IP_NoiseSmoothing - taught as an impulse-noise type
whose correct restoration is a median filter, not e.g. Non-Local Means
(see restoration/median_filter.py).
"""
import numpy as np

from ipproj import config

NAME = "salt_pepper"
LEVELS = config.SALT_PEPPER_AMOUNTS  # fraction of pixels replaced, per intensity level


def distort(image: np.ndarray, level: int, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """`level` indexes into LEVELS (higher level = more corrupted pixels)."""
    amount = LEVELS[level]
    rng = np.random.default_rng(seed)
    out = image.copy()
    mask = rng.random(image.shape[:2])
    out[mask < amount / 2] = 255  # salt
    out[(mask >= amount / 2) & (mask < amount)] = 0  # pepper
    return out
