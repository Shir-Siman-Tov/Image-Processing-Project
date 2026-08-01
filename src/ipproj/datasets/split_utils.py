"""Shared deterministic train/val/test splitting, reused by every KITTI loader."""
import random

from ipproj import config


def train_val_test_split(items, ratios=config.TRAIN_VAL_TEST_SPLIT, seed=config.RANDOM_SEED):
    """Deterministically partition `items` into (train, val, test) lists."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
