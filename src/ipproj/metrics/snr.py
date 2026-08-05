"""Signal-to-noise ratio - used to characterize distortion intensity on a
common axis regardless of which distortion produced it (required: "measure
performance per SNR, range of distortion intensities")."""
import numpy as np


def compute_snr_db(clean: np.ndarray, distorted: np.ndarray) -> float:
    clean = clean.astype(np.float64)
    distorted = distorted.astype(np.float64)
    signal_power = np.mean(clean**2)
    noise_power = np.mean((clean - distorted) ** 2)
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return 0.0
    return float(10 * np.log10(signal_power / noise_power))
