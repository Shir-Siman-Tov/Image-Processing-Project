"""Restoration registry - keyed by the distortion name it restores (matches
ipproj.distortions.REGISTRY's keys), so 04_restoration.ipynb can look up the
matching restoration method for each distortion generically.
"""
from ipproj.restoration import bilateral_interp, deconvolution, median_filter

REGISTRY = {
    median_filter.NAME: median_filter,
    deconvolution.NAME: deconvolution,
    bilateral_interp.NAME: bilateral_interp,
}
