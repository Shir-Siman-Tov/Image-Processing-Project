"""Distortion registry - keyed by name, so notebooks can loop over all
distortions x intensity levels generically instead of hardcoding per-distortion
branches. Adding a new distortion means adding one module here and one entry.
"""
from ipproj.distortions import jpeg_compression, motion_blur, salt_pepper

REGISTRY = {
    salt_pepper.NAME: salt_pepper,
    motion_blur.NAME: motion_blur,
    jpeg_compression.NAME: jpeg_compression,
}
