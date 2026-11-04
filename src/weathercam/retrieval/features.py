from collections.abc import Sequence

import numpy as np


def _check_bins(bins: int) -> None:
    if bins < 2:
        raise ValueError("bins must be at least 2")


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.dtype != np.float32:
        raise ValueError("image must be float32")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must be finite")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("image values must lie in [0, 1]")


def color_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    _check_bins(bins)
    _check_image(image)
    levels = np.minimum((image * bins).astype(np.int64), bins - 1)
    joint = (levels[..., 0] * bins + levels[..., 1]) * bins + levels[..., 2]
    counts = np.bincount(joint.ravel(), minlength=bins**3)
    return (counts / joint.size).astype(np.float32)


def color_histograms(images: Sequence[np.ndarray], bins: int = 8) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("no images")
    return np.stack([color_histogram(image, bins) for image in images])
