import math

import numpy as np
from PIL import Image

from weathercam.core.boxes import clip_box
from weathercam.core.types import Box
from weathercam.data.augment import _check_image


def expand_box(box, margin, width, height):
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    dx, dy = margin * box.width, margin * box.height
    return clip_box(Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy), width, height)


def pixel_box(box, width, height):
    clipped = clip_box(box, width, height)
    x1, y1 = math.floor(clipped.x1), math.floor(clipped.y1)
    x2, y2 = math.ceil(clipped.x2), math.ceil(clipped.y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box is empty inside the {width}x{height} image: {box}")
    return (x1, y1, x2, y2)


def crop_array(image, box):
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must be a numpy array of shape (H, W, C)")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = pixel_box(box, width, height)
    return image[y1:y2, x1:x2].copy()


def jitter_box(box, rng, shift, scale):
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}")
    if not 0.0 <= scale < 1.0:
        raise ValueError(f"scale must be in [0, 1), got {scale}")
    cx = (box.x1 + box.x2) / 2 + rng.uniform(-shift, shift) * box.width
    cy = (box.y1 + box.y2) / 2 + rng.uniform(-shift, shift) * box.height
    w = box.width * (1 + rng.uniform(-scale, scale))
    h = box.height * (1 + rng.uniform(-scale, scale))
    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def resize_rgb(image, height, width):
    if height < 1 or width < 1:
        raise ValueError(f"size must be positive, got {height}x{width}")
    checked = _check_image(image)
    channels = [
        np.asarray(Image.fromarray(checked[:, :, c]).resize((width, height), Image.Resampling.BICUBIC))
        for c in range(3)
    ]
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0).astype(np.float32)
