import io
import math

import numpy as np
from PIL import Image

from weathercam.data.augment import _check_image, _check_severity
from weathercam.data.imagecodec import decode_rgb

BLUR_SIGMA_FRACTION = 0.02
NOISE_STD = 0.2
DARKEN_STRENGTH = 0.8
JPEG_QUALITY_TOP = 95.0
JPEG_QUALITY_SPAN = 90.0


def _blur_axis(array, kernel, axis):
    radius = len(kernel) // 2
    length = array.shape[axis]
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="edge")
    out = np.zeros(array.shape, dtype=np.float64)
    for i, weight in enumerate(kernel):
        index = [slice(None)] * array.ndim
        index[axis] = slice(i, i + length)
        out += weight * padded[tuple(index)]
    return out


def gaussian_blur(image, severity):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    sigma = BLUR_SIGMA_FRACTION * severity * max(out.shape[0], out.shape[1])
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    blurred = _blur_axis(_blur_axis(out.astype(np.float64), kernel, 0), kernel, 1)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def gaussian_noise(image, severity, rng):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    noise = rng.normal(0.0, NOISE_STD * severity, size=out.shape)
    return np.clip(out + noise, 0.0, 1.0).astype(np.float32)


def darken(image, severity):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    return (out * np.float32(1.0 - DARKEN_STRENGTH * severity)).astype(np.float32)


def jpeg(image, severity):
    out = _check_image(image)
    severity = _check_severity(severity)
    if severity == 0.0:
        return out
    quality = int(round(JPEG_QUALITY_TOP - JPEG_QUALITY_SPAN * severity))
    pixels = np.rint(out * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality)
    return decode_rgb(buffer.getvalue())
