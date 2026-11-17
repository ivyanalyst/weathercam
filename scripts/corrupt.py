"""Seeded synthetic corruptions for phase 15. Same input + seed -> same bytes."""
import io, zlib
import numpy as np
from PIL import Image, ImageFilter

PARAMS = {
    "noise": {1: 15.0, 2: 40.0},   # gaussian sigma, 0-255 scale
    "blur": {1: 2.0, 2: 5.0},      # gaussian blur radius, px
    "jpeg": {1: 20, 2: 5},         # JPEG quality of the corrupted image
    "dark": {1: 0.4, 2: 0.15},     # brightness multiplier
    "fog": {1: 0.4, 2: 0.7},       # uniform haze: blend toward light gray (225)
}
SAVE_QUALITY = 95  # re-encode quality for every non-jpeg corruption


def apply(raw: bytes, name: str, severity: int, seed: int, key: str) -> bytes:
    p = PARAMS[name][severity]
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if name == "noise":
        rng = np.random.default_rng(seed + zlib.crc32(key.encode()))
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0.0, p, (img.height, img.width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif name == "blur":
        img = img.filter(ImageFilter.GaussianBlur(p))
    elif name == "dark":
        img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * p, 0, 255).astype(np.uint8))
    elif name == "fog":
        arr = np.asarray(img, dtype=np.float32) * (1 - p) + p * 225.0
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif name != "jpeg":
        raise ValueError(name)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(p if name == "jpeg" else SAVE_QUALITY))
    return buf.getvalue()
