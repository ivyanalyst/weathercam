import io
from pathlib import Path

import numpy as np
from PIL import Image

from weathercam.data.augment import _check_image

_DECODE_ERRORS = (OSError, SyntaxError, EOFError, ValueError, Image.DecompressionBombError)


def decode_rgb(data) -> np.ndarray:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"data must be bytes-like, got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("data is empty")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except _DECODE_ERRORS as error:
        raise ValueError(f"cannot decode image: {error}") from error
    return pixels.astype(np.float32) / np.float32(255.0)


def load_rgb(path) -> np.ndarray:
    return decode_rgb(Path(path).read_bytes())


def save_png(path, image) -> None:
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError(f"path must end in .png, got {target.name}")
    checked = _check_image(image)
    pixels = np.rint(checked * 255.0).astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(target, "PNG")
