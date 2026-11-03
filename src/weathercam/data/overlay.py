import math

import numpy as np

from weathercam.data.augment import _check_image


def _check_color(color) -> np.ndarray:
    try:
        rgb = np.asarray(color, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"color must be three numbers, got {color!r}") from error
    if rgb.shape != (3,):
        raise ValueError(f"color must have 3 values, got shape {rgb.shape}")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("color contains non-finite values")
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        raise ValueError("color values must be in [0, 1]")
    return rgb


def _check_thickness(thickness) -> int:
    value = int(thickness)
    if value != thickness or value < 1:
        raise ValueError(f"thickness must be an integer >= 1, got {thickness}")
    return value


def _round(value: float) -> int:
    return math.floor(value + 0.5)


def draw_boxes(image, boxes, color, thickness=1) -> np.ndarray:
    out = _check_image(image)
    rgb = _check_color(color)
    thick = _check_thickness(thickness)
    height, width = out.shape[:2]
    for box in boxes:
        x1, y1 = _round(box.x1), _round(box.y1)
        x2, y2 = _round(box.x2), _round(box.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx1, cy1 = max(x1, 0), max(y1, 0)
        cx2, cy2 = min(x2, width), min(y2, height)
        if cx1 >= cx2 or cy1 >= cy2:
            continue
        mask = np.ones((cy2 - cy1, cx2 - cx1), dtype=bool)
        ix1, iy1, ix2, iy2 = x1 + thick, y1 + thick, x2 - thick, y2 - thick
        if ix1 < ix2 and iy1 < iy2:
            jx1, jy1 = max(ix1, cx1), max(iy1, cy1)
            jx2, jy2 = min(ix2, cx2), min(iy2, cy2)
            if jx1 < jx2 and jy1 < jy2:
                mask[jy1 - cy1 : jy2 - cy1, jx1 - cx1 : jx2 - cx1] = False
        region = out[cy1:cy2, cx1:cx2]
        region[mask] = rgb
    return out
