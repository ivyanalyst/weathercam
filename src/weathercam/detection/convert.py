import math

import numpy as np

from weathercam.core.types import Box, Detection


def to_detections(boxes, scores, labels, label_map, width, height) -> list[Detection]:
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError(f"image size must be positive and finite, got {width}x{height}")
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {boxes.shape}")
    count = boxes.shape[0]
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (count,):
        raise ValueError(f"scores must have shape ({count},), got {scores.shape}")
    labels = np.asarray(labels)
    if labels.shape != (count,):
        raise ValueError(f"labels must have shape ({count},), got {labels.shape}")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes must be finite")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("scores must be in [0, 1]")
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all() or (labels != np.rint(labels)).any():
            raise ValueError("labels must be integral")
    elif labels.dtype.kind not in "iu":
        raise ValueError(f"labels must be integers, got dtype {labels.dtype}")
    if ((boxes[:, 2] < boxes[:, 0]) | (boxes[:, 3] < boxes[:, 1])).any():
        raise ValueError("box requires x2 >= x1 and y2 >= y1")
    right = float(width)
    bottom = float(height)
    result: list[Detection] = []
    for index in range(count):
        mapped = label_map.get(int(labels[index]))
        if mapped is None:
            continue
        x1 = min(max(float(boxes[index, 0]), 0.0), right)
        y1 = min(max(float(boxes[index, 1]), 0.0), bottom)
        x2 = min(max(float(boxes[index, 2]), 0.0), right)
        y2 = min(max(float(boxes[index, 3]), 0.0), bottom)
        if x2 <= x1 or y2 <= y1:
            continue
        result.append(Detection(Box(x1, y1, x2, y2), int(mapped), float(scores[index])))
    return result
