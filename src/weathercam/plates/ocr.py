from typing import NamedTuple

import numpy as np

from weathercam.data.ccpd import ADS, ALPHABETS, PROVINCES

_TABLES = (PROVINCES, ALPHABETS) + (ADS,) * 5
POSITION_CLASSES = tuple(len(table) for table in _TABLES)
NUM_CLASSES = max(POSITION_CLASSES)


class PlateReading(NamedTuple):
    text: str
    confidence: float
    positions: tuple


def decode_positions(logits):
    if not isinstance(logits, np.ndarray) or logits.shape != (7, NUM_CLASSES):
        raise ValueError(f"logits must be an array of shape (7, {NUM_CLASSES})")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits contain non-finite values")
    chars, probs = [], []
    for position, (table, valid) in enumerate(zip(_TABLES, POSITION_CLASSES)):
        row = logits[position, :valid].astype(np.float64)
        weights = np.exp(row - row.max())
        probability = weights / weights.sum()
        best = int(probability.argmax())
        chars.append(table[best])
        probs.append(float(probability[best]))
    return PlateReading("".join(chars), min(probs), tuple(probs))


def decode_batch(logits):
    if not isinstance(logits, np.ndarray) or logits.ndim != 3 or logits.shape[1:] != (7, NUM_CLASSES):
        raise ValueError(f"logits must be an array of shape (N, 7, {NUM_CLASSES})")
    return [decode_positions(item) for item in logits]


def read_all(reader, crops, batch_size=64):
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    readings = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        found = reader.read(batch)
        if len(found) != len(batch):
            raise ValueError(f"reader returned {len(found)} readings for {len(batch)} crops")
        if not all(isinstance(item, PlateReading) for item in found):
            raise ValueError("reader must return PlateReading objects")
        readings.extend(found)
    return readings
