from collections import Counter

import numpy as np

from weathercam.data.ccpd import ADS, ALPHABETS, PROVINCES

_TABLES = (PROVINCES, ALPHABETS) + (ADS,) * 5


def encode_plate(plate):
    if not isinstance(plate, str) or len(plate) != 7:
        raise ValueError(f"plate needs 7 characters, got {plate!r}")
    indices = []
    for position, (table, char) in enumerate(zip(_TABLES, plate), start=1):
        if char not in table:
            raise ValueError(f"character {char!r} is not allowed at position {position}")
        indices.append(table.index(char))
    return tuple(indices)


def draw_probabilities(first_chars, exponent=0.5, cap_draws=16.0):
    n = len(first_chars)
    if n == 0:
        raise ValueError("first_chars must not be empty")
    if exponent < 0:
        raise ValueError(f"exponent must be non-negative, got {exponent}")
    if cap_draws < 1:
        raise ValueError(f"cap_draws must be at least 1, got {cap_draws}")
    counts = Counter(first_chars)
    weights = np.array([counts[char] ** -exponent for char in first_chars], dtype=np.float64)
    cap = cap_draws / n
    probs = weights / weights.sum()
    capped = np.zeros(n, dtype=bool)
    while True:
        over = ~capped & (probs > cap * (1 + 1e-12))
        if not over.any():
            return probs
        capped |= over
        free = ~capped
        probs = np.where(capped, cap, 0.0)
        if free.any():
            remaining = 1.0 - cap * int(capped.sum())
            probs[free] = weights[free] / weights[free].sum() * remaining


def non_wan_share(probabilities, first_chars, majority="皖"):
    if len(probabilities) != len(first_chars):
        raise ValueError(f"got {len(probabilities)} probabilities for {len(first_chars)} characters")
    return float(sum(p for p, char in zip(probabilities, first_chars) if char != majority))
