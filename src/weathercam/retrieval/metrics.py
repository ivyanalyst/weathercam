from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalScore:
    precision_at_1: float
    map_at_r: float
    scored: int
    skipped: int


def average_precision_at_r(hits: Sequence[bool] | np.ndarray, r: int) -> float:
    if r < 1:
        raise ValueError("r must be at least 1")
    flags = np.asarray(hits, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("hits must be one-dimensional")
    if flags.shape[0] < r:
        raise ValueError("hits must contain at least r results")
    top = flags[:r]
    cumulative = np.cumsum(top)
    ranks = np.arange(1, r + 1)
    return float(np.sum(cumulative[top] / ranks[top]) / r)


def evaluate_rankings(
    ranked_hits: Sequence[Sequence[bool] | np.ndarray],
    relevant_counts: Sequence[int],
) -> RetrievalScore:
    if len(ranked_hits) != len(relevant_counts):
        raise ValueError("ranked_hits and relevant_counts must have the same length")
    if len(ranked_hits) == 0:
        raise ValueError("no queries")
    first_sum = 0.0
    ap_sum = 0.0
    scored = 0
    skipped = 0
    for hits, count in zip(ranked_hits, relevant_counts):
        if count < 0:
            raise ValueError("relevant counts must not be negative")
        if count == 0:
            skipped += 1
            continue
        ap_sum += average_precision_at_r(hits, count)
        first_sum += 1.0 if bool(hits[0]) else 0.0
        scored += 1
    if scored == 0:
        raise ValueError("every query was skipped")
    return RetrievalScore(first_sum / scored, ap_sum / scored, scored, skipped)
