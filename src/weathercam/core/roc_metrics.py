import numpy as np


def _average_ranks(scores):
    n = len(scores)
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.concatenate(([0], np.flatnonzero(ordered[1:] != ordered[:-1]) + 1))
    ends = np.concatenate((starts[1:], [n]))
    averages = (starts + ends + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(averages, ends - starts)
    return ranks


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores and labels must be 1-D")
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    if len(scores) == 0:
        raise ValueError("no samples")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    positive = labels.astype(np.int64) == 1
    n_pos = int(positive.sum())
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes are required")
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
