import numpy as np


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length vector")
    return matrix / norms


def knn_distances(bank, queries, k, chunk=512):
    bank = np.asarray(bank, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if bank.ndim != 2 or queries.ndim != 2:
        raise ValueError("bank and queries must be 2-D")
    if bank.shape[1] != queries.shape[1]:
        raise ValueError("bank and queries differ in dimension")
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > len(bank):
        raise ValueError("k exceeds bank size")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    bank = _unit_rows(bank)
    queries = _unit_rows(queries)
    position = len(bank) - k
    out = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        sims = queries[start:stop] @ bank.T
        kth = np.partition(sims, position, axis=1)[:, position]
        out[start:stop] = np.sqrt(np.clip(2.0 - 2.0 * kth, 0.0, None))
    return out
