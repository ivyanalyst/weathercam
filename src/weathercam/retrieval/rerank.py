import numpy as np


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("expanded query has zero length")
    return matrix / norms


def expand_queries(gallery, queries, neighbours, weight, exclude_self=False, chunk=512):
    gallery = np.asarray(gallery, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if gallery.ndim != 2 or queries.ndim != 2:
        raise ValueError("gallery and queries must be 2-D")
    if gallery.shape[1] != queries.shape[1]:
        raise ValueError("gallery and queries differ in dimension")
    if neighbours < 0:
        raise ValueError("neighbours must be >= 0")
    if weight < 0:
        raise ValueError("weight must be >= 0")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    if exclude_self and len(gallery) != len(queries):
        raise ValueError("exclude_self needs queries and gallery of equal length")
    available = len(gallery) - 1 if exclude_self else len(gallery)
    if neighbours > available:
        raise ValueError("more neighbours than gallery rows")
    out = np.empty_like(queries)
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        block = queries[start:stop]
        if neighbours == 0 or weight == 0:
            out[start:stop] = block
            continue
        sims = block @ gallery.T
        if exclude_self:
            rows = np.arange(stop - start)
            sims[rows, rows + start] = -np.inf
        order = np.argsort(-sims, axis=1, kind="stable")[:, :neighbours]
        mean = gallery[order].mean(axis=1)
        out[start:stop] = block + weight * mean
    return _unit_rows(out)
