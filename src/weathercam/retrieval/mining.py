import numpy as np


def k_center_greedy(features, budget, start=0):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be 2-D")
    n = len(features)
    if budget < 1 or budget > n:
        raise ValueError("budget must be between 1 and the number of rows")
    if start < 0 or start >= n:
        raise ValueError("start out of range")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length vector")
    unit = features / norms
    selected = [start]
    closest = unit @ unit[start]
    closest[start] = np.inf
    while len(selected) < budget:
        nxt = int(np.argmin(closest))
        selected.append(nxt)
        closest = np.maximum(closest, unit @ unit[nxt])
        closest[nxt] = np.inf
    return np.array(selected, dtype=np.int64)
