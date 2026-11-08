from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricHead:
    weights: np.ndarray
    prototypes: np.ndarray
    scale: float


def _unit_rows(matrix):
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix / norms, norms


def loss_and_grads(weights, prototypes, x, y, scale, l2=0.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    z = x @ weights
    u, z_norm = _unit_rows(z)
    v, p_norm = _unit_rows(prototypes)
    logits = scale * (u @ v.T)
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    probs = e / e.sum(axis=1, keepdims=True)
    count = len(y)
    rows = np.arange(count)
    loss = float(-np.log(probs[rows, y]).mean() + 0.5 * l2 * np.sum(weights * weights))
    diff = probs.copy()
    diff[rows, y] -= 1.0
    diff /= count
    d_u = scale * (diff @ v)
    d_v = scale * (diff.T @ u)
    d_z = (d_u - u * np.sum(d_u * u, axis=1, keepdims=True)) / z_norm
    d_p = (d_v - v * np.sum(d_v * v, axis=1, keepdims=True)) / p_norm
    d_weights = x.T @ d_z + l2 * weights
    return loss, d_weights, d_p


def fit(x, y, n_classes, dim, scale, lr, epochs, batch_size, seed, l2=0.0, momentum=0.9):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
        raise ValueError("x must be 2-d and match y in length")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must contain integers")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be in 0..{n_classes - 1}")
    if dim < 1 or scale <= 0 or lr <= 0 or epochs < 1 or batch_size < 1:
        raise ValueError("dim, scale, lr, epochs and batch_size must be positive")
    if l2 < 0 or not 0.0 <= momentum < 1.0:
        raise ValueError("l2 must be non-negative and momentum in [0, 1)")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], dim))
    prototypes = rng.normal(0.0, 1.0, size=(n_classes, dim))
    vel_w = np.zeros_like(weights)
    vel_p = np.zeros_like(prototypes)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start : start + batch_size]
            _, d_w, d_p = loss_and_grads(weights, prototypes, x[batch], y[batch], scale, l2)
            vel_w = momentum * vel_w - lr * d_w
            vel_p = momentum * vel_p - lr * d_p
            weights = weights + vel_w
            prototypes = prototypes + vel_p
    return MetricHead(weights, prototypes, float(scale))


def project(model, x):
    z = np.asarray(x, dtype=np.float64) @ model.weights
    return _unit_rows(z)[0]
