from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoftmaxModel:
    weights: np.ndarray
    bias: np.ndarray


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(model, x):
    return _softmax(np.asarray(x, dtype=np.float64) @ model.weights + model.bias)


def predict(model, x):
    return predict_proba(model, x).argmax(axis=1)


def fit(x, y, n_classes, l2, lr, epochs, seed):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
        raise ValueError("x must be 2-d and match y in length")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be in 0..{n_classes - 1}")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], n_classes))
    bias = np.zeros(n_classes)
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y] = 1.0
    for _ in range(epochs):
        p = _softmax(x @ weights + bias)
        diff = p - onehot
        weights = weights - lr * (x.T @ diff / len(y) + l2 * weights)
        bias = bias - lr * diff.mean(axis=0)
    return SoftmaxModel(weights, bias)
