import numpy as np


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def class_vectors(text):
    text = np.asarray(text, dtype=np.float64)
    if text.ndim != 3:
        raise ValueError("text must have shape (classes, prompts, dim)")
    return _unit(_unit(text).mean(axis=1))


def image_probabilities(images, class_vecs, scale):
    logits = scale * (np.asarray(images, dtype=np.float64) @ class_vecs.T)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_tracks(probs, keys):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] != len(keys):
        raise ValueError("probs and keys differ in length")
    order = {}
    idx = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        idx[i] = order.setdefault(k, len(order))
    sums = np.zeros((len(order), probs.shape[1]))
    np.add.at(sums, idx, probs)
    counts = np.bincount(idx, minlength=len(order))
    return list(order), sums / counts[:, None], counts


def track_predictions(mean_probs):
    labels = mean_probs.argmax(axis=1)
    conf = mean_probs[np.arange(len(labels)), labels]
    return labels, conf
