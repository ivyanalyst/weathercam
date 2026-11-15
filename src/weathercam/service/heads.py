from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict_proba
from weathercam.retrieval.metric_head import MetricHead, project

COLOR_NAMES = (
    "yellow", "orange", "green", "gray", "red",
    "blue", "white", "golden", "brown", "black",
)
TYPE_NAMES = (
    "sedan", "suv", "van", "hatchback", "mpv",
    "pickup", "bus", "truck", "estate",
)


class Prediction(NamedTuple):
    class_id: int
    label: str
    probability: float
    top: tuple


def standardise(x, mean, std):
    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError("x must be two-dimensional")
    if array.shape[1] != len(mean):
        raise ValueError(f"x must have {len(mean)} columns, got {array.shape[1]}")
    if not np.all(np.isfinite(array)):
        raise ValueError("x must be finite")
    return (array - mean) / std


def _check_scale(mean, std, dim):
    if mean.shape != (dim,) or std.shape != (dim,):
        raise ValueError(f"mean and std must have shape ({dim},)")
    if not np.all(std > 0):
        raise ValueError("std must be positive")


@dataclass(frozen=True)
class LinearHead:
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    class_ids: np.ndarray
    names: tuple

    def __post_init__(self):
        if self.weights.ndim != 2:
            raise ValueError("weights must be two-dimensional")
        dim, classes = self.weights.shape
        if self.bias.shape != (classes,):
            raise ValueError(f"bias must have shape ({classes},)")
        if self.class_ids.shape != (classes,):
            raise ValueError(f"class_ids must have shape ({classes},)")
        if len(self.names) != classes:
            raise ValueError(f"names must have {classes} entries")
        _check_scale(self.mean, self.std, dim)


def classify(head, x, top_k=3):
    classes = len(head.class_ids)
    if not 1 <= top_k <= classes:
        raise ValueError(f"top_k must be in [1, {classes}], got {top_k}")
    z = standardise(x, head.mean, head.std)
    probabilities = predict_proba(SoftmaxModel(head.weights, head.bias), z)
    result = []
    for row in probabilities:
        order = np.argsort(-row, kind="stable")[:top_k]
        top = tuple((int(head.class_ids[i]), head.names[i], float(row[i])) for i in order)
        result.append(Prediction(top[0][0], top[0][1], top[0][2], top))
    return result


def _read(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            if key not in archive.files:
                raise ValueError(f"{path}: missing key {key!r}")
        return {key: archive[key] for key in keys}


def load_linear_head(path, names=None, names_key=None):
    if names is None and names_key is None:
        raise ValueError("names or names_key is required")
    keys = ["weights", "bias", "mean", "std", "class_ids"]
    if names is None:
        keys.append(names_key)
    data = _read(path, keys)
    source = names if names is not None else data[names_key]
    labels = tuple(str(v) for v in source)
    return LinearHead(
        data["weights"], data["bias"], data["mean"], data["std"], data["class_ids"], labels
    )


class MetricProjector:
    def __init__(self, weights, mean, std):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if self.weights.ndim != 2:
            raise ValueError("weights must be two-dimensional")
        _check_scale(self.mean, self.std, self.weights.shape[0])
        self._model = MetricHead(self.weights, np.empty((0, self.weights.shape[1])), 1.0)

    def project(self, x):
        z = standardise(np.asarray(x, dtype=np.float64), self.mean, self.std)
        return project(self._model, z)


def load_metric_head(path):
    data = _read(path, ("weights", "mean", "std"))
    return MetricProjector(data["weights"], data["mean"], data["std"])
