from dataclasses import dataclass

import numpy as np

PARTS = ("train", "val", "test")


@dataclass(frozen=True)
class WeatherData:
    embeddings: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    weathers: tuple
    image_names: tuple


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be in 0..n")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def load_weather(path):
    with np.load(path, allow_pickle=False) as d:
        embeddings = d["embeddings"]
        labels = d["weather_ids"]
        split = d["split"]
        weathers = tuple(str(w) for w in d["weathers"])
        names = tuple(str(n) for n in d["image_names"])
    n = len(embeddings)
    if embeddings.ndim != 2 or not (len(labels) == len(split) == len(names) == n):
        raise ValueError("embeddings, weather_ids, split and image_names must match in length")
    if n == 0:
        raise ValueError("no rows")
    if labels.min() < 0 or labels.max() >= len(weathers):
        raise ValueError("weather_ids outside the class list")
    unknown = set(split.tolist()) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")
    return WeatherData(embeddings, labels, split, weathers, names)


def part_arrays(data, name):
    if name not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {name!r}")
    mask = data.split == name
    return data.embeddings[mask], data.labels[mask]


def majority_accuracy(y_train, y_eval):
    y_train = np.asarray(y_train)
    y_eval = np.asarray(y_eval)
    if len(y_train) == 0 or len(y_eval) == 0:
        raise ValueError("y_train and y_eval must not be empty")
    majority = np.bincount(y_train).argmax()
    return float(np.mean(y_eval == majority))
