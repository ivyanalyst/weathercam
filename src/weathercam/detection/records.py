import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from weathercam.core.types import Box, Detection

FORMAT_VERSION = 1


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_text(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ImagePrediction:
    name: str
    width: int
    height: int
    detections: tuple

    def __post_init__(self):
        _check_text(self.name, "image name")
        if not _is_int(self.width) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if not _is_int(self.height) or self.height <= 0:
            raise ValueError("height must be a positive integer")
        if not isinstance(self.detections, (list, tuple)):
            raise ValueError("detections must be a list or tuple")
        items = tuple(self.detections)
        for item in items:
            if not isinstance(item, Detection):
                raise ValueError("detections must hold Detection objects")
        object.__setattr__(self, "detections", items)


@dataclass(frozen=True, slots=True)
class PredictionSet:
    model: str
    settings: dict
    images: tuple

    def __post_init__(self):
        _check_text(self.model, "model")
        if not isinstance(self.settings, dict):
            raise ValueError("settings must be a dict")
        for key in self.settings:
            if not isinstance(key, str):
                raise ValueError("settings keys must be strings")
        try:
            normal = json.loads(json.dumps(self.settings, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(f"settings are not JSON serialisable: {error}")
        object.__setattr__(self, "settings", normal)
        if not isinstance(self.images, (list, tuple)):
            raise ValueError("images must be a list or tuple")
        items = tuple(self.images)
        names = set()
        for item in items:
            if not isinstance(item, ImagePrediction):
                raise ValueError("images must hold ImagePrediction objects")
            if item.name in names:
                raise ValueError(f"duplicate image name: {item.name}")
            names.add(item.name)
        object.__setattr__(self, "images", items)


def _detection_to_dict(detection):
    return {
        "box": [float(v) for v in detection.box.as_tuple()],
        "label": int(detection.label),
        "score": float(detection.score),
    }


def _to_payload(predictions):
    return {
        "format": FORMAT_VERSION,
        "model": predictions.model,
        "settings": predictions.settings,
        "images": [
            {
                "name": image.name,
                "width": image.width,
                "height": image.height,
                "detections": [_detection_to_dict(d) for d in image.detections],
            }
            for image in predictions.images
        ],
    }


def save_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_payload(predictions), sort_keys=True, allow_nan=False)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _require(mapping, keys, what):
    if not isinstance(mapping, dict):
        raise ValueError(f"{what} must be an object")
    for key in keys:
        if key not in mapping:
            raise ValueError(f"{what} is missing key: {key}")


def _parse_detection(raw):
    _require(raw, ("box", "label", "score"), "detection")
    box = raw["box"]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("box must be a list of four numbers")
    for value in box:
        if not _is_number(value):
            raise ValueError("box values must be numbers")
    label = raw["label"]
    if not _is_int(label) or label < 0:
        raise ValueError("label must be a non-negative integer")
    score = raw["score"]
    if not _is_number(score) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a number in [0, 1]")
    return Detection(Box(*[float(v) for v in box]), label, float(score))


def _parse_image(raw):
    _require(raw, ("name", "width", "height", "detections"), "image")
    if not isinstance(raw["detections"], list):
        raise ValueError("detections must be a list")
    detections = tuple(_parse_detection(item) for item in raw["detections"])
    return ImagePrediction(raw["name"], raw["width"], raw["height"], detections)


def load_predictions(path):
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    _require(payload, ("format", "model", "settings", "images"), "file")
    version = payload["format"]
    if not _is_int(version) or version != FORMAT_VERSION:
        raise ValueError(f"unsupported format version: {version!r}")
    if not isinstance(payload["images"], list):
        raise ValueError("images must be a list")
    images = tuple(_parse_image(item) for item in payload["images"])
    return PredictionSet(payload["model"], payload["settings"], images)
