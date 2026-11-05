from collections import Counter
from dataclasses import dataclass

from weathercam.core.det_metrics import IOU_50, IOU_50_95, ap_per_class
from weathercam.core.nms import nms
from weathercam.core.types import Detection
from weathercam.data.dawn import map_classes
from weathercam.detection.classmap import CLASS_NAMES

VEHICLE_LABEL = len(CLASS_NAMES)
VEHICLE_SOURCE_LABELS = frozenset(
    CLASS_NAMES.index(name) for name in ("car", "motorcycle", "bus", "truck")
)


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: int
    boxes: int
    ap50: float
    ap50_95: float


@dataclass(frozen=True, slots=True)
class ImageSetScore:
    images: int
    classes: tuple
    vehicle: object


def ground_truth_detections(annotation, mapping):
    return [
        Detection(box, label, 1.0) for label, box in map_classes(annotation, mapping)
    ]


def merge_vehicle(detections, nms_iou=0.5):
    merged = [
        Detection(d.box, VEHICLE_LABEL, d.score)
        for d in detections
        if d.label in VEHICLE_SOURCE_LABELS
    ]
    return nms(merged, nms_iou)


def _relabel_vehicle(detections):
    return [
        Detection(d.box, VEHICLE_LABEL, d.score)
        for d in detections
        if d.label in VEHICLE_SOURCE_LABELS
    ]


def score_images(predictions, ground_truths, vehicle_nms_iou=0.5):
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")
    counts = Counter(d.label for image in ground_truths for d in image)
    ap50 = ap_per_class(predictions, ground_truths, IOU_50)
    ap50_95 = ap_per_class(predictions, ground_truths, IOU_50_95)
    classes = tuple(
        ClassScore(label, counts[label], float(ap50[label]), float(ap50_95[label]))
        for label in sorted(ap50)
    )
    vehicle_gts = [_relabel_vehicle(image) for image in ground_truths]
    vehicle_boxes = sum(len(image) for image in vehicle_gts)
    vehicle = None
    if vehicle_boxes:
        vehicle_preds = [merge_vehicle(image, vehicle_nms_iou) for image in predictions]
        v50 = ap_per_class(vehicle_preds, vehicle_gts, IOU_50)[VEHICLE_LABEL]
        v5095 = ap_per_class(vehicle_preds, vehicle_gts, IOU_50_95)[VEHICLE_LABEL]
        vehicle = ClassScore(VEHICLE_LABEL, vehicle_boxes, float(v50), float(v5095))
    return ImageSetScore(len(ground_truths), classes, vehicle)


def score_by_weather(weathers, predictions, ground_truths, vehicle_nms_iou=0.5):
    if not len(weathers) == len(predictions) == len(ground_truths):
        raise ValueError("weathers, predictions and ground_truths must match in length")
    for weather in weathers:
        if not isinstance(weather, str) or not weather:
            raise ValueError("weather names must be non-empty strings")
    groups = {}
    for weather, prediction, truth in zip(weathers, predictions, ground_truths):
        entry = groups.setdefault(weather, ([], []))
        entry[0].append(prediction)
        entry[1].append(truth)
    return {
        weather: score_images(preds, gts, vehicle_nms_iou)
        for weather, (preds, gts) in groups.items()
    }
