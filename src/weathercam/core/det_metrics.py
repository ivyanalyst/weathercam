from bisect import bisect_left

from weathercam.core.boxes import iou
from weathercam.core.types import Detection

IOU_50 = (0.5,)
IOU_50_95 = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
RECALL_POINTS = 101


def _check_lengths(predictions: list[list[Detection]], ground_truths: list[list[Detection]]) -> None:
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions and ground_truths must have the same length, got {len(predictions)} and {len(ground_truths)}"
        )


def average_precision(
    predictions: list[list[Detection]],
    ground_truths: list[list[Detection]],
    label: int,
    iou_threshold: float = 0.5,
) -> float:
    _check_lengths(predictions, ground_truths)
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    gt_boxes = [[d.box for d in image if d.label == label] for image in ground_truths]
    total = sum(len(boxes) for boxes in gt_boxes)
    if total == 0:
        raise ValueError(f"No ground truth for label {label}")
    ranked = sorted(
        (
            (d.score, index, d.box)
            for index, image in enumerate(predictions)
            for d in image
            if d.label == label
        ),
        key=lambda item: -item[0],
    )
    matched = [[False] * len(boxes) for boxes in gt_boxes]
    tp = 0
    fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, image_index, box in ranked:
        best_iou = -1.0
        best_index = -1
        for j, gt in enumerate(gt_boxes[image_index]):
            if matched[image_index][j]:
                continue
            value = iou(box, gt)
            if value > best_iou:
                best_iou = value
                best_index = j
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[image_index][best_index] = True
            tp += 1
        else:
            fp += 1
        recalls.append(tp / total)
        precisions.append(tp / (tp + fp))
    for k in range(len(precisions) - 2, -1, -1):
        precisions[k] = max(precisions[k], precisions[k + 1])
    total_precision = 0.0
    for i in range(RECALL_POINTS):
        rank = bisect_left(recalls, i / 100)
        if rank < len(recalls):
            total_precision += precisions[rank]
    return total_precision / RECALL_POINTS


def ap_per_class(
    predictions: list[list[Detection]],
    ground_truths: list[list[Detection]],
    iou_thresholds: tuple[float, ...] = IOU_50_95,
) -> dict[int, float]:
    _check_lengths(predictions, ground_truths)
    if not iou_thresholds:
        raise ValueError("iou_thresholds must not be empty")
    labels = sorted({d.label for image in ground_truths for d in image})
    return {
        label: sum(average_precision(predictions, ground_truths, label, t) for t in iou_thresholds)
        / len(iou_thresholds)
        for label in labels
    }


def mean_average_precision(
    predictions: list[list[Detection]],
    ground_truths: list[list[Detection]],
    iou_thresholds: tuple[float, ...] = IOU_50_95,
) -> float:
    per_class = ap_per_class(predictions, ground_truths, iou_thresholds)
    if not per_class:
        raise ValueError("No ground truth in any image")
    return sum(per_class.values()) / len(per_class)
