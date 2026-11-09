from weathercam.core.boxes import iou


def best_detection(detections):
    return max(detections, key=lambda detection: detection.score, default=None)


def top1_ious(predictions, truths):
    if len(predictions) != len(truths):
        raise ValueError(f"length mismatch: {len(predictions)} predictions, {len(truths)} truths")
    result = []
    for detections, truth in zip(predictions, truths):
        best = best_detection(detections)
        result.append(0.0 if best is None else iou(best.box, truth))
    return result


def hit_rate(ious, threshold=0.7):
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    if not ious:
        raise ValueError("ious must not be empty")
    return sum(value > threshold for value in ious) / len(ious)
