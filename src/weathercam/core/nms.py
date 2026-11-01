from weathercam.core.boxes import iou
from weathercam.core.types import Detection

def nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
	if not (0.0 <= iou_threshold <=1.0):
		raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
	ordered = sorted(detections, key=lambda d: d.score, reverse=True)
	kept: list[Detection] = []
	for candidate in ordered:
		suppressed = any(
			k.label == candidate.label and iou(k.box, candidate.box) > iou_threshold
			for k in kept
		)
		if not suppressed:
			kept.append(candidate)
	return kept
