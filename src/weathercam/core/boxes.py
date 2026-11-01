from weathercam.core.types import Box


def area(box: Box) -> float:
    return box.width * box.height


def intersection_area(a: Box, b: Box) -> float:
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def clip_box(box: Box, width: float, height: float) -> Box:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {width}x{height}")
    return Box(
        min(max(box.x1, 0.0), width),
        min(max(box.y1, 0.0), height),
        min(max(box.x2, 0.0), width),
        min(max(box.y2, 0.0), height),
    )


def to_xywh(box: Box) -> tuple[float, float, float, float]:
    return (box.x1, box.y1, box.width, box.height)


def from_xywh(x: float, y: float, w: float, h: float) -> Box:
    if w < 0 or h < 0:
        raise ValueError(f"Width and height must be >= 0, got {w}x{h}")
    return Box(x, y, x + w, y + h)


def to_yolo(box: Box, img_w: float, img_h: float) -> tuple[float, float, float, float]:
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image size must be positive, got {img_w}x{img_h}")
    cx = (box.x1 + box.x2) / 2 / img_w
    cy = (box.y1 + box.y2) / 2 / img_h
    return (cx, cy, box.width / img_w, box.height / img_h)


def from_yolo(cx: float, cy: float, w: float, h: float, img_w: float, img_h: float) -> Box:
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image size must be positive, got {img_w}x{img_h}")
    half_w = w * img_w / 2
    half_h = h * img_h / 2
    center_x = cx * img_w
    center_y = cy * img_h
    return Box(center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h)
