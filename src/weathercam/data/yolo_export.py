def yolo_line(label, box, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if label < 0:
        raise ValueError("label must be non-negative")
    if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
        raise ValueError("box outside image")
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def yolo_label_text(items, width, height):
    return "".join(yolo_line(label, box, width, height) + "\n" for label, box in items)
