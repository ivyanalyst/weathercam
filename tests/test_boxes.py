import pytest

from weathercam.core.boxes import (
    area,
    clip_box,
    from_xywh,
    from_yolo,
    intersection_area,
    iou,
    to_xywh,
    to_yolo,
)
from weathercam.core.types import Box


def test_area():
    assert area(Box(0, 0, 10, 5)) == 50


def test_area_zero():
    assert area(Box(3, 3, 3, 9)) == 0


def test_intersection_area_overlap():
    assert intersection_area(Box(0, 0, 10, 10), Box(5, 5, 15, 15)) == 25


def test_intersection_area_disjoint():
    assert intersection_area(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0


def test_iou_identical():
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_partial():
    assert iou(Box(0, 0, 10, 10), Box(5, 5, 15, 15)) == pytest.approx(1 / 7)


def test_iou_contained():
    assert iou(Box(0, 0, 10, 10), Box(2, 2, 6, 6)) == pytest.approx(0.16)


def test_iou_disjoint():
    assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0


def test_iou_touching_edge():
    assert iou(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == 0.0


def test_iou_both_zero_area():
    assert iou(Box(5, 5, 5, 5), Box(5, 5, 5, 5)) == 0.0


def test_iou_symmetric():
    a = Box(0, 0, 10, 10)
    b = Box(3, 4, 14, 9)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_clip_inside_unchanged():
    assert clip_box(Box(10, 10, 20, 20), 40, 100) == Box(10, 10, 20, 20)


def test_clip_partial():
    assert clip_box(Box(-5, -5, 50, 120), 40, 100) == Box(0, 0, 40, 100)


def test_clip_outside_zero_area():
    clipped = clip_box(Box(50, 50, 60, 60), 40, 40)
    assert clipped == Box(40, 40, 40, 40)
    assert area(clipped) == 0


def test_clip_invalid_image_size():
    with pytest.raises(ValueError):
        clip_box(Box(0, 0, 1, 1), 0, 100)


def test_to_xywh():
    assert to_xywh(Box(10, 20, 50, 80)) == (10, 20, 40, 60)


def test_from_xywh():
    assert from_xywh(10, 20, 40, 60) == Box(10, 20, 50, 80)


def test_from_xywh_negative():
    with pytest.raises(ValueError):
        from_xywh(10, 20, -4, 6)


def test_to_yolo():
    assert to_yolo(Box(50, 25, 150, 75), 200, 100) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert to_yolo(Box(0, 0, 20, 10), 200, 100) == pytest.approx((0.05, 0.05, 0.1, 0.1))


def test_from_yolo():
    box = from_yolo(0.5, 0.5, 0.5, 0.5, 200, 100)
    assert box.as_tuple() == pytest.approx((50, 25, 150, 75))


def test_yolo_roundtrip():
    original = Box(13, 7, 91, 44)
    cx, cy, w, h = to_yolo(original, 200, 100)
    assert from_yolo(cx, cy, w, h, 200, 100).as_tuple() == pytest.approx(original.as_tuple())


def test_yolo_invalid_size():
    with pytest.raises(ValueError):
        to_yolo(Box(0, 0, 1, 1), 100, 0)
    with pytest.raises(ValueError):
        from_yolo(0.5, 0.5, 0.1, 0.1, 0, 100)
