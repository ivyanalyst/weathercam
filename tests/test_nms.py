import pytest

from weathercam.core.nms import nms
from weathercam.core.types import Box, Detection


def test_empty():
    assert nms([]) == []


def test_single():
    d = Detection(Box(0, 0, 10, 10), 0, 0.9)
    assert nms([d]) == [d]


def test_suppresses_overlap_keeps_higher_score():
    high = Detection(Box(0, 0, 10, 10), 0, 0.9)
    low = Detection(Box(1, 1, 11, 11), 0, 0.8)
    assert nms([low, high], 0.5) == [high]


def test_keeps_low_overlap():
    a = Detection(Box(0, 0, 10, 10), 0, 0.9)
    b = Detection(Box(5, 5, 15, 15), 0, 0.8)
    assert nms([a, b], 0.5) == [a, b]


def test_class_aware():
    car = Detection(Box(0, 0, 10, 10), 0, 0.9)
    truck = Detection(Box(1, 1, 11, 11), 1, 0.8)
    assert nms([car, truck], 0.5) == [car, truck]


def test_boundary_iou_equal_threshold_kept():
    a = Detection(Box(0, 0, 10, 10), 0, 0.9)
    b = Detection(Box(0, 0, 10, 5), 0, 0.8)
    assert nms([a, b], 0.5) == [a, b]


def test_boundary_just_below_threshold_suppresses():
    a = Detection(Box(0, 0, 10, 10), 0, 0.9)
    b = Detection(Box(0, 0, 10, 5), 0, 0.8)
    assert nms([a, b], 0.49) == [a]


def test_chain_suppression_uses_kept_boxes_only():
    a = Detection(Box(0, 0, 10, 10), 0, 0.9)
    b = Detection(Box(1, 1, 11, 11), 0, 0.8)
    c = Detection(Box(2, 2, 12, 12), 0, 0.7)
    assert nms([a, b, c], 0.5) == [a, c]


def test_output_sorted_by_score():
    a = Detection(Box(0, 0, 10, 10), 0, 0.5)
    b = Detection(Box(100, 100, 110, 110), 0, 0.9)
    c = Detection(Box(200, 200, 210, 210), 0, 0.7)
    assert nms([a, b, c]) == [b, c, a]


def test_ties_keep_input_order():
    a = Detection(Box(0, 0, 10, 10), 0, 0.5)
    b = Detection(Box(100, 100, 110, 110), 0, 0.5)
    assert nms([a, b]) == [a, b]
    assert nms([b, a]) == [b, a]


def test_input_not_mutated():
    a = Detection(Box(0, 0, 10, 10), 0, 0.5)
    b = Detection(Box(1, 1, 11, 11), 0, 0.9)
    items = [a, b]
    nms(items)
    assert items == [a, b]


def test_invalid_threshold():
    with pytest.raises(ValueError):
        nms([], -0.1)
    with pytest.raises(ValueError):
        nms([], 1.1)
