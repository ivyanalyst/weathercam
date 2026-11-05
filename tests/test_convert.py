import numpy as np
import pytest

from weathercam.core.types import Box, Detection
from weathercam.detection.convert import to_detections

LABELS = {0: 0, 2: 2}


def test_converts_boxes_scores_and_labels():
    boxes = np.array([[10, 20, 30, 40], [0, 0, 5, 5]], dtype=np.float64)
    scores = np.array([0.9, 0.5])
    labels = np.array([2, 0])
    result = to_detections(boxes, scores, labels, LABELS, 100, 100)
    assert result == [
        Detection(Box(10.0, 20.0, 30.0, 40.0), 2, 0.9),
        Detection(Box(0.0, 0.0, 5.0, 5.0), 0, 0.5),
    ]


def test_output_uses_python_scalars():
    boxes = np.array([[1, 2, 3, 4]], dtype=np.float32)
    scores = np.array([0.5], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)
    (item,) = to_detections(boxes, scores, labels, LABELS, 10, 10)
    assert type(item.label) is int
    assert type(item.score) is float
    assert type(item.box.x1) is float
    assert type(item.box.y2) is float


def test_float32_inputs_are_accepted():
    boxes = np.array([[10.5, 20.25, 30.75, 40.5]], dtype=np.float32)
    scores = np.array([0.5], dtype=np.float32)
    labels = np.array([2])
    assert to_detections(boxes, scores, labels, LABELS, 100, 100) == [
        Detection(Box(10.5, 20.25, 30.75, 40.5), 2, 0.5)
    ]


def test_labels_are_remapped_and_unmapped_are_dropped():
    boxes = np.array([[1, 1, 10, 10], [2, 2, 20, 20], [3, 3, 30, 30]], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([5, 99, 7])
    result = to_detections(boxes, scores, labels, {5: 4, 7: 6}, 100, 100)
    assert result == [
        Detection(Box(1, 1, 10, 10), 4, 0.9),
        Detection(Box(3, 3, 30, 30), 6, 0.7),
    ]


def test_boxes_are_clipped_to_the_image():
    boxes = np.array([[-5, -3, 120, 60], [90, 40, 110, 45]], dtype=np.float64)
    scores = np.array([0.9, 0.8])
    labels = np.array([2, 2])
    result = to_detections(boxes, scores, labels, LABELS, 100, 50)
    assert [d.box for d in result] == [Box(0, 0, 100, 50), Box(90, 40, 100, 45)]


def test_boxes_without_area_are_dropped():
    boxes = np.array([[10, 10, 10, 30], [10, 10, 30, 10], [20, 20, 40, 40]], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([2, 2, 2])
    assert to_detections(boxes, scores, labels, LABELS, 100, 100) == [
        Detection(Box(20, 20, 40, 40), 2, 0.7)
    ]


def test_boxes_outside_the_image_are_dropped():
    boxes = np.array(
        [[150, 10, 160, 20], [10, 150, 20, 160], [-20, 10, -5, 20]], dtype=np.float64
    )
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([2, 2, 2])
    assert to_detections(boxes, scores, labels, LABELS, 100, 100) == []


def test_input_order_is_kept():
    boxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]], dtype=np.float64)
    scores = np.array([0.2, 0.9, 0.5])
    labels = np.array([2, 2, 2])
    result = to_detections(boxes, scores, labels, LABELS, 100, 100)
    assert [d.score for d in result] == [0.2, 0.9, 0.5]


def test_empty_input_gives_empty_list():
    result = to_detections(
        np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64), LABELS, 10, 10
    )
    assert result == []


def test_all_unmapped_gives_empty_list():
    boxes = np.array([[1, 1, 5, 5], [2, 2, 6, 6]], dtype=np.float64)
    scores = np.array([0.9, 0.8])
    labels = np.array([90, 91])
    assert to_detections(boxes, scores, labels, LABELS, 100, 100) == []


def test_integral_float_labels_are_accepted():
    boxes = np.array([[1, 1, 5, 5], [2, 2, 6, 6]], dtype=np.float64)
    scores = np.array([0.9, 0.8])
    labels = np.array([2.0, 0.0])
    result = to_detections(boxes, scores, labels, LABELS, 100, 100)
    assert [d.label for d in result] == [2, 0]
    assert all(type(d.label) is int for d in result)


def test_non_integral_label_raises():
    boxes = np.array([[1, 1, 5, 5]], dtype=np.float64)
    with pytest.raises(ValueError, match="label"):
        to_detections(boxes, np.array([0.9]), np.array([2.5]), LABELS, 100, 100)


def test_non_finite_box_raises():
    scores = np.array([0.9])
    labels = np.array([2])
    with pytest.raises(ValueError, match="finite"):
        to_detections(np.array([[0, np.nan, 10, 10]]), scores, labels, LABELS, 100, 100)
    with pytest.raises(ValueError, match="finite"):
        to_detections(np.array([[0, 0, np.inf, 10]]), scores, labels, LABELS, 100, 100)


def test_non_finite_score_raises():
    boxes = np.array([[1, 1, 5, 5]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        to_detections(boxes, np.array([np.nan]), np.array([2]), LABELS, 100, 100)


def test_non_finite_row_raises_even_for_unmapped_label():
    boxes = np.array([[0, np.nan, 10, 10]])
    with pytest.raises(ValueError, match="finite"):
        to_detections(boxes, np.array([0.9]), np.array([90]), LABELS, 100, 100)


def test_inverted_box_raises():
    boxes = np.array([[30, 10, 20, 40]], dtype=np.float64)
    with pytest.raises(ValueError, match="box"):
        to_detections(boxes, np.array([0.9]), np.array([2]), LABELS, 100, 100)


def test_score_out_of_range_raises():
    boxes = np.array([[1, 1, 5, 5]], dtype=np.float64)
    with pytest.raises(ValueError, match="score"):
        to_detections(boxes, np.array([1.5]), np.array([2]), LABELS, 100, 100)
    with pytest.raises(ValueError, match="score"):
        to_detections(boxes, np.array([-0.1]), np.array([2]), LABELS, 100, 100)


def test_boxes_must_have_four_columns():
    with pytest.raises(ValueError, match="boxes"):
        to_detections(np.zeros((2, 3)), np.zeros(2), np.zeros(2, dtype=np.int64), LABELS, 10, 10)


def test_scores_length_must_match():
    with pytest.raises(ValueError, match="scores"):
        to_detections(np.zeros((2, 4)), np.zeros(3), np.zeros(2, dtype=np.int64), LABELS, 10, 10)


def test_labels_length_must_match():
    with pytest.raises(ValueError, match="labels"):
        to_detections(np.zeros((2, 4)), np.zeros(2), np.zeros(3, dtype=np.int64), LABELS, 10, 10)


def test_image_size_must_be_positive():
    boxes = np.array([[1, 1, 5, 5]], dtype=np.float64)
    with pytest.raises(ValueError, match="image size"):
        to_detections(boxes, np.array([0.9]), np.array([2]), LABELS, 0, 100)
    with pytest.raises(ValueError, match="image size"):
        to_detections(boxes, np.array([0.9]), np.array([2]), LABELS, 100, -1)


def test_empty_python_lists_give_empty_list():
    assert to_detections([], [], [], LABELS, 10, 10) == []


def test_one_dimensional_non_empty_boxes_raise():
    with pytest.raises(ValueError, match="boxes"):
        to_detections(np.zeros(4), np.zeros(1), np.zeros(1, dtype=np.int64), LABELS, 10, 10)


def test_empty_boxes_with_wrong_columns_raise():
    with pytest.raises(ValueError, match="boxes"):
        to_detections(np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64), LABELS, 10, 10)


def test_empty_boxes_with_scores_raise():
    with pytest.raises(ValueError, match="scores"):
        to_detections([], np.zeros(1), np.zeros(1, dtype=np.int64), LABELS, 10, 10)
