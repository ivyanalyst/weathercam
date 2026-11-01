import pytest

from weathercam.core.det_metrics import (
    IOU_50,
    IOU_50_95,
    ap_per_class,
    average_precision,
    mean_average_precision,
)
from weathercam.core.types import Box, Detection

A = Box(0, 0, 10, 10)
B = Box(100, 100, 110, 110)
FAR = Box(50, 50, 60, 60)


def test_perfect_detection():
    gts = [[Detection(A, 0)]]
    preds = [[Detection(A, 0, 0.9)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(1.0)


def test_no_predictions():
    gts = [[Detection(A, 0)]]
    assert average_precision([[]], gts, 0, 0.5) == 0.0


def test_false_positive_before_true_positive():
    gts = [[Detection(A, 0)]]
    preds = [[Detection(FAR, 0, 0.9), Detection(A, 0, 0.8)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(0.5)


def test_false_positive_after_full_recall():
    gts = [[Detection(A, 0)]]
    preds = [[Detection(A, 0, 0.9), Detection(FAR, 0, 0.8)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(1.0)


def test_partial_recall():
    gts = [[Detection(A, 0), Detection(B, 0)]]
    preds = [[Detection(A, 0, 0.9)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(51 / 101)


def test_duplicate_detection_is_false_positive():
    gts = [[Detection(A, 0), Detection(B, 0)]]
    preds = [[Detection(A, 0, 0.9), Detection(A, 0, 0.8), Detection(B, 0, 0.7)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(253 / 303)


def test_iou_below_threshold_not_matched():
    gts = [[Detection(Box(0, 0, 100, 100), 0)]]
    preds = [[Detection(Box(0, 0, 100, 82), 0, 0.9)]]
    assert average_precision(preds, gts, 0, 0.9) == 0.0


def test_iou_above_threshold_matched():
    gts = [[Detection(Box(0, 0, 100, 100), 0)]]
    preds = [[Detection(Box(0, 0, 100, 82), 0, 0.9)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(1.0)


def test_iou_equal_threshold_matched():
    gts = [[Detection(Box(0, 0, 10, 10), 0)]]
    preds = [[Detection(Box(0, 0, 10, 5), 0, 0.9)]]
    assert average_precision(preds, gts, 0, 0.5) == pytest.approx(1.0)


def test_matches_best_iou_ground_truth():
    gts = [[Detection(Box(0, 0, 100, 100), 0), Detection(Box(0, 0, 100, 70), 0)]]
    preds = [[Detection(Box(0, 0, 100, 80), 0, 0.9), Detection(Box(0, 0, 100, 100), 0, 0.8)]]
    assert average_precision(preds, gts, 0, 0.75) == pytest.approx(1.0)


def test_detections_do_not_match_across_images():
    gts = [[Detection(A, 0)], []]
    preds = [[], [Detection(A, 0, 0.9)]]
    assert average_precision(preds, gts, 0, 0.5) == 0.0


def test_label_mismatch_not_matched():
    gts = [[Detection(A, 0)]]
    preds = [[Detection(A, 1, 0.9)]]
    assert average_precision(preds, gts, 0, 0.5) == 0.0


def test_average_precision_no_ground_truth_raises():
    gts = [[Detection(A, 0)]]
    preds = [[Detection(A, 1, 0.9)]]
    with pytest.raises(ValueError):
        average_precision(preds, gts, 1, 0.5)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        average_precision([[]], [[], []], 0, 0.5)


def test_invalid_threshold_raises():
    gts = [[Detection(A, 0)]]
    with pytest.raises(ValueError):
        average_precision([[]], gts, 0, 0.0)
    with pytest.raises(ValueError):
        average_precision([[]], gts, 0, 1.5)


def test_ap_per_class_values():
    gts = [[Detection(A, 0), Detection(B, 1)]]
    preds = [[Detection(A, 0, 0.9)]]
    assert ap_per_class(preds, gts, IOU_50) == pytest.approx({0: 1.0, 1: 0.0})


def test_ap_per_class_skips_labels_without_ground_truth():
    gts = [[Detection(A, 0)]]
    preds = [[Detection(A, 0, 0.9), Detection(FAR, 2, 0.8)]]
    assert sorted(ap_per_class(preds, gts, IOU_50)) == [0]


def test_map_two_classes():
    gts = [[Detection(A, 0), Detection(B, 1)]]
    preds = [[Detection(A, 0, 0.9)]]
    assert mean_average_precision(preds, gts, IOU_50) == pytest.approx(0.5)


def test_map_50_95():
    gts = [[Detection(Box(0, 0, 100, 100), 0)]]
    preds = [[Detection(Box(0, 0, 100, 82), 0, 0.9)]]
    assert mean_average_precision(preds, gts, IOU_50_95) == pytest.approx(0.7)
    assert mean_average_precision(preds, gts) == pytest.approx(0.7)


def test_threshold_constants():
    assert IOU_50 == (0.5,)
    assert len(IOU_50_95) == 10
    assert IOU_50_95[0] == 0.5
    assert IOU_50_95[-1] == 0.95
    assert IOU_50_95[6] == 0.8


def test_map_no_ground_truth_raises():
    with pytest.raises(ValueError):
        mean_average_precision([[Detection(A, 0, 0.9)]], [[]], IOU_50)


def test_map_empty_thresholds_raises():
    gts = [[Detection(A, 0)]]
    with pytest.raises(ValueError):
        mean_average_precision([[]], gts, ())
