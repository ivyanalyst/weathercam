import pytest

from weathercam.core.types import Box, Detection
from weathercam.plates.detector import best_detection, hit_rate, top1_ious

TRUTH = Box(100, 200, 300, 260)


def det(box, score):
    return Detection(box, 0, score)


def test_best_detection_picks_highest_score():
    low, high = det(Box(0, 0, 1, 1), 0.2), det(Box(5, 5, 9, 9), 0.9)
    assert best_detection([low, high]) is high


def test_best_detection_tie_keeps_first():
    first, second = det(Box(0, 0, 1, 1), 0.5), det(Box(5, 5, 9, 9), 0.5)
    assert best_detection([first, second]) is first


def test_best_detection_empty_is_none():
    assert best_detection([]) is None


def test_top1_ious_exact_match():
    assert top1_ious([[det(TRUTH, 0.9)]], [TRUTH]) == [1.0]


def test_top1_ious_no_detection_is_zero():
    assert top1_ious([[]], [TRUTH]) == [0.0]


def test_top1_ious_uses_score_not_overlap():
    wrong_but_confident = det(Box(500, 500, 600, 560), 0.9)
    right_but_unsure = det(TRUTH, 0.5)
    assert top1_ious([[right_but_unsure, wrong_but_confident]], [TRUTH]) == [0.0]


def test_top1_ious_partial_overlap():
    assert top1_ious([[det(Box(0, 0, 10, 5), 0.8)]], [Box(0, 0, 10, 10)]) == pytest.approx([0.5])


def test_top1_ious_one_value_per_image():
    result = top1_ious([[det(TRUTH, 0.9)], []], [TRUTH, TRUTH])
    assert result == [1.0, 0.0]


def test_top1_ious_length_mismatch_raises():
    with pytest.raises(ValueError):
        top1_ious([[det(TRUTH, 0.9)]], [TRUTH, TRUTH])


def test_hit_rate_is_strictly_above_threshold():
    assert hit_rate([0.7, 0.71, 0.3]) == pytest.approx(1 / 3)


def test_hit_rate_custom_threshold():
    assert hit_rate([0.7, 0.71, 0.3], threshold=0.5) == pytest.approx(2 / 3)


def test_hit_rate_empty_raises():
    with pytest.raises(ValueError):
        hit_rate([])


@pytest.mark.parametrize("threshold", [-0.1, 1.0])
def test_hit_rate_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError):
        hit_rate([0.5], threshold=threshold)
