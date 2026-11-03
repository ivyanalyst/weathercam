import pytest

from weathercam.core.boxes import area, from_voc, to_voc
from weathercam.core.types import Box


def test_from_voc_adds_one_to_max_corner():
    assert from_voc(10, 20, 30, 40) == Box(10, 20, 31, 41)


def test_from_voc_single_pixel_box():
    box = from_voc(5, 7, 5, 7)
    assert box == Box(5, 7, 6, 8)
    assert box.width == 1
    assert box.height == 1


def test_from_voc_full_frame():
    assert from_voc(0, 0, 1279, 719) == Box(0, 0, 1280, 720)


def test_from_voc_area():
    assert area(from_voc(0, 0, 7, 9)) == 80


def test_from_voc_rejects_inverted_x():
    with pytest.raises(ValueError):
        from_voc(10, 0, 9, 5)


def test_from_voc_rejects_inverted_y():
    with pytest.raises(ValueError):
        from_voc(0, 10, 5, 9)


def test_from_voc_rejects_non_finite():
    with pytest.raises(ValueError):
        from_voc(float("nan"), 0, 5, 5)
    with pytest.raises(ValueError):
        from_voc(0, 0, float("inf"), 5)


def test_to_voc_subtracts_one():
    assert to_voc(Box(5, 7, 6, 8)) == (5, 7, 5, 7)


def test_to_voc_full_frame():
    assert to_voc(Box(0, 0, 1280, 720)) == (0, 0, 1279, 719)


def test_to_voc_rejects_zero_width():
    with pytest.raises(ValueError):
        to_voc(Box(5, 7, 5, 9))


def test_to_voc_rejects_zero_height():
    with pytest.raises(ValueError):
        to_voc(Box(5, 7, 8, 7))


def test_voc_round_trip():
    assert to_voc(from_voc(12, 34, 56, 78)) == (12, 34, 56, 78)
