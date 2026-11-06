import pytest

from weathercam.core.types import Box
from weathercam.data.yolo_export import yolo_label_text, yolo_line


def test_basic_line():
    assert yolo_line(2, Box(20, 10, 60, 50), 200, 100) == "2 0.200000 0.300000 0.200000 0.400000"


def test_full_image_box():
    assert yolo_line(0, Box(0, 0, 200, 100), 200, 100) == "0 0.500000 0.500000 1.000000 1.000000"


def test_rounds_to_six_decimals():
    assert yolo_line(1, Box(0, 0, 1, 1), 3, 3) == "1 0.166667 0.166667 0.333333 0.333333"


@pytest.mark.parametrize(
    "box",
    [Box(-1, 0, 10, 10), Box(0, -1, 10, 10), Box(0, 0, 201, 10), Box(0, 0, 10, 101)],
)
def test_box_outside_image_raises(box):
    with pytest.raises(ValueError):
        yolo_line(0, box, 200, 100)


@pytest.mark.parametrize("width,height", [(0, 100), (200, 0), (-1, 100)])
def test_bad_image_size_raises(width, height):
    with pytest.raises(ValueError):
        yolo_line(0, Box(0, 0, 10, 10), width, height)


def test_negative_label_raises():
    with pytest.raises(ValueError):
        yolo_line(-1, Box(0, 0, 10, 10), 200, 100)


def test_label_text_two_boxes():
    items = [(2, Box(20, 10, 60, 50)), (0, Box(0, 0, 200, 100))]
    expected = "2 0.200000 0.300000 0.200000 0.400000\n0 0.500000 0.500000 1.000000 1.000000\n"
    assert yolo_label_text(items, 200, 100) == expected


def test_label_text_empty():
    assert yolo_label_text([], 200, 100) == ""
