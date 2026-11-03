import numpy as np
import pytest

from weathercam.core.types import Box
from weathercam.data.overlay import draw_boxes

RED = (1.0, 0.0, 0.0)


def blank(height=10, width=10):
    return np.zeros((height, width, 3), dtype=np.float32)


def painted(image):
    ys, xs = np.nonzero(image.any(axis=2))
    return {(int(y), int(x)) for y, x in zip(ys, xs)}


def outline(x1, y1, x2, y2):
    return {
        (y, x)
        for y in range(y1, y2)
        for x in range(x1, x2)
        if y in (y1, y2 - 1) or x in (x1, x2 - 1)
    }


def test_returns_new_float32_array():
    image = blank()
    out = draw_boxes(image, [Box(2, 3, 6, 8)], RED)
    assert out is not image
    assert out.dtype == np.float32
    assert out.shape == image.shape


def test_input_not_modified():
    image = blank()
    draw_boxes(image, [Box(2, 3, 6, 8)], RED)
    assert not image.any()


def test_outline_pixels_half_open():
    out = draw_boxes(blank(), [Box(2, 3, 6, 8)], RED)
    assert painted(out) == outline(2, 3, 6, 8)


def test_pixels_past_half_open_edge_untouched():
    out = draw_boxes(blank(), [Box(2, 3, 6, 8)], RED)
    assert not out[:, 6:, :].any()
    assert not out[8:, :, :].any()
    assert not out[:3, :, :].any()
    assert not out[:, :2, :].any()


def test_interior_untouched():
    out = draw_boxes(blank(), [Box(2, 3, 6, 8)], RED)
    assert not out[4:7, 3:5, :].any()


def test_color_written():
    out = draw_boxes(blank(), [Box(2, 3, 6, 8)], (0.25, 0.5, 0.75))
    assert tuple(out[3, 2]) == (0.25, 0.5, 0.75)


def test_voc_box_covers_inclusive_pixels():
    from weathercam.core.boxes import from_voc

    out = draw_boxes(blank(), [from_voc(2, 3, 5, 7)], RED)
    assert painted(out) == outline(2, 3, 6, 8)


def test_thickness_two_grows_inward():
    out = draw_boxes(blank(), [Box(1, 1, 9, 9)], RED, thickness=2)
    expected = {
        (y, x)
        for y in range(1, 9)
        for x in range(1, 9)
        if not (3 <= y <= 6 and 3 <= x <= 6)
    }
    assert painted(out) == expected


def test_thick_outline_fills_small_box():
    out = draw_boxes(blank(), [Box(2, 2, 5, 5)], RED, thickness=5)
    assert painted(out) == {(y, x) for y in range(2, 5) for x in range(2, 5)}


def test_clipping_keeps_visible_part_of_real_outline():
    out = draw_boxes(blank(), [Box(-3, -3, 5, 5)], RED)
    expected = {(y, 4) for y in range(0, 5)} | {(4, x) for x in range(0, 5)}
    assert painted(out) == expected


def test_box_wholly_outside_draws_nothing():
    out = draw_boxes(blank(), [Box(20, 20, 25, 25)], RED)
    assert not out.any()


def test_box_past_right_edge_is_clipped():
    out = draw_boxes(blank(), [Box(7, 2, 14, 5)], RED)
    expected = {(2, x) for x in range(7, 10)} | {(4, x) for x in range(7, 10)}
    expected |= {(y, 7) for y in range(2, 5)}
    assert painted(out) == expected


def test_rounds_to_nearest_pixel():
    out = draw_boxes(blank(), [Box(2.4, 3.4, 5.6, 7.6)], RED)
    assert painted(out) == outline(2, 3, 6, 8)


def test_zero_size_box_draws_nothing():
    out = draw_boxes(blank(), [Box(3, 3, 3, 3)], RED)
    assert not out.any()


def test_empty_box_list_returns_equal_copy():
    image = blank()
    image[1, 1] = (0.1, 0.2, 0.3)
    out = draw_boxes(image, [], RED)
    assert out is not image
    assert np.array_equal(out, image)


def test_multiple_boxes_union():
    out = draw_boxes(blank(), [Box(0, 0, 3, 3), Box(5, 5, 9, 9)], RED)
    assert painted(out) == outline(0, 0, 3, 3) | outline(5, 5, 9, 9)


def test_later_box_overwrites_earlier():
    out = draw_boxes(blank(), [Box(0, 0, 4, 4), Box(0, 0, 4, 4)], (0.0, 1.0, 0.0))
    assert tuple(out[0, 0]) == (0.0, 1.0, 0.0)


def test_float64_input_accepted():
    image = np.zeros((10, 10, 3), dtype=np.float64)
    out = draw_boxes(image, [Box(2, 3, 6, 8)], RED)
    assert out.dtype == np.float32


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.float32),
        np.full((10, 10, 3), 1.5, dtype=np.float32),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
        [[0.0, 0.0, 0.0]],
    ],
)
def test_invalid_image_raises(image):
    with pytest.raises(ValueError):
        draw_boxes(image, [Box(1, 1, 3, 3)], RED)


@pytest.mark.parametrize(
    "color",
    [(1.0, 0.0), (1.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-0.1, 0.0, 0.0), (float("nan"), 0.0, 0.0)],
)
def test_invalid_color_raises(color):
    with pytest.raises(ValueError):
        draw_boxes(blank(), [Box(1, 1, 3, 3)], color)


@pytest.mark.parametrize("thickness", [0, -1])
def test_invalid_thickness_raises(thickness):
    with pytest.raises(ValueError):
        draw_boxes(blank(), [Box(1, 1, 3, 3)], RED, thickness=thickness)
