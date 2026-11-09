import numpy as np
import pytest

from weathercam.core.types import Box
from weathercam.plates.crop import crop_array, expand_box, jitter_box, pixel_box

BOX = Box(100, 200, 300, 260)


def test_expand_box_adds_margin_on_each_side():
    assert expand_box(BOX, 0.1, 720, 1160).as_tuple() == pytest.approx((80, 194, 320, 266))


def test_expand_box_zero_margin_keeps_box():
    assert expand_box(BOX, 0.0, 720, 1160) == BOX


def test_expand_box_clips_to_image():
    assert expand_box(Box(10, 10, 110, 50), 0.5, 720, 1160).as_tuple() == pytest.approx((0, 0, 160, 70))


def test_expand_box_negative_margin_raises():
    with pytest.raises(ValueError):
        expand_box(BOX, -0.1, 720, 1160)


def test_pixel_box_integer_box_unchanged():
    assert pixel_box(Box(10, 20, 30, 40), 720, 1160) == (10, 20, 30, 40)


def test_pixel_box_rounds_outward():
    assert pixel_box(Box(10.2, 20.7, 30.1, 40.0), 720, 1160) == (10, 20, 31, 40)


def test_pixel_box_clips_to_image():
    assert pixel_box(Box(-5, -5, 50, 50), 720, 1160) == (0, 0, 50, 50)
    assert pixel_box(Box(700, 1150, 800, 1200), 720, 1160) == (700, 1150, 720, 1160)


@pytest.mark.parametrize("box", [Box(10, 10, 10, 30), Box(800, 10, 900, 50)])
def test_pixel_box_empty_raises(box):
    with pytest.raises(ValueError):
        pixel_box(box, 720, 1160)


def test_crop_array_slices_rows_then_columns():
    image = np.arange(10 * 20 * 3).reshape(10, 20, 3)
    crop = crop_array(image, Box(2, 3, 6, 8))
    assert crop.shape == (5, 4, 3)
    assert np.array_equal(crop, image[3:8, 2:6])


def test_crop_array_clips_to_image_size():
    image = np.zeros((10, 20, 3))
    assert crop_array(image, Box(15, 5, 30, 9)).shape == (4, 5, 3)


def test_crop_array_returns_copy():
    image = np.arange(10 * 20 * 3).reshape(10, 20, 3)
    before = image.copy()
    crop = crop_array(image, Box(2, 3, 6, 8))
    crop[...] = -1
    assert np.array_equal(image, before)


@pytest.mark.parametrize("image", [[[1]], np.zeros((4, 4))])
def test_crop_array_rejects_bad_input(image):
    with pytest.raises(ValueError):
        crop_array(image, Box(0, 0, 2, 2))


def test_jitter_box_zero_is_identity():
    result = jitter_box(BOX, np.random.default_rng(0), 0.0, 0.0)
    assert result.as_tuple() == pytest.approx(BOX.as_tuple())


def test_jitter_box_stays_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        box = jitter_box(BOX, rng, 0.1, 0.2)
        assert abs((box.x1 + box.x2) / 2 - 200) <= 0.1 * BOX.width + 1e-9
        assert abs((box.y1 + box.y2) / 2 - 230) <= 0.1 * BOX.height + 1e-9
        assert 0.8 - 1e-9 <= box.width / BOX.width <= 1.2 + 1e-9
        assert 0.8 - 1e-9 <= box.height / BOX.height <= 1.2 + 1e-9


def test_jitter_box_same_seed_is_reproducible():
    first = jitter_box(BOX, np.random.default_rng(5), 0.1, 0.2)
    second = jitter_box(BOX, np.random.default_rng(5), 0.1, 0.2)
    assert first == second


def test_jitter_box_different_seed_changes_result():
    first = jitter_box(BOX, np.random.default_rng(1), 0.1, 0.2)
    second = jitter_box(BOX, np.random.default_rng(2), 0.1, 0.2)
    assert first != second


@pytest.mark.parametrize("shift, scale", [(-0.1, 0.1), (0.1, -0.1), (0.1, 1.0)])
def test_jitter_box_invalid_arguments_raise(shift, scale):
    with pytest.raises(ValueError):
        jitter_box(BOX, np.random.default_rng(0), shift, scale)
