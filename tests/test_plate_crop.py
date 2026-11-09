import numpy as np
import pytest

from weathercam.core.types import Box
from weathercam.plates.crop import crop_array, expand_box, jitter_box, pixel_box, resize_rgb

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


def test_resize_rgb_shape_and_dtype():
    out = resize_rgb(np.zeros((40, 120, 3), dtype=np.float32), 24, 48)
    assert out.shape == (24, 48, 3)
    assert out.dtype == np.float32


def test_resize_rgb_constant_image_stays_constant():
    image = np.full((30, 90, 3), 0.25, dtype=np.float32)
    assert resize_rgb(image, 12, 36) == pytest.approx(0.25, abs=1e-5)


def test_resize_rgb_output_stays_in_unit_range():
    checker = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32)
    image = np.repeat(checker[:, :, None], 3, axis=2)
    out = resize_rgb(image, 37, 53)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_rgb_same_size_is_identity():
    image = np.random.default_rng(0).random((20, 60, 3)).astype(np.float32)
    assert resize_rgb(image, 20, 60) == pytest.approx(image, abs=1e-6)


def test_resize_rgb_keeps_orientation():
    ramp = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (16, 1))
    out = resize_rgb(np.repeat(ramp[:, :, None], 3, axis=2), 8, 16)
    assert out[:, 0].mean() < out[:, -1].mean()


def test_resize_rgb_does_not_modify_input():
    image = np.random.default_rng(1).random((20, 60, 3)).astype(np.float32)
    before = image.copy()
    resize_rgb(image, 10, 30)
    assert np.array_equal(image, before)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.float32),
        np.full((4, 4, 3), 2.0, dtype=np.float32),
        [[1]],
    ],
)
def test_resize_rgb_rejects_bad_image(image):
    with pytest.raises(ValueError):
        resize_rgb(image, 4, 4)


@pytest.mark.parametrize("height, width", [(0, 10), (10, 0), (-1, 5)])
def test_resize_rgb_rejects_bad_size(height, width):
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((4, 4, 3), dtype=np.float32), height, width)
