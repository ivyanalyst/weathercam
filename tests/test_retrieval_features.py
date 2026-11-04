import numpy as np
import pytest

from weathercam.retrieval.features import color_histogram, color_histograms


def _constant(colour, height=3, width=3):
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[...] = np.asarray(colour, dtype=np.float32)
    return image


def _random(seed, shape):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


BAD_IMAGES = [
    [[0.0]],
    np.zeros((4, 4), dtype=np.float32),
    np.zeros((4, 4, 4), dtype=np.float32),
    np.zeros((4, 4, 3), dtype=np.uint8),
    np.full((2, 2, 3), np.nan, dtype=np.float32),
    np.full((2, 2, 3), 1.5, dtype=np.float32),
    np.full((2, 2, 3), -0.1, dtype=np.float32),
    np.zeros((0, 4, 3), dtype=np.float32),
]


def test_default_shape_and_dtype():
    vector = color_histogram(_random(0, (5, 7, 3)))
    assert vector.shape == (512,)
    assert vector.dtype == np.float32


def test_sums_to_one():
    vector = color_histogram(_random(1, (9, 11, 3)))
    assert float(vector.sum()) == pytest.approx(1.0, abs=1e-6)


def test_black_goes_to_first_bin():
    vector = color_histogram(_constant((0.0, 0.0, 0.0)))
    assert vector[0] == pytest.approx(1.0)
    assert np.count_nonzero(vector) == 1


@pytest.mark.parametrize("bins", [2, 8])
def test_white_goes_to_last_bin(bins):
    vector = color_histogram(_constant((1.0, 1.0, 1.0)), bins=bins)
    assert vector[bins**3 - 1] == pytest.approx(1.0)
    assert np.count_nonzero(vector) == 1


@pytest.mark.parametrize(
    "colour,index",
    [((1.0, 0.0, 0.0), 4), ((0.0, 1.0, 0.0), 2), ((0.0, 0.0, 1.0), 1)],
)
def test_primary_colours_bin_index(colour, index):
    vector = color_histogram(_constant(colour), bins=2)
    assert vector[index] == pytest.approx(1.0)
    assert np.count_nonzero(vector) == 1


def test_half_red_half_blue():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[:, 0] = [1.0, 0.0, 0.0]
    image[:, 1] = [0.0, 0.0, 1.0]
    vector = color_histogram(image, bins=2)
    assert np.nonzero(vector)[0].tolist() == [1, 4]
    assert vector[1] == pytest.approx(0.5)
    assert vector[4] == pytest.approx(0.5)


def test_mixed_proportions():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[1, 1] = [1.0, 1.0, 1.0]
    vector = color_histogram(image, bins=2)
    assert vector[0] == pytest.approx(0.75)
    assert vector[7] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "value,bins,index",
    [(0.5, 2, 7), (0.49, 2, 0), (0.25, 4, 21), (0.75, 4, 63), (1.0, 4, 63), (0.0, 4, 0)],
)
def test_bin_boundaries(value, bins, index):
    vector = color_histogram(_constant((value, value, value)), bins=bins)
    assert vector[index] == pytest.approx(1.0)
    assert np.count_nonzero(vector) == 1


def test_tiling_does_not_change_histogram():
    image = _random(2, (3, 5, 3))
    tiled = np.tile(image, (2, 3, 1))
    np.testing.assert_allclose(color_histogram(tiled), color_histogram(image), atol=1e-6)


def test_constant_images_of_different_sizes_match():
    small = color_histogram(_constant((0.3, 0.6, 0.9), 1, 1))
    large = color_histogram(_constant((0.3, 0.6, 0.9), 5, 7))
    np.testing.assert_allclose(small, large, atol=1e-6)


def test_bins_set_vector_length():
    assert color_histogram(_random(3, (4, 4, 3)), bins=4).shape == (64,)


@pytest.mark.parametrize("bins", [1, 0, -3])
def test_rejects_bad_bins(bins):
    with pytest.raises(ValueError):
        color_histogram(_random(4, (4, 4, 3)), bins=bins)


@pytest.mark.parametrize("image", BAD_IMAGES)
def test_rejects_bad_images(image):
    with pytest.raises(ValueError):
        color_histogram(image)


def test_input_is_not_modified():
    image = _random(5, (6, 6, 3))
    copy = image.copy()
    color_histogram(image)
    np.testing.assert_array_equal(image, copy)


def test_non_contiguous_input():
    image = _random(6, (6, 8, 3))
    view = image[:, ::2]
    np.testing.assert_allclose(
        color_histogram(view), color_histogram(np.ascontiguousarray(view)), atol=1e-7
    )


def test_batch_stacks_rows():
    images = [_random(7, (3, 4, 3)), _random(8, (5, 2, 3)), _random(9, (1, 1, 3))]
    matrix = color_histograms(images)
    assert matrix.shape == (3, 512)
    assert matrix.dtype == np.float32
    for row, image in zip(matrix, images):
        np.testing.assert_allclose(row, color_histogram(image), atol=1e-7)


def test_batch_rejects_empty():
    with pytest.raises(ValueError):
        color_histograms([])


def test_batch_propagates_bad_image():
    with pytest.raises(ValueError):
        color_histograms([_random(10, (3, 3, 3)), np.zeros((3, 3, 3), dtype=np.uint8)])


def test_batch_bins_argument():
    matrix = color_histograms([_random(11, (3, 3, 3)), _random(12, (2, 2, 3))], bins=2)
    assert matrix.shape == (2, 8)
