import numpy as np
import pytest

from weathercam.data.augment import add_fog, add_haze, add_rain, add_snow

ALL = [add_fog, add_haze, add_rain, add_snow]
DETERMINISTIC = [add_fog, add_haze]
RANDOM = [add_rain, add_snow]


def noise_image(seed=0, h=120, w=160):
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def flat_image(value, h=120, w=160):
    return np.full((h, w, 3), value, dtype=np.float32)


def luminance_std(image):
    lum = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(lum.std())


def changed_fraction(before, after):
    return float(np.mean(np.any(before != after, axis=2)))


@pytest.mark.parametrize("fn", ALL)
def test_output_shape_dtype_and_range(fn):
    image = noise_image()
    out = fn(image, 0.7, np.random.default_rng(0))
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


@pytest.mark.parametrize("fn", ALL)
def test_input_not_modified(fn):
    image = noise_image()
    before = image.copy()
    out = fn(image, 0.7, np.random.default_rng(0))
    assert np.array_equal(image, before)
    assert out is not image
    assert not np.shares_memory(out, image)


@pytest.mark.parametrize("fn", ALL)
def test_severity_zero_returns_equal_copy(fn):
    image = noise_image()
    out = fn(image, 0.0, np.random.default_rng(0))
    assert np.array_equal(out, image)
    assert not np.shares_memory(out, image)


@pytest.mark.parametrize("fn", ALL)
def test_same_seed_same_result(fn):
    image = noise_image()
    first = fn(image, 0.6, np.random.default_rng(3))
    second = fn(image, 0.6, np.random.default_rng(3))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("fn", ALL)
def test_severity_out_of_range_raises(fn):
    image = noise_image()
    for severity in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            fn(image, severity, np.random.default_rng(0))


@pytest.mark.parametrize("fn", ALL)
def test_invalid_image_raises(fn):
    bad_images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.float32),
        np.full((10, 10, 3), 1.5, dtype=np.float32),
        np.full((10, 10, 3), -0.1, dtype=np.float32),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
    ]
    for bad in bad_images:
        with pytest.raises(ValueError):
            fn(bad, 0.5, np.random.default_rng(0))


@pytest.mark.parametrize("fn", DETERMINISTIC)
def test_contrast_decreases_with_severity(fn):
    image = noise_image()
    stds = [luminance_std(fn(image, s, np.random.default_rng(0))) for s in (0.0, 0.3, 0.6, 1.0)]
    assert all(a > b for a, b in zip(stds, stds[1:]))


@pytest.mark.parametrize("fn", DETERMINISTIC)
def test_brightness_of_dark_image_rises_with_severity(fn):
    image = flat_image(0.1)
    means = [float(fn(image, s, np.random.default_rng(0)).mean()) for s in (0.0, 0.5, 1.0)]
    assert means[0] < means[1] < means[2]
    assert means[1] > 0.2


def test_fog_is_denser_at_top():
    out = add_fog(flat_image(0.1), 1.0, np.random.default_rng(0))
    assert out[:10].mean() > out[-10:].mean()


def test_haze_is_uniform_over_rows():
    out = add_haze(flat_image(0.1), 0.8, np.random.default_rng(0))
    assert np.allclose(out, out[0, 0])


@pytest.mark.parametrize("fn", DETERMINISTIC)
def test_deterministic_functions_ignore_rng(fn):
    image = noise_image()
    first = fn(image, 0.6, np.random.default_rng(1))
    second = fn(image, 0.6, np.random.default_rng(2))
    assert np.array_equal(first, second)


def test_fog_noise_depends_on_seed():
    image = flat_image(0.3)
    a = add_fog(image, 0.6, np.random.default_rng(1), noise=True)
    b = add_fog(image, 0.6, np.random.default_rng(2), noise=True)
    c = add_fog(image, 0.6, np.random.default_rng(1), noise=True)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)
    assert a.min() >= 0.0
    assert a.max() <= 1.0


def test_haze_tint_shifts_color():
    out = add_haze(flat_image(0.5), 1.0, np.random.default_rng(0), tint=(0.9, 0.6, 0.3))
    red, green, blue = (float(out[..., c].mean()) for c in range(3))
    assert red > green > blue


def test_haze_rejects_bad_tint():
    image = flat_image(0.5)
    for tint in [(0.9, 0.6), (0.9, 0.6, 0.3, 0.1), (1.2, 0.5, 0.5), (-0.1, 0.5, 0.5)]:
        with pytest.raises(ValueError):
            add_haze(image, 0.5, np.random.default_rng(0), tint=tint)


@pytest.mark.parametrize("fn", RANDOM)
def test_rain_snow_only_brighten(fn):
    image = flat_image(0.3)
    out = fn(image, 0.8, np.random.default_rng(0))
    assert np.all(out >= image)


@pytest.mark.parametrize("fn", RANDOM)
def test_changed_fraction_is_bounded(fn):
    image = flat_image(0.3)
    half = changed_fraction(image, fn(image, 0.5, np.random.default_rng(0)))
    full = changed_fraction(image, fn(image, 1.0, np.random.default_rng(0)))
    assert 0.0 < half
    assert full <= 0.25


@pytest.mark.parametrize("fn", RANDOM)
def test_changed_fraction_grows_with_severity(fn):
    image = flat_image(0.3)
    low = changed_fraction(image, fn(image, 0.2, np.random.default_rng(0)))
    high = changed_fraction(image, fn(image, 1.0, np.random.default_rng(0)))
    assert 0.0 < low < high


@pytest.mark.parametrize("fn", RANDOM)
def test_different_seed_changes_rain_snow(fn):
    image = flat_image(0.3)
    a = fn(image, 0.8, np.random.default_rng(1))
    b = fn(image, 0.8, np.random.default_rng(2))
    assert not np.array_equal(a, b)
