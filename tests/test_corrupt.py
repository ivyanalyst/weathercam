import numpy as np
import pytest

from weathercam.data.corrupt import darken, gaussian_blur, gaussian_noise, jpeg

ALL = [gaussian_blur, gaussian_noise, darken, jpeg]
BAD_IMAGES = [
    pytest.param([[[0.5, 0.5, 0.5]]], id="list"),
    pytest.param(np.zeros((8, 8, 3), dtype=np.uint8), id="uint8"),
    pytest.param(np.full((8, 8, 3), 1.5, dtype=np.float32), id="above_one"),
]


def call(fn, image, severity, seed=0):
    if fn is gaussian_noise:
        return fn(image, severity, np.random.default_rng(seed))
    return fn(image, severity)


def noise_image(seed=0, h=120, w=160):
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def flat_image(value, h=120, w=160):
    return np.full((h, w, 3), value, dtype=np.float32)


def luminance_std(image):
    lum = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(lum.std())


def impulse_image(size=101, channels=(0, 1, 2)):
    image = np.zeros((size, size, 3), dtype=np.float32)
    centre = size // 2
    for c in channels:
        image[centre, centre, c] = 1.0
    return image


@pytest.mark.parametrize("fn", ALL)
def test_output_shape_dtype_and_range(fn):
    image = noise_image()
    out = call(fn, image, 0.5)
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("fn", ALL)
def test_input_is_not_modified(fn):
    image = noise_image()
    before = image.copy()
    call(fn, image, 0.7)
    assert np.array_equal(image, before)


@pytest.mark.parametrize("fn", ALL)
def test_severity_zero_returns_equal_copy(fn):
    image = noise_image()
    out = call(fn, image, 0.0)
    assert out is not image
    assert np.array_equal(out, image)


@pytest.mark.parametrize("fn", ALL)
@pytest.mark.parametrize("severity", [-0.1, 1.5, float("nan")])
def test_rejects_severity_outside_unit_interval(fn, severity):
    with pytest.raises(ValueError):
        call(fn, noise_image(), severity)


@pytest.mark.parametrize("fn", ALL)
@pytest.mark.parametrize("image", BAD_IMAGES)
def test_rejects_invalid_images(fn, image):
    with pytest.raises(ValueError):
        call(fn, image, 0.5)


@pytest.mark.parametrize("fn", ALL)
def test_float64_input_gives_float32_output(fn):
    image = noise_image().astype(np.float64)
    out = call(fn, image, 0.5)
    assert out.dtype == np.float32


def test_blur_preserves_constant_image():
    out = gaussian_blur(flat_image(0.4), 1.0)
    assert np.allclose(out, 0.4, atol=1e-5)


def test_blur_impulse_neighbour_ratio_matches_sigma():
    out = gaussian_blur(impulse_image(), 1.0)
    assert out[50, 51, 0] / out[50, 50, 0] == pytest.approx(0.8847, abs=1e-4)
    assert out[51, 50, 0] / out[50, 50, 0] == pytest.approx(0.8847, abs=1e-4)


def test_blur_conserves_total_intensity_of_interior_impulse():
    image = impulse_image()
    out = gaussian_blur(image, 1.0)
    assert float(out.sum()) == pytest.approx(float(image.sum()), abs=1e-4)


def test_blur_impulse_response_is_symmetric():
    out = gaussian_blur(impulse_image(), 1.0)
    for d in range(1, 6):
        assert np.allclose(out[50, 50 + d], out[50, 50 - d], atol=1e-7)
        assert np.allclose(out[50 + d, 50], out[50 - d, 50], atol=1e-7)
        assert np.allclose(out[50, 50 + d], out[50 + d, 50], atol=1e-7)


def test_blur_reduces_contrast_more_at_higher_severity():
    image = noise_image()
    stds = [luminance_std(gaussian_blur(image, s)) for s in (0.2, 0.5, 1.0)]
    assert luminance_std(image) > stds[0] > stds[1] > stds[2]


def test_blur_keeps_channels_independent():
    out = gaussian_blur(impulse_image(channels=(0,)), 1.0)
    assert out[:, :, 0].max() > 0.0
    assert np.all(out[:, :, 1:] == 0.0)


def test_noise_std_and_mean_match_severity():
    out = gaussian_noise(flat_image(0.5), 0.5, np.random.default_rng(0))
    assert float(out.std()) == pytest.approx(0.1, abs=0.003)
    assert float(out.mean()) == pytest.approx(0.5, abs=0.003)


def test_noise_same_seed_same_output():
    image = noise_image()
    a = gaussian_noise(image, 0.5, np.random.default_rng(7))
    b = gaussian_noise(image, 0.5, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_noise_different_seed_changes_output():
    image = noise_image()
    a = gaussian_noise(image, 0.5, np.random.default_rng(1))
    b = gaussian_noise(image, 0.5, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_noise_clips_to_unit_range():
    out = gaussian_noise(noise_image(), 1.0, np.random.default_rng(0))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert (out == 0.0).any() and (out == 1.0).any()


def test_darken_known_values():
    assert np.allclose(darken(flat_image(0.5), 0.5), 0.3, atol=1e-6)
    assert np.allclose(darken(flat_image(0.5), 1.0), 0.1, atol=1e-6)


def test_darken_scales_every_pixel_by_the_same_factor():
    image = noise_image()
    image[0, 0] = 0.0
    out = darken(image, 0.25)
    assert np.allclose(out, image * 0.8, atol=1e-6)
    assert np.all(out[0, 0] == 0.0)


@pytest.mark.parametrize("severity", [0.1, 0.5, 1.0])
def test_jpeg_keeps_flat_gray_close_to_gray(severity):
    out = jpeg(flat_image(0.5), severity)
    assert np.allclose(out, 0.5, atol=0.01)


def test_jpeg_error_grows_with_severity():
    image = noise_image()
    errors = [float(np.abs(jpeg(image, s) - image).mean()) for s in (0.1, 0.5, 1.0)]
    assert errors[0] < errors[1] < errors[2]


def test_jpeg_is_deterministic():
    image = noise_image()
    assert np.array_equal(jpeg(image, 0.6), jpeg(image, 0.6))
