import json

import numpy as np
import pytest

from weathercam.plates import ocr_augment
from weathercam.plates.ocr_augment import augment_crop, settings

SHAPE = (96, 192, 3)


def _only(monkeypatch, name):
    for key in ocr_augment.PROBABILITY:
        monkeypatch.setitem(ocr_augment.PROBABILITY, key, 1.0 if key == name else 0.0)


def _constant(value=0.5):
    return np.full(SHAPE, value, dtype=np.float32)


def test_output_is_float32_same_shape_and_in_range():
    base = np.random.default_rng(0).random(SHAPE).astype(np.float32)
    for seed in range(60):
        out = augment_crop(base, np.random.default_rng(seed))
        assert out.dtype == np.float32
        assert out.shape == SHAPE
        assert np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_same_seed_same_output_and_different_seed_differs():
    base = np.random.default_rng(1).random(SHAPE).astype(np.float32)
    a = augment_crop(base, np.random.default_rng(5))
    b = augment_crop(base, np.random.default_rng(5))
    c = augment_crop(base, np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_input_is_not_modified():
    base = np.random.default_rng(2).random(SHAPE).astype(np.float32)
    before = base.copy()
    for seed in range(20):
        augment_crop(base, np.random.default_rng(seed))
    assert np.array_equal(base, before)


def test_rejects_uint8_and_wrong_shape():
    with pytest.raises(ValueError):
        augment_crop(np.zeros(SHAPE, dtype=np.uint8), np.random.default_rng(0))
    with pytest.raises(ValueError):
        augment_crop(np.zeros((96, 192), dtype=np.float32), np.random.default_rng(0))


def test_identity_when_every_operation_is_off(monkeypatch):
    for key in ocr_augment.PROBABILITY:
        monkeypatch.setitem(ocr_augment.PROBABILITY, key, 0.0)
    base = np.random.default_rng(3).random(SHAPE).astype(np.float32)
    assert np.array_equal(augment_crop(base, np.random.default_rng(0)), base)


def test_brightness_scales_a_constant_image_within_the_range(monkeypatch):
    _only(monkeypatch, "brightness")
    values = []
    for seed in range(30):
        out = augment_crop(_constant(), np.random.default_rng(seed))
        assert out.max() - out.min() < 1e-6
        values.append(float(out[0, 0, 0]))
    assert all(0.5 * 0.6 - 1e-4 <= v <= 0.5 * 1.4 + 1e-4 for v in values)
    assert max(values) - min(values) > 0.05


def test_contrast_scales_the_spread_within_the_range(monkeypatch):
    _only(monkeypatch, "contrast")
    base = _constant(0.3)
    base[:, 96:] = 0.7
    ratios = []
    for seed in range(30):
        out = augment_crop(base, np.random.default_rng(seed))
        ratios.append(float(out.std() / base.std()))
    assert all(0.6 - 1e-3 <= r <= 1.4 + 1e-3 for r in ratios)
    assert max(ratios) - min(ratios) > 0.1


def test_gamma_stays_between_the_bounds_on_a_constant_image(monkeypatch):
    _only(monkeypatch, "gamma")
    values = [float(augment_crop(_constant(), np.random.default_rng(s))[0, 0, 0]) for s in range(30)]
    assert all(0.5**1.4 - 1e-4 <= v <= 0.5**0.7 + 1e-4 for v in values)
    assert max(values) - min(values) > 0.05


def test_saturation_leaves_a_gray_image_unchanged(monkeypatch):
    _only(monkeypatch, "saturation")
    gray = np.repeat(np.random.default_rng(4).random((20, 30, 1)).astype(np.float32), 3, axis=2)
    out = augment_crop(gray, np.random.default_rng(0))
    assert np.allclose(out, gray, atol=1e-5)


def test_channel_gain_stays_within_the_bounds(monkeypatch):
    _only(monkeypatch, "gain")
    gains = []
    for seed in range(30):
        out = augment_crop(_constant(), np.random.default_rng(seed))
        gains.append(out[0, 0] / 0.5)
    gains = np.array(gains)
    assert gains.min() >= 0.92 - 1e-4 and gains.max() <= 1.08 + 1e-4
    assert gains.std() > 0.01


def test_grayscale_makes_the_channels_equal(monkeypatch):
    _only(monkeypatch, "gray")
    base = np.random.default_rng(5).random((20, 30, 3)).astype(np.float32)
    out = augment_crop(base, np.random.default_rng(0))
    assert np.allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
    assert np.allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)
    assert not np.allclose(out, base)


def test_blur_keeps_a_constant_image_and_the_mass_of_an_impulse(monkeypatch):
    _only(monkeypatch, "blur")
    for seed in range(10):
        const = augment_crop(_constant(), np.random.default_rng(seed))
        assert np.allclose(const, 0.5, atol=1e-5)
        impulse = np.zeros((41, 41, 3), dtype=np.float32)
        impulse[20, 20] = 1.0
        out = augment_crop(impulse, np.random.default_rng(seed))
        assert out.max() < 0.7
        assert np.allclose(out.sum(axis=(0, 1)), 1.0, atol=1e-4)


def test_noise_has_zero_mean_and_bounded_spread(monkeypatch):
    _only(monkeypatch, "noise")
    spreads = []
    for seed in range(20):
        diff = augment_crop(_constant(), np.random.default_rng(seed)) - 0.5
        assert abs(float(diff.mean())) < 1e-3
        spreads.append(float(diff.std()))
    assert max(spreads) <= 0.04 * 1.05
    assert max(spreads) > 0.02


def test_weather_changes_the_image_and_keeps_the_range(monkeypatch):
    _only(monkeypatch, "weather")
    for seed in range(20):
        out = augment_crop(_constant(), np.random.default_rng(seed))
        assert np.any(out != 0.5)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_settings_are_json_serialisable_and_consistent():
    text = json.dumps(settings())
    loaded = json.loads(text)
    assert set(loaded["probability"]) == set(ocr_augment.PROBABILITY)
    assert all(0.0 <= p <= 1.0 for p in loaded["probability"].values())
    assert loaded["weather_effects"] == ["add_fog", "add_haze", "add_rain", "add_snow"]
