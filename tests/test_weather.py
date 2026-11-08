import numpy as np
import pytest

from weathercam.attributes.weather import (
    PARTS,
    load_weather,
    majority_accuracy,
    part_arrays,
    wilson_interval,
)


def test_wilson_half():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)


def test_wilson_zero_successes():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_all_successes():
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high == 1.0


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def _write(path, **over):
    arrays = dict(
        embeddings=np.arange(12, dtype=np.float32).reshape(6, 2),
        weather_ids=np.array([0, 1, 2, 0, 1, 2]),
        split=np.array(["train", "train", "val", "val", "test", "test"]),
        weathers=np.array(["Fog", "Rain", "Sand"]),
        image_names=np.array(["a", "b", "c", "d", "e", "f"]),
    )
    arrays.update(over)
    np.savez(path, **arrays)


def test_load_and_part_arrays(tmp_path):
    f = tmp_path / "w.npz"
    _write(f)
    data = load_weather(f)
    assert data.weathers == ("Fog", "Rain", "Sand")
    x, y = part_arrays(data, "val")
    np.testing.assert_array_equal(x, [[4.0, 5.0], [6.0, 7.0]])
    np.testing.assert_array_equal(y, [2, 0])
    assert PARTS == ("train", "val", "test")


def test_part_arrays_rejects_unknown_part(tmp_path):
    f = tmp_path / "w.npz"
    _write(f)
    with pytest.raises(ValueError):
        part_arrays(load_weather(f), "dev")


def test_load_rejects_length_mismatch(tmp_path):
    f = tmp_path / "w.npz"
    _write(f, weather_ids=np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        load_weather(f)


def test_load_rejects_label_outside_classes(tmp_path):
    f = tmp_path / "w.npz"
    _write(f, weather_ids=np.array([0, 1, 2, 0, 1, 3]))
    with pytest.raises(ValueError):
        load_weather(f)


def test_load_rejects_unknown_split_name(tmp_path):
    f = tmp_path / "w.npz"
    _write(f, split=np.array(["train", "train", "val", "val", "test", "dev"]))
    with pytest.raises(ValueError):
        load_weather(f)


def test_majority_accuracy_uses_train_majority():
    assert majority_accuracy([2, 2, 1], [2, 0, 2, 1]) == 0.5


def test_majority_accuracy_rejects_empty():
    with pytest.raises(ValueError):
        majority_accuracy([], [1])
