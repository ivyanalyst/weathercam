import numpy as np
import pytest

from weathercam.service.heads import (
    COLOR_NAMES,
    TYPE_NAMES,
    LinearHead,
    MetricProjector,
    Prediction,
    classify,
    load_linear_head,
    load_metric_head,
    standardise,
)

NAMES = ("a", "b", "c")


def make_head(**changes):
    fields = {
        "weights": np.eye(3),
        "bias": np.zeros(3),
        "mean": np.zeros(3),
        "std": np.ones(3),
        "class_ids": np.array([1, 2, 3]),
        "names": NAMES,
    }
    fields.update(changes)
    return LinearHead(**fields)


def save_head(path, drop=(), **changes):
    arrays = {
        "weights": np.eye(3),
        "bias": np.zeros(3),
        "mean": np.zeros(3, dtype=np.float32),
        "std": np.ones(3, dtype=np.float32),
        "class_ids": np.array([1, 2, 3], dtype=np.int64),
    }
    arrays.update(changes)
    for key in drop:
        del arrays[key]
    np.savez(path, **arrays)
    return path


def test_names_have_expected_lengths_and_ends():
    assert len(COLOR_NAMES) == 10
    assert len(TYPE_NAMES) == 9
    assert (COLOR_NAMES[0], COLOR_NAMES[-1]) == ("yellow", "black")
    assert (TYPE_NAMES[0], TYPE_NAMES[-1]) == ("sedan", "estate")


def test_standardise_applies_mean_and_std():
    result = standardise(np.array([[2.0, 4.0]]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert result.tolist() == [[1.0, 1.0]]


def test_standardise_rejects_bad_input():
    mean, std = np.zeros(2), np.ones(2)
    with pytest.raises(ValueError):
        standardise(np.zeros(2), mean, std)
    with pytest.raises(ValueError):
        standardise(np.zeros((1, 3)), mean, std)
    with pytest.raises(ValueError):
        standardise(np.array([[1.0, np.nan]]), mean, std)


def test_linear_head_rejects_inconsistent_shapes():
    for changes in (
        {"bias": np.zeros(2)},
        {"mean": np.zeros(2)},
        {"std": np.ones(2)},
        {"class_ids": np.array([1, 2])},
        {"names": ("a", "b")},
        {"weights": np.zeros(3)},
    ):
        with pytest.raises(ValueError):
            make_head(**changes)


def test_linear_head_rejects_non_positive_std():
    with pytest.raises(ValueError):
        make_head(std=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        make_head(std=np.array([1.0, -1.0, 1.0]))


def test_classify_maps_index_through_class_ids():
    result = classify(make_head(), np.array([[0.0, 0.0, 5.0]]))
    assert len(result) == 1
    assert result[0].class_id == 3
    assert result[0].label == "c"


def test_classify_probability_value():
    result = classify(make_head(), np.array([[1.0, 0.0, 0.0]]))
    assert result[0].probability == pytest.approx(0.5761169, abs=1e-6)


def test_classify_standardises_before_scoring():
    head = make_head(mean=np.array([0.0, 2.0, 0.0]))
    result = classify(head, np.array([[1.0, 2.5, 0.0]]))
    assert result[0].class_id == 1


def test_top_is_sorted_and_has_top_k_entries():
    head = make_head(class_ids=np.array([10, 20, 30]))
    result = classify(head, np.array([[1.0, 3.0, 2.0]]), top_k=2)[0]
    assert [entry[0] for entry in result.top] == [20, 30]
    assert [entry[1] for entry in result.top] == ["b", "c"]
    assert result.top[0][2] > result.top[1][2]
    assert result.class_id == 20
    assert result.probability == pytest.approx(result.top[0][2])


def test_top_ties_go_to_lower_index():
    result = classify(make_head(), np.array([[1.0, 1.0, 0.0]]), top_k=2)[0]
    assert [entry[0] for entry in result.top] == [1, 2]
    assert result.class_id == 1


def test_classify_returns_one_prediction_per_row():
    result = classify(make_head(), np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]))
    assert [item.class_id for item in result] == [1, 3]
    assert all(isinstance(item, Prediction) for item in result)


def test_classify_rejects_bad_top_k():
    for top_k in (0, 4):
        with pytest.raises(ValueError):
            classify(make_head(), np.zeros((1, 3)), top_k=top_k)


def test_load_linear_head_reads_names_from_key(tmp_path):
    path = save_head(tmp_path / "head.npz", weathers=np.array(["Fog", "Rain", "Sand"]))
    head = load_linear_head(path, names_key="weathers")
    assert head.names == ("Fog", "Rain", "Sand")
    assert head.class_ids.tolist() == [1, 2, 3]


def test_load_linear_head_uses_given_names_and_keeps_dtypes(tmp_path):
    path = save_head(tmp_path / "head.npz")
    head = load_linear_head(path, names=NAMES)
    assert head.names == NAMES
    assert head.mean.dtype == np.float32
    assert head.std.dtype == np.float32
    assert head.weights.dtype == np.float64


def test_load_linear_head_requires_names(tmp_path):
    path = save_head(tmp_path / "head.npz")
    with pytest.raises(ValueError):
        load_linear_head(path)


def test_load_linear_head_missing_key_raises(tmp_path):
    path = save_head(tmp_path / "head.npz", drop=("std",))
    with pytest.raises(ValueError, match="std"):
        load_linear_head(path, names=NAMES)


def test_load_linear_head_refuses_pickled_arrays(tmp_path):
    path = save_head(tmp_path / "head.npz", names=np.array(["a", "b", "c"], dtype=object))
    with pytest.raises(ValueError):
        load_linear_head(path, names_key="names")


def test_metric_projection_gives_unit_rows():
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    projector = MetricProjector(weights, np.zeros(3), np.ones(3))
    result = projector.project(np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]]))
    assert result.dtype == np.float64
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], atol=1e-12)


def test_metric_projection_standardises_first():
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    projector = MetricProjector(weights, np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))
    result = projector.project(np.array([[3.0, 4.0, 0.0]]))
    np.testing.assert_allclose(result, [[0.2425356, 0.9701425]], atol=1e-6)


def test_metric_projector_rejects_bad_input_and_std():
    weights = np.eye(3)[:, :2]
    with pytest.raises(ValueError):
        MetricProjector(weights, np.zeros(2), np.ones(3))
    with pytest.raises(ValueError):
        MetricProjector(weights, np.zeros(3), np.array([1.0, 0.0, 1.0]))
    projector = MetricProjector(weights, np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        projector.project(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        projector.project(np.array([[1.0, np.nan, 0.0]]))


def test_load_metric_head_round_trip_and_missing_key(tmp_path):
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    path = tmp_path / "metric.npz"
    np.savez(path, weights=weights, mean=np.zeros(3), std=np.ones(3), scale=np.float64(8.0), dim=np.int64(2))
    projector = load_metric_head(path)
    np.testing.assert_allclose(projector.project(np.array([[3.0, 4.0, 0.0]])), [[0.6, 0.8]], atol=1e-12)
    broken = tmp_path / "broken.npz"
    np.savez(broken, weights=weights, mean=np.zeros(3))
    with pytest.raises(ValueError, match="std"):
        load_metric_head(broken)
