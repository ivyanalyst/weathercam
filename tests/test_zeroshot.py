import numpy as np
import pytest

from weathercam.attributes.zeroshot import (
    aggregate_tracks,
    class_vectors,
    image_probabilities,
    track_predictions,
)


def test_class_vectors_normalises_before_mean():
    text = np.array([[[2.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
    out = class_vectors(text)
    r = np.sqrt(0.5)
    np.testing.assert_allclose(out, [[r, r], [1.0, 0.0]], atol=1e-7)


def test_class_vectors_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        class_vectors(np.ones((3, 4)))


def test_image_probabilities_hand_values():
    images = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = image_probabilities(images, np.eye(2), scale=1.0)
    hi = np.e / (1.0 + np.e)
    np.testing.assert_allclose(out, [[hi, 1 - hi], [1 - hi, hi]], atol=1e-7)


def test_image_probabilities_stable_for_large_scale():
    out = image_probabilities(np.array([[1.0, 0.0]]), np.eye(2), scale=1000.0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-12)


def test_aggregate_tracks_means_in_first_seen_order():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]])
    keys = [("a", "1"), ("b", "1"), ("a", "1")]
    out_keys, means, counts = aggregate_tracks(probs, keys)
    assert out_keys == [("a", "1"), ("b", "1")]
    np.testing.assert_allclose(means, [[0.7, 0.3], [0.1, 0.9]])
    assert counts.tolist() == [2, 1]


def test_aggregate_tracks_length_mismatch():
    with pytest.raises(ValueError):
        aggregate_tracks(np.zeros((2, 2)), [("a", "1")])


def test_track_predictions():
    means = np.array([[0.7, 0.3], [0.1, 0.9], [0.4, 0.6]])
    labels, conf = track_predictions(means)
    assert labels.tolist() == [0, 1, 1]
    np.testing.assert_allclose(conf, [0.7, 0.9, 0.6])
