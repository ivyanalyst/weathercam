import numpy as np
import pytest

from weathercam.retrieval.ood import knn_distances


def bank():
    return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])


def probes():
    return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_k1_distance_to_nearest():
    out = knn_distances(bank(), probes(), 1)
    expected = np.array([0.0, 0.0, np.sqrt(2.0 - np.sqrt(2.0))])
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_k2_distance():
    out = knn_distances(bank(), probes(), 2)
    expected = np.array([np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0 - np.sqrt(2.0))])
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_k3_distance():
    out = knn_distances(bank(), probes(), 3)
    expected = np.array([2.0, np.sqrt(2.0), np.sqrt(2.0 + np.sqrt(2.0))])
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_query_scale_invariant():
    out = knn_distances(bank(), np.array([[3.0, 0.0], [0.0, 0.5]]), 2)
    expected = knn_distances(bank(), np.array([[1.0, 0.0], [0.0, 1.0]]), 2)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_bank_scale_invariant():
    scaled = bank() * np.array([[5.0], [0.2], [3.0]])
    np.testing.assert_allclose(
        knn_distances(scaled, probes(), 2), knn_distances(bank(), probes(), 2), atol=1e-12
    )


def test_larger_k_never_smaller():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(30, 6))
    q = rng.normal(size=(20, 6))
    stacked = np.stack([knn_distances(b, q, k) for k in range(1, 6)])
    assert np.all(np.diff(stacked, axis=0) >= -1e-12)


def test_chunk_size_does_not_change_result():
    rng = np.random.default_rng(1)
    b = rng.normal(size=(40, 6))
    q = rng.normal(size=(25, 6))
    whole = knn_distances(b, q, 3, chunk=512)
    for chunk in (1, 7):
        np.testing.assert_allclose(knn_distances(b, q, 3, chunk=chunk), whole, atol=1e-12)


def test_does_not_modify_inputs():
    b = bank()
    q = probes()
    knn_distances(b, q, 2)
    assert np.array_equal(b, bank())
    assert np.array_equal(q, probes())


def test_output_shape():
    assert knn_distances(bank(), probes(), 1).shape == (3,)


def test_far_query_scores_higher():
    b = np.array([[1.0, 0.0], [0.8, 0.6]])
    inside = knn_distances(b, np.array([[1.0, 0.0]]), 1)
    outside = knn_distances(b, np.array([[-1.0, 0.0]]), 1)
    assert outside[0] > inside[0]


def test_rejects_k_zero():
    with pytest.raises(ValueError):
        knn_distances(bank(), probes(), 0)


def test_rejects_k_above_bank_size():
    with pytest.raises(ValueError):
        knn_distances(bank(), probes(), 4)


def test_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        knn_distances(bank(), np.ones((2, 3)), 1)


def test_rejects_non_positive_chunk():
    with pytest.raises(ValueError):
        knn_distances(bank(), probes(), 1, chunk=0)


def test_rejects_zero_query():
    with pytest.raises(ValueError):
        knn_distances(bank(), np.array([[0.0, 0.0]]), 1)


def test_rejects_zero_bank_row():
    with pytest.raises(ValueError):
        knn_distances(np.array([[1.0, 0.0], [0.0, 0.0]]), probes(), 1)


def test_rejects_1d_queries():
    with pytest.raises(ValueError):
        knn_distances(bank(), np.ones(2), 1)
