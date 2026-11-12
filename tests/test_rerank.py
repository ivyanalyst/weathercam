import numpy as np
import pytest

from weathercam.retrieval.rerank import expand_queries


def gallery():
    return np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]])


def test_expands_with_top_neighbour_excluding_self():
    g = gallery()
    out = expand_queries(g, g, 1, 1.0, exclude_self=True)
    expected = np.array(
        [
            np.array([2.0, 1.0]) / np.sqrt(5.0),
            np.array([1.0, 3.0]) / np.sqrt(10.0),
            np.array([1.0, 3.0]) / np.sqrt(10.0),
            np.array([-1.0, 1.0]) / np.sqrt(2.0),
        ]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_averages_several_neighbours_with_weight():
    g = gallery()
    out = expand_queries(g, g[:1], 2, 2.0, exclude_self=False)
    assert out.shape == (1, 2)
    out = expand_queries(g, g, 2, 2.0, exclude_self=True)
    np.testing.assert_allclose(out[0], np.array([1.6, 1.8]) / np.sqrt(5.8), atol=1e-12)


def test_zero_weight_returns_normalised_queries():
    out = expand_queries(gallery(), np.array([[3.0, 4.0]]), 1, 0.0)
    np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-12)


def test_zero_neighbours_returns_normalised_queries():
    out = expand_queries(gallery(), np.array([[3.0, 4.0]]), 0, 1.0)
    np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-12)


def test_includes_self_when_not_excluded():
    g = gallery()
    out = expand_queries(g, g[:1], 1, 1.0, exclude_self=False)
    np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-12)


def test_ties_take_lower_index():
    g = np.array([[0.6, 0.8], [0.6, -0.8], [-1.0, 0.0]])
    out = expand_queries(g, np.array([[1.0, 0.0]]), 1, 1.0)
    np.testing.assert_allclose(out, [np.array([2.0, 1.0]) / np.sqrt(5.0)], atol=1e-12)


def test_chunk_size_does_not_change_result():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(40, 6))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    whole = expand_queries(g, g, 3, 1.5, exclude_self=True, chunk=512)
    for chunk in (1, 7):
        part = expand_queries(g, g, 3, 1.5, exclude_self=True, chunk=chunk)
        np.testing.assert_allclose(part, whole, atol=1e-12)


def test_does_not_modify_inputs():
    g = gallery()
    q = g.copy()
    expand_queries(g, q, 2, 1.0, exclude_self=True)
    assert np.array_equal(g, gallery())
    assert np.array_equal(q, gallery())


def test_output_shape():
    out = expand_queries(gallery(), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 2, 1.0)
    assert out.shape == (3, 2)


def test_rejects_negative_neighbours():
    with pytest.raises(ValueError):
        expand_queries(gallery(), gallery(), -1, 1.0)


def test_rejects_negative_weight():
    with pytest.raises(ValueError):
        expand_queries(gallery(), gallery(), 1, -0.5)


def test_rejects_too_many_neighbours():
    with pytest.raises(ValueError):
        expand_queries(gallery(), gallery(), 5, 1.0)


def test_rejects_too_many_neighbours_when_excluding_self():
    with pytest.raises(ValueError):
        expand_queries(gallery(), gallery(), 4, 1.0, exclude_self=True)


def test_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        expand_queries(gallery(), np.ones((1, 3)), 1, 1.0)


def test_rejects_exclude_self_with_different_lengths():
    with pytest.raises(ValueError):
        expand_queries(gallery(), gallery()[:3], 1, 1.0, exclude_self=True)


def test_rejects_non_positive_chunk():
    with pytest.raises(ValueError):
        expand_queries(gallery(), gallery(), 1, 1.0, chunk=0)


def test_rejects_zero_vector_result():
    with pytest.raises(ValueError):
        expand_queries(np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]), 1, 1.0)
