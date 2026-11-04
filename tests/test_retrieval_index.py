import dataclasses
import tracemalloc

import numpy as np
import pytest

from weathercam.retrieval.index import BruteForceIndex, SearchResult, VectorIndex


def _small_index():
    index = BruteForceIndex(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    return index


def _four_index():
    index = BruteForceIndex(3)
    index.add(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
    return index


FOUR = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])


def test_protocol_is_satisfied():
    assert isinstance(BruteForceIndex(4), VectorIndex)


def test_len_and_dim():
    index = BruteForceIndex(3)
    assert len(index) == 0
    assert index.dim == 3
    index.add(np.ones((5, 3)))
    assert len(index) == 5


def test_add_in_two_calls_keeps_global_ids():
    index = BruteForceIndex(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
    index.add(np.array([[1.0, 1.0]]))
    assert len(index) == 3
    result = index.search(np.array([[1.0, 1.0]]), 1)
    assert result.indices[0, 0] == 2


def test_add_zero_rows_is_noop():
    index = BruteForceIndex(2)
    index.add(np.zeros((0, 2)))
    assert len(index) == 0


def test_cosine_known_values():
    result = _small_index().search(np.array([[1.0, 0.0]]), 3)
    assert result.indices.tolist() == [[0, 2, 1]]
    np.testing.assert_allclose(result.scores, [[1.0, 2 ** -0.5, 0.0]], atol=1e-6)


def test_query_and_gallery_scale_do_not_matter():
    reference = _small_index().search(np.array([[1.0, 0.0]]), 3)
    index = BruteForceIndex(2)
    index.add(5.0 * np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    scaled = index.search(np.array([[3.0, 0.0]]), 3)
    assert scaled.indices.tolist() == reference.indices.tolist()
    np.testing.assert_allclose(scaled.scores, reference.scores, atol=1e-6)


def test_result_dtypes_and_shapes():
    result = _small_index().search(np.array([[1.0, 0.0], [0.0, 1.0]]), 2)
    assert result.indices.shape == (2, 2)
    assert result.scores.shape == (2, 2)
    assert result.indices.dtype == np.int64
    assert result.scores.dtype == np.float32


def test_result_is_frozen():
    result = _small_index().search(np.array([[1.0, 0.0]]), 1)
    assert isinstance(result, SearchResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.indices = None


def test_ties_keep_lower_gallery_index_first():
    index = BruteForceIndex(2)
    index.add(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    result = index.search(np.array([[1.0, 0.0]]), 4)
    assert result.indices.tolist() == [[0, 1, 2, 3]]
    other = BruteForceIndex(2)
    other.add(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]))
    result = other.search(np.array([[1.0, 0.0]]), 3)
    assert result.indices.tolist() == [[1, 2, 0]]


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
def test_chunk_size_does_not_change_results(chunk_size):
    rng = np.random.default_rng(0)
    gallery = rng.normal(size=(40, 5))
    queries = rng.normal(size=(11, 5))
    reference = BruteForceIndex(5, chunk_size=1000)
    reference.add(gallery)
    expected = reference.search(queries, 6)
    index = BruteForceIndex(5, chunk_size=chunk_size)
    index.add(gallery)
    result = index.search(queries, 6)
    assert result.indices.tolist() == expected.indices.tolist()
    np.testing.assert_allclose(result.scores, expected.scores, atol=1e-6)


def test_search_does_not_hold_full_matrix():
    rng = np.random.default_rng(1)
    gallery = rng.normal(size=(2000, 16))
    queries = rng.normal(size=(2000, 16))
    index = BruteForceIndex(16, chunk_size=16)
    index.add(gallery)
    tracemalloc.start()
    try:
        index.search(queries, 5)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000


def test_exclude_removes_self():
    index = _four_index()
    without = index.search(FOUR, 3)
    assert without.indices[:, 0].tolist() == [0, 1, 2, 3]
    result = index.search(FOUR, 3, exclude=np.arange(4))
    for row in range(4):
        assert row not in result.indices[row].tolist()


def test_exclude_applies_per_query():
    result = _four_index().search(FOUR[:2], 3, exclude=np.array([0, -1]))
    assert 0 not in result.indices[0].tolist()
    assert result.indices[1, 0] == 1


def test_exclude_bad_shape_raises():
    with pytest.raises(ValueError):
        _four_index().search(FOUR, 3, exclude=np.array([0, 1]))


@pytest.mark.parametrize("bad", [4, -2])
def test_exclude_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        _four_index().search(FOUR, 3, exclude=np.array([bad, 0, 0, 0]))


def test_exclude_rejects_float_dtype():
    with pytest.raises(ValueError):
        _four_index().search(FOUR, 3, exclude=np.array([0.0, 1.0, 2.0, 3.0]))


def test_k_larger_than_size_raises():
    with pytest.raises(ValueError):
        _four_index().search(FOUR, 5)


def test_k_with_exclude_needs_one_spare():
    with pytest.raises(ValueError):
        _four_index().search(FOUR, 4, exclude=np.arange(4))


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(k):
    with pytest.raises(ValueError):
        _four_index().search(FOUR, k)


def test_search_empty_index_raises():
    with pytest.raises(ValueError):
        BruteForceIndex(2).search(np.array([[1.0, 0.0]]), 1)


@pytest.mark.parametrize("dim,chunk_size", [(0, 10), (4, 0), (-1, 10)])
def test_constructor_rejects_bad_arguments(dim, chunk_size):
    with pytest.raises(ValueError):
        BruteForceIndex(dim, chunk_size=chunk_size)


def test_add_rejects_wrong_width():
    with pytest.raises(ValueError):
        BruteForceIndex(2).add(np.ones((2, 3)))


def test_add_rejects_one_dimensional():
    with pytest.raises(ValueError):
        BruteForceIndex(2).add(np.ones(2))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_add_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        BruteForceIndex(2).add(np.array([[bad, 1.0]]))


def test_add_rejects_zero_vector():
    with pytest.raises(ValueError):
        BruteForceIndex(2).add(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_failed_add_leaves_index_unchanged():
    index = BruteForceIndex(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        index.add(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert len(index) == 2
    result = index.search(np.array([[1.0, 0.0]]), 2)
    assert result.indices.tolist() == [[0, 1]]


@pytest.mark.parametrize(
    "queries",
    [np.ones((1, 3)), np.ones(2), np.array([[np.nan, 1.0]]), np.zeros((1, 2))],
)
def test_search_rejects_bad_queries(queries):
    with pytest.raises(ValueError):
        _small_index().search(queries, 1)


def test_index_does_not_alias_input():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
    index = BruteForceIndex(2)
    index.add(gallery)
    gallery[:] = 0.0
    result = index.search(np.array([[1.0, 0.0]]), 2)
    assert result.indices.tolist() == [[0, 1]]


def test_inputs_are_not_modified():
    gallery = np.array([[3.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    queries = np.array([[5.0, 0.0]], dtype=np.float32)
    gallery_copy = gallery.copy()
    queries_copy = queries.copy()
    index = BruteForceIndex(2)
    index.add(gallery)
    index.search(queries, 2)
    np.testing.assert_array_equal(gallery, gallery_copy)
    np.testing.assert_array_equal(queries, queries_copy)
