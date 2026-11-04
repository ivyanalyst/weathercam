import numpy as np
import pytest

from weathercam.retrieval.evaluate import (
    cross_vehicle_scores,
    group_scores,
    overall_score,
    query_scores,
)
from weathercam.retrieval.index import BruteForceIndex


def _unit(degrees):
    radians = np.radians(np.asarray(degrees, dtype=np.float64))
    return np.stack([np.cos(radians), np.sin(radians)], axis=1)


def _index(vectors):
    index = BruteForceIndex(vectors.shape[1])
    index.add(vectors)
    return index


CASE_C = _unit([0, 5, 20, 45, 90])
ATTRIBUTES_C = ["A", "A", "A", "B", "A"]
VEHICLES_C = ["v1", "v1", "v2", "v3", "v4"]


def _scores(vectors, attributes, vehicles, block_size=512):
    return cross_vehicle_scores(
        _index(vectors), vectors, attributes, vehicles, block_size=block_size
    )


def test_case_c_query_values():
    scores = _scores(CASE_C, ATTRIBUTES_C, VEHICLES_C)
    assert scores.relevant.tolist() == [2, 2, 3, 0, 3]
    assert scores.precision_at_1.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    np.testing.assert_allclose(
        scores.average_precision, [0.5, 0.5, 2.0 / 3.0, 0.0, 7.0 / 18.0]
    )


def test_case_c_overall():
    score = overall_score(_scores(CASE_C, ATTRIBUTES_C, VEHICLES_C))
    assert score.precision_at_1 == pytest.approx(0.75)
    assert score.map_at_r == pytest.approx(37.0 / 72.0)
    assert score.scored == 4
    assert score.skipped == 1


@pytest.mark.parametrize("block_size", [1, 2, 100])
def test_block_size_does_not_change_scores(block_size):
    reference = _scores(CASE_C, ATTRIBUTES_C, VEHICLES_C, block_size=512)
    scores = _scores(CASE_C, ATTRIBUTES_C, VEHICLES_C, block_size=block_size)
    np.testing.assert_array_equal(scores.precision_at_1, reference.precision_at_1)
    np.testing.assert_allclose(scores.average_precision, reference.average_precision)
    np.testing.assert_array_equal(scores.relevant, reference.relevant)


def test_own_vehicle_images_do_not_count():
    vectors = _unit([0, 10, 90])
    scores = _scores(vectors, ["A", "A", "A"], ["v1", "v1", "v2"])
    assert scores.relevant.tolist() == [1, 1, 2]
    assert scores.precision_at_1.tolist() == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(scores.average_precision, [1.0, 1.0, 1.0])


def test_single_image_vehicles_match_plain_scores():
    vectors = _unit([0, 10, 30, 90])
    attributes = ["A", "A", "B", "B"]
    cross = _scores(vectors, attributes, ["a", "b", "c", "d"])
    plain = query_scores(_index(vectors), vectors, attributes)
    np.testing.assert_array_equal(cross.precision_at_1, plain.precision_at_1)
    np.testing.assert_allclose(cross.average_precision, plain.average_precision)
    np.testing.assert_array_equal(cross.relevant, plain.relevant)


def test_all_skipped_raises():
    vectors = _unit([0, 10, 90])
    with pytest.raises(ValueError):
        _scores(vectors, ["A", "A", "B"], ["v1", "v1", "v2"])


def test_group_scores_on_cross_vehicle():
    scores = _scores(CASE_C, ATTRIBUTES_C, VEHICLES_C)
    result = group_scores(scores, ATTRIBUTES_C, VEHICLES_C)
    a = result["A"]
    assert a.precision_at_1 == pytest.approx(0.75)
    assert a.map_at_r == pytest.approx(37.0 / 72.0)
    assert (a.queries, a.scored, a.skipped, a.vehicles) == (4, 4, 0, 4)
    b = result["B"]
    assert b.precision_at_1 is None
    assert b.map_at_r is None
    assert (b.queries, b.scored, b.skipped, b.vehicles) == (1, 0, 1, 1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        _scores(CASE_C, ATTRIBUTES_C[:4], VEHICLES_C)
    with pytest.raises(ValueError):
        _scores(CASE_C, ATTRIBUTES_C, VEHICLES_C[:4])


def test_queries_must_be_two_dimensional():
    with pytest.raises(ValueError):
        cross_vehicle_scores(_index(CASE_C), np.ones(5), ATTRIBUTES_C, VEHICLES_C)


def test_index_size_must_match_queries():
    with pytest.raises(ValueError):
        cross_vehicle_scores(_index(CASE_C), CASE_C[:4], ATTRIBUTES_C[:4], VEHICLES_C[:4])


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_size_must_be_positive(block_size):
    with pytest.raises(ValueError):
        _scores(CASE_C, ATTRIBUTES_C, VEHICLES_C, block_size=block_size)


def test_empty_queries_raises():
    with pytest.raises(ValueError):
        cross_vehicle_scores(BruteForceIndex(2), np.zeros((0, 2)), [], [])


def test_integer_labels():
    scores = _scores(CASE_C, np.array([1, 1, 1, 2, 1]), np.array([1, 1, 2, 3, 4]))
    assert scores.precision_at_1.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_inputs_are_not_modified():
    vectors = CASE_C.copy()
    attributes = list(ATTRIBUTES_C)
    vehicles = list(VEHICLES_C)
    _scores(vectors, attributes, vehicles)
    np.testing.assert_array_equal(vectors, CASE_C)
    assert attributes == ATTRIBUTES_C
    assert vehicles == VEHICLES_C
