import dataclasses

import numpy as np
import pytest

from weathercam.retrieval.evaluate import (
    GroupScore,
    QueryScores,
    camera_aware_scores,
    group_scores,
    overall_score,
    query_scores,
)
from weathercam.retrieval.index import BruteForceIndex
from weathercam.retrieval.metrics import RetrievalScore


def _unit(degrees):
    radians = np.radians(np.asarray(degrees, dtype=np.float64))
    return np.stack([np.cos(radians), np.sin(radians)], axis=1)


def _index(vectors):
    index = BruteForceIndex(vectors.shape[1])
    index.add(vectors)
    return index


CASE_A = _unit([0, 10, 30, 90])
CASE_A_LABELS = ["A", "A", "B", "B"]
CASE_B = _unit([0, 10, 30, 90, 100])
CASE_B_LABELS = ["A", "B", "A", "B", "A"]
SKIP = _unit([0, 10, 90])
SKIP_LABELS = ["A", "A", "B"]


def _scores(vectors, labels, block_size=512):
    return query_scores(_index(vectors), vectors, labels, block_size=block_size)


def test_case_a_query_values():
    scores = _scores(CASE_A, CASE_A_LABELS)
    assert isinstance(scores, QueryScores)
    assert scores.precision_at_1.tolist() == [1.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(scores.average_precision, [1.0, 1.0, 0.0, 1.0])
    assert scores.relevant.tolist() == [1, 1, 1, 1]


def test_case_b_query_values():
    scores = _scores(CASE_B, CASE_B_LABELS)
    assert scores.precision_at_1.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(scores.average_precision, [0.25, 0.0, 0.25, 0.0, 0.25])
    assert scores.relevant.tolist() == [2, 1, 2, 1, 2]


def test_query_does_not_retrieve_itself():
    scores = _scores(CASE_A, CASE_A_LABELS)
    assert scores.average_precision[2] == 0.0
    assert scores.precision_at_1[2] == 0.0


@pytest.mark.parametrize("block_size", [1, 2, 3, 100])
def test_block_size_does_not_change_scores(block_size):
    reference = _scores(CASE_B, CASE_B_LABELS, block_size=512)
    scores = _scores(CASE_B, CASE_B_LABELS, block_size=block_size)
    np.testing.assert_array_equal(scores.precision_at_1, reference.precision_at_1)
    np.testing.assert_allclose(scores.average_precision, reference.average_precision)
    np.testing.assert_array_equal(scores.relevant, reference.relevant)


def test_dtypes_and_shapes():
    scores = _scores(CASE_A, CASE_A_LABELS)
    assert scores.precision_at_1.shape == (4,)
    assert scores.average_precision.shape == (4,)
    assert scores.relevant.shape == (4,)
    assert scores.precision_at_1.dtype == np.float64
    assert scores.average_precision.dtype == np.float64
    assert scores.relevant.dtype == np.int64


def test_singleton_label_is_skipped():
    scores = _scores(SKIP, SKIP_LABELS)
    assert scores.relevant.tolist() == [1, 1, 0]
    assert scores.precision_at_1.tolist() == [1.0, 1.0, 0.0]
    assert scores.average_precision.tolist() == [1.0, 1.0, 0.0]


def test_overall_case_a():
    score = overall_score(_scores(CASE_A, CASE_A_LABELS))
    assert isinstance(score, RetrievalScore)
    assert score.precision_at_1 == pytest.approx(0.75)
    assert score.map_at_r == pytest.approx(0.75)
    assert score.scored == 4
    assert score.skipped == 0


def test_overall_case_b():
    score = overall_score(_scores(CASE_B, CASE_B_LABELS))
    assert score.precision_at_1 == pytest.approx(0.0)
    assert score.map_at_r == pytest.approx(0.15)
    assert score.scored == 5
    assert score.skipped == 0


def test_overall_with_skipped():
    score = overall_score(_scores(SKIP, SKIP_LABELS))
    assert score.precision_at_1 == pytest.approx(1.0)
    assert score.map_at_r == pytest.approx(1.0)
    assert score.scored == 2
    assert score.skipped == 1


def test_all_skipped_raises():
    vectors = _unit([0, 10, 90])
    with pytest.raises(ValueError):
        _scores(vectors, ["A", "B", "C"])


def test_group_scores_case_a():
    scores = _scores(CASE_A, CASE_A_LABELS)
    result = group_scores(scores, CASE_A_LABELS, ["v1", "v1", "v2", "v3"])
    assert sorted(result) == ["A", "B"]
    a = result["A"]
    assert isinstance(a, GroupScore)
    assert a.precision_at_1 == pytest.approx(1.0)
    assert a.map_at_r == pytest.approx(1.0)
    assert (a.queries, a.scored, a.skipped, a.vehicles) == (2, 2, 0, 1)
    b = result["B"]
    assert b.precision_at_1 == pytest.approx(0.5)
    assert b.map_at_r == pytest.approx(0.5)
    assert (b.queries, b.scored, b.skipped, b.vehicles) == (2, 2, 0, 2)


def test_group_scores_case_b():
    scores = _scores(CASE_B, CASE_B_LABELS)
    result = group_scores(scores, CASE_B_LABELS, ["v1", "v2", "v1", "v2", "v3"])
    a = result["A"]
    assert a.precision_at_1 == pytest.approx(0.0)
    assert a.map_at_r == pytest.approx(0.25)
    assert (a.queries, a.scored, a.skipped, a.vehicles) == (3, 3, 0, 2)
    b = result["B"]
    assert b.map_at_r == pytest.approx(0.0)
    assert (b.queries, b.scored, b.skipped, b.vehicles) == (2, 2, 0, 1)


def test_group_scores_unscored_group():
    scores = _scores(SKIP, SKIP_LABELS)
    result = group_scores(scores, ["x", "x", "y"], ["v1", "v1", "v2"])
    y = result["y"]
    assert y.precision_at_1 is None
    assert y.map_at_r is None
    assert (y.queries, y.scored, y.skipped, y.vehicles) == (1, 0, 1, 1)
    x = result["x"]
    assert x.precision_at_1 == pytest.approx(1.0)
    assert (x.queries, x.scored, x.skipped, x.vehicles) == (2, 2, 0, 1)


def test_group_scores_one_group_counts_vehicles():
    scores = _scores(CASE_A, CASE_A_LABELS)
    result = group_scores(scores, ["g"] * 4, ["v1", "v1", "v2", "v3"])
    assert list(result) == ["g"]
    assert result["g"].precision_at_1 == pytest.approx(0.75)
    assert result["g"].map_at_r == pytest.approx(0.75)
    assert result["g"].vehicles == 3


def test_group_scores_length_mismatch_raises():
    scores = _scores(CASE_A, CASE_A_LABELS)
    with pytest.raises(ValueError):
        group_scores(scores, ["A", "B"], ["v1"] * 4)
    with pytest.raises(ValueError):
        group_scores(scores, CASE_A_LABELS, ["v1"] * 3)


def test_labels_length_mismatch_raises():
    with pytest.raises(ValueError):
        _scores(CASE_A, ["A", "A", "B"])


def test_queries_must_be_two_dimensional():
    with pytest.raises(ValueError):
        query_scores(_index(CASE_A), np.ones(4), CASE_A_LABELS)


def test_index_size_must_match_queries():
    with pytest.raises(ValueError):
        query_scores(_index(CASE_A), CASE_A[:3], CASE_A_LABELS[:3])


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_size_must_be_positive(block_size):
    with pytest.raises(ValueError):
        _scores(CASE_A, CASE_A_LABELS, block_size=block_size)


def test_empty_queries_raises():
    index = BruteForceIndex(2)
    with pytest.raises(ValueError):
        query_scores(index, np.zeros((0, 2)), [])


def test_integer_label_array():
    scores = _scores(CASE_A, np.array([1, 1, 2, 2]))
    assert scores.precision_at_1.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_inputs_are_not_modified():
    vectors = CASE_B.copy()
    labels = list(CASE_B_LABELS)
    _scores(vectors, labels)
    np.testing.assert_array_equal(vectors, CASE_B)
    assert labels == CASE_B_LABELS


def test_group_score_is_frozen():
    score = GroupScore(0.5, 0.25, 2, 2, 0, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.scored = 3


def test_camera_aware_ignores_same_camera_lookalikes():
    vectors = np.array([(1, 0), (1, 0.01), (1, 0.2), (0, 1)], dtype=np.float64)
    vehicles = ["A", "A", "B", "A"]
    cameras = [1, 1, 1, 2]
    scores = camera_aware_scores(_index(vectors), vectors, vehicles, cameras)
    assert scores.relevant.tolist() == [1, 1, 0, 2]
    assert scores.precision_at_1[[0, 1, 3]].tolist() == [0.0, 0.0, 0.0]
    assert scores.average_precision[3] == pytest.approx(0.25)


def test_camera_aware_perfect_embedding():
    vectors = np.array([(1, 0), (1, 0.01), (1, -0.01), (0, 1)], dtype=np.float64)
    vehicles = ["A", "A", "A", "B"]
    cameras = [1, 1, 2, 1]
    scores = camera_aware_scores(_index(vectors), vectors, vehicles, cameras)
    assert scores.relevant.tolist() == [1, 1, 2, 0]
    assert scores.precision_at_1[:3].tolist() == [1.0, 1.0, 1.0]
    assert scores.average_precision[:3].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_camera_aware_rejects_bad_input():
    vectors = np.array([(1, 0), (0, 1)], dtype=np.float64)
    index = _index(vectors)
    with pytest.raises(ValueError):
        camera_aware_scores(index, vectors, ["A", "B"], [1])
    with pytest.raises(ValueError):
        camera_aware_scores(index, vectors, ["A", "B"], [1, 1])
