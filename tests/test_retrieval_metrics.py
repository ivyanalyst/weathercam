import dataclasses

import numpy as np
import pytest

from weathercam.retrieval.metrics import (
    RetrievalScore,
    average_precision_at_r,
    evaluate_rankings,
)


def test_paper_case_only_first_correct():
    hits = [True] + [False] * 9
    assert average_precision_at_r(hits, 10) == pytest.approx(0.10)


def test_paper_case_first_and_last_correct():
    hits = [True] + [False] * 8 + [True]
    assert average_precision_at_r(hits, 10) == pytest.approx(0.12)


def test_perfect_ranking():
    assert average_precision_at_r([True, True, True], 3) == pytest.approx(1.0)


def test_no_hits():
    assert average_precision_at_r([False, False, False], 3) == 0.0


def test_hits_beyond_r_are_ignored():
    assert average_precision_at_r([True, False, True, True], 2) == pytest.approx(0.5)


def test_rank_order_matters():
    assert average_precision_at_r([True, False], 2) == pytest.approx(0.5)
    assert average_precision_at_r([False, True], 2) == pytest.approx(0.25)


def test_r_equals_one():
    assert average_precision_at_r([True], 1) == pytest.approx(1.0)
    assert average_precision_at_r([False, True], 1) == 0.0


def test_accepts_numpy_bool_array():
    hits = np.array([True, False, True])
    assert average_precision_at_r(hits, 3) == pytest.approx((1.0 + 2.0 / 3.0) / 3.0)


@pytest.mark.parametrize("r", [0, -1])
def test_average_precision_rejects_bad_r(r):
    with pytest.raises(ValueError):
        average_precision_at_r([True, False], r)


def test_average_precision_rejects_short_hits():
    with pytest.raises(ValueError):
        average_precision_at_r([True], 2)


def test_evaluate_mixed_queries():
    score = evaluate_rankings([[True, False], [False, True], [False]], [2, 1, 0])
    assert score.precision_at_1 == pytest.approx(0.5)
    assert score.map_at_r == pytest.approx(0.25)
    assert score.scored == 2
    assert score.skipped == 1


def test_evaluate_all_perfect():
    score = evaluate_rankings([[True, True], [True]], [2, 1])
    assert score.precision_at_1 == pytest.approx(1.0)
    assert score.map_at_r == pytest.approx(1.0)
    assert score.scored == 2
    assert score.skipped == 0


def test_evaluate_all_skipped_raises():
    with pytest.raises(ValueError):
        evaluate_rankings([[False], [False]], [0, 0])


def test_evaluate_empty_raises():
    with pytest.raises(ValueError):
        evaluate_rankings([], [])


def test_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_rankings([[True], [True]], [1])


def test_evaluate_negative_count_raises():
    with pytest.raises(ValueError):
        evaluate_rankings([[True], [True]], [1, -1])


def test_evaluate_short_hits_raises():
    with pytest.raises(ValueError):
        evaluate_rankings([[True]], [2])


def test_evaluate_skipped_query_may_have_empty_hits():
    score = evaluate_rankings([[], [True]], [0, 1])
    assert score.scored == 1
    assert score.skipped == 1
    assert score.precision_at_1 == pytest.approx(1.0)


def test_score_is_frozen():
    score = RetrievalScore(0.5, 0.25, 2, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.scored = 3
