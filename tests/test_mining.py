import numpy as np
import pytest

from weathercam.retrieval.mining import k_center_greedy


def points():
    return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])


def test_five_point_order():
    assert k_center_greedy(points(), 5).tolist() == [0, 2, 1, 3, 4]


def test_start_changes_first_and_order():
    assert k_center_greedy(points(), 5, start=2).tolist() == [2, 0, 1, 3, 4]


def test_budget_one_returns_start():
    assert k_center_greedy(points(), 1, start=3).tolist() == [3]


def test_prefix_property():
    assert k_center_greedy(points(), 3).tolist() == [0, 2, 1]


def test_full_budget_is_permutation():
    out = k_center_greedy(points(), 5)
    assert sorted(out.tolist()) == [0, 1, 2, 3, 4]
    assert out.dtype == np.int64


def test_row_scale_invariant():
    scaled = points() * np.array([[5.0], [0.2], [3.0], [1.5], [0.7]])
    assert k_center_greedy(scaled, 5).tolist() == [0, 2, 1, 3, 4]


def test_duplicates_are_picked_last():
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert k_center_greedy(x, 2).tolist() == [0, 2]
    assert k_center_greedy(x, 3).tolist() == [0, 2, 1]


def test_does_not_modify_input():
    x = points()
    k_center_greedy(x, 3)
    assert np.array_equal(x, points())


def test_rejects_budget_zero():
    with pytest.raises(ValueError):
        k_center_greedy(points(), 0)


def test_rejects_budget_above_rows():
    with pytest.raises(ValueError):
        k_center_greedy(points(), 6)


def test_rejects_start_out_of_range():
    with pytest.raises(ValueError):
        k_center_greedy(points(), 2, start=5)


def test_rejects_1d_input():
    with pytest.raises(ValueError):
        k_center_greedy(np.ones(4), 2)


def test_rejects_zero_row():
    with pytest.raises(ValueError):
        k_center_greedy(np.array([[1.0, 0.0], [0.0, 0.0]]), 1)
