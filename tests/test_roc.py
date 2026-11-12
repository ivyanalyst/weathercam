import numpy as np
import pytest

from weathercam.core.roc_metrics import roc_auc


def test_classic_example():
    assert roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)


def test_perfect_separation():
    assert roc_auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_reversed_separation():
    assert roc_auc(np.array([4.0, 3.0, 2.0, 1.0]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)


def test_ties_get_half_credit():
    assert roc_auc(np.array([1.0, 2.0, 2.0, 3.0]), np.array([0, 0, 1, 1])) == pytest.approx(0.875)


def test_all_equal_scores():
    assert roc_auc(np.array([2.0, 2.0, 2.0, 2.0]), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_monotone_transform_invariant():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.5, 0.2])
    labels = np.array([0, 0, 1, 1, 1, 0])
    base = roc_auc(scores, labels)
    assert roc_auc(scores * 3.0 + 1.0, labels) == pytest.approx(base)
    assert roc_auc(np.exp(scores), labels) == pytest.approx(base)


def test_order_invariant():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.5, 0.2])
    labels = np.array([0, 0, 1, 1, 1, 0])
    perm = np.array([3, 0, 5, 1, 4, 2])
    assert roc_auc(scores[perm], labels[perm]) == pytest.approx(roc_auc(scores, labels))


def test_accepts_bool_labels():
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(scores, np.array([False, False, True, True])) == pytest.approx(0.75)


def test_matches_pair_count():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=60).astype(float)
    labels = rng.integers(0, 2, size=60)
    total = 0.0
    for p in scores[labels == 1]:
        for n in scores[labels == 0]:
            total += 1.0 if p > n else 0.5 if p == n else 0.0
    expected = total / ((labels == 1).sum() * (labels == 0).sum())
    assert roc_auc(scores, labels) == pytest.approx(expected, abs=1e-12)


def test_rejects_only_negatives():
    with pytest.raises(ValueError):
        roc_auc(np.array([1.0, 2.0]), np.array([0, 0]))


def test_rejects_only_positives():
    with pytest.raises(ValueError):
        roc_auc(np.array([1.0, 2.0]), np.array([1, 1]))


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        roc_auc(np.array([1.0, 2.0, 3.0]), np.array([0, 1]))


def test_rejects_empty():
    with pytest.raises(ValueError):
        roc_auc(np.array([]), np.array([]))


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        roc_auc(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]))


def test_rejects_nan_scores():
    with pytest.raises(ValueError):
        roc_auc(np.array([1.0, np.nan]), np.array([0, 1]))


def test_rejects_2d_scores():
    with pytest.raises(ValueError):
        roc_auc(np.ones((2, 2)), np.array([0, 1]))
