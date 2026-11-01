import pytest

from weathercam.core.cls_metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_scores,
)

Y_TRUE = [0, 0, 0, 1, 1, 2]
Y_PRED = [0, 0, 1, 1, 2, 2]


def test_accuracy_perfect():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_value():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)


def test_accuracy_string_labels():
    assert accuracy(["red", "blue", "red"], ["red", "red", "red"]) == pytest.approx(2 / 3)


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_confusion_matrix_values():
    assert confusion_matrix(Y_TRUE, Y_PRED) == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_confusion_matrix_sorted_string_labels():
    matrix = confusion_matrix(["red", "red", "blue"], ["red", "blue", "blue"])
    assert matrix == [[1, 0], [1, 1]]


def test_confusion_matrix_explicit_label_order():
    matrix = confusion_matrix(["red", "red", "blue"], ["red", "blue", "blue"], labels=["red", "blue"])
    assert matrix == [[1, 1], [0, 1]]


def test_confusion_matrix_unlisted_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0, 2], labels=[0, 1])
    with pytest.raises(ValueError):
        confusion_matrix([0, 2], [0, 1], labels=[0, 1])


def test_confusion_matrix_unseen_explicit_label():
    assert confusion_matrix([0, 1], [0, 1], labels=[0, 1, 2]) == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_confusion_matrix_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])


def test_per_class_scores_values():
    scores = per_class_scores(Y_TRUE, Y_PRED)
    assert scores[0].precision == pytest.approx(1.0)
    assert scores[0].recall == pytest.approx(2 / 3)
    assert scores[0].f1 == pytest.approx(0.8)
    assert scores[0].support == 3
    assert scores[1].precision == pytest.approx(0.5)
    assert scores[1].recall == pytest.approx(0.5)
    assert scores[1].f1 == pytest.approx(0.5)
    assert scores[1].support == 2
    assert scores[2].precision == pytest.approx(0.5)
    assert scores[2].recall == pytest.approx(1.0)
    assert scores[2].f1 == pytest.approx(2 / 3)
    assert scores[2].support == 1


def test_per_class_scores_keys_sorted():
    assert list(per_class_scores(Y_TRUE, Y_PRED)) == [0, 1, 2]
    assert list(per_class_scores(Y_TRUE, Y_PRED, labels=[2, 0, 1])) == [2, 0, 1]


def test_per_class_never_predicted_class():
    scores = per_class_scores([0, 1], [0, 0])
    assert scores[0].precision == pytest.approx(0.5)
    assert scores[0].recall == pytest.approx(1.0)
    assert scores[0].f1 == pytest.approx(2 / 3)
    assert scores[1].precision == 0.0
    assert scores[1].recall == 0.0
    assert scores[1].f1 == 0.0
    assert scores[1].support == 1


def test_per_class_only_predicted_class():
    scores = per_class_scores([0, 0], [0, 1])
    assert scores[0].precision == pytest.approx(1.0)
    assert scores[0].recall == pytest.approx(0.5)
    assert scores[0].f1 == pytest.approx(2 / 3)
    assert scores[1].precision == 0.0
    assert scores[1].recall == 0.0
    assert scores[1].f1 == 0.0
    assert scores[1].support == 0


def test_macro_f1_value():
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx(59 / 90)


def test_macro_f1_perfect():
    assert macro_f1([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)


def test_macro_f1_excludes_labels_without_support():
    assert macro_f1([0, 0], [0, 0], labels=[0, 1]) == pytest.approx(1.0)
    assert macro_f1([0, 0], [0, 1]) == pytest.approx(2 / 3)


def test_macro_f1_never_predicted_class():
    assert macro_f1([0, 1], [0, 0]) == pytest.approx(1 / 3)


def test_macro_f1_empty_raises():
    with pytest.raises(ValueError):
        macro_f1([], [])


def test_macro_f1_length_mismatch_raises():
    with pytest.raises(ValueError):
        macro_f1([0, 1], [0])
