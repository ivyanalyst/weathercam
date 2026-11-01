from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassScore:
    precision: float
    recall: float
    f1: float
    support: int


def _validate(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}")
    if not y_true:
        raise ValueError("y_true must not be empty")


def accuracy(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    _validate(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> list[list[int]]:
    _validate(y_true, y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    elif len(set(labels)) != len(labels):
        raise ValueError(f"labels must be unique, got {list(labels)}")
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"Label not in labels: true={t!r}, predicted={p!r}")
        matrix[index[t]][index[p]] += 1
    return matrix


def per_class_scores(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> dict[Hashable, ClassScore]:
    if labels is None:
        _validate(y_true, y_pred)
        labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels)
    scores: dict[Hashable, ClassScore] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = ClassScore(precision, recall, f1, support)
    return scores


def macro_f1(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> float:
    _validate(y_true, y_pred)
    scores = per_class_scores(y_true, y_pred, labels)
    values = [s.f1 for s in scores.values() if s.support > 0]
    return sum(values) / len(values)
