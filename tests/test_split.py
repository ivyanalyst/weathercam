import pytest

from weathercam.data.split import group_split


def build(spec, samples=1):
    groups = []
    classes = []
    for cls, n in spec.items():
        for k in range(n):
            for _ in range(samples):
                groups.append(f"{cls}-{k}")
                classes.append(cls)
    return groups, classes


def group_counts(parts, groups, classes):
    result = {}
    for p, indices in enumerate(parts):
        seen = {(classes[i], groups[i]) for i in indices}
        for cls, _ in seen:
            result.setdefault(cls, [0] * len(parts))[p] += 1
    return result


def test_every_index_assigned_once():
    groups, classes = build({"a": 10, "b": 7}, samples=3)
    parts = group_split(groups, classes, (0.8, 0.2), seed=1)
    assert len(parts) == 2
    assert sorted(i for part in parts for i in part) == list(range(len(groups)))


def test_parts_are_sorted():
    groups, classes = build({"a": 10, "b": 7}, samples=3)
    for part in group_split(groups, classes, (0.8, 0.2), seed=1):
        assert part == sorted(part)


def test_groups_stay_together():
    groups, classes = build({"a": 10, "b": 7}, samples=3)
    parts = group_split(groups, classes, (0.8, 0.2), seed=1)
    where = {}
    for p, indices in enumerate(parts):
        for i in indices:
            where.setdefault(groups[i], set()).add(p)
    assert len(where) == 17
    assert all(len(v) == 1 for v in where.values())


def test_same_seed_is_reproducible():
    groups, classes = build({"a": 20, "b": 20}, samples=2)
    first = group_split(groups, classes, (0.8, 0.2), seed=5)
    second = group_split(groups, classes, (0.8, 0.2), seed=5)
    assert first == second


def test_different_seed_changes_result():
    groups, classes = build({"a": 40})
    first = group_split(groups, classes, (0.8, 0.2), seed=1)
    second = group_split(groups, classes, (0.8, 0.2), seed=2)
    assert first != second


def test_stratified_group_counts():
    groups, classes = build({"a": 10, "b": 10}, samples=3)
    parts = group_split(groups, classes, (0.8, 0.2), seed=3)
    assert group_counts(parts, groups, classes) == {"a": [8, 2], "b": [8, 2]}


def test_three_parts_counts():
    groups, classes = build({"a": 10})
    parts = group_split(groups, classes, (0.6, 0.2, 0.2), seed=3)
    assert group_counts(parts, groups, classes) == {"a": [6, 2, 2]}


def test_remainder_goes_to_earlier_part_on_tie():
    groups, classes = build({"a": 7})
    parts = group_split(groups, classes, (0.5, 0.5), seed=3)
    assert group_counts(parts, groups, classes) == {"a": [4, 3]}


def test_remainder_goes_to_largest_remainder():
    groups, classes = build({"a": 3})
    parts = group_split(groups, classes, (0.8, 0.2), seed=3)
    assert group_counts(parts, groups, classes) == {"a": [2, 1]}


def test_single_group_class_goes_to_first_part():
    groups, classes = build({"a": 10, "rare": 1}, samples=2)
    parts = group_split(groups, classes, (0.8, 0.2), seed=3)
    assert group_counts(parts, groups, classes)["rare"] == [1, 0]


def test_two_group_class_is_spread_over_two_parts():
    groups, classes = build({"a": 10, "pair": 2}, samples=2)
    parts = group_split(groups, classes, (0.8, 0.2), seed=3)
    assert group_counts(parts, groups, classes)["pair"] == [1, 1]


def test_unique_groups_like_dawn():
    groups, classes = build({"Fog": 10, "Rain": 5})
    parts = group_split(groups, classes, (0.8, 0.2), seed=3)
    assert group_counts(parts, groups, classes) == {"Fog": [8, 2], "Rain": [4, 1]}


def test_empty_input():
    assert group_split([], [], (0.8, 0.2), seed=1) == [[], []]


def test_group_with_two_classes_raises():
    with pytest.raises(ValueError):
        group_split(["g", "g"], ["a", "b"], (0.8, 0.2), seed=1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        group_split(["g", "h"], ["a", "a", "a"], (0.8, 0.2), seed=1)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        group_split(["g"], ["a"], (0.8, 0.3), seed=1)


def test_fractions_must_be_positive():
    with pytest.raises(ValueError):
        group_split(["g"], ["a"], (1.0, 0.0), seed=1)
    with pytest.raises(ValueError):
        group_split(["g"], ["a"], (1.2, -0.2), seed=1)


def test_needs_two_parts():
    with pytest.raises(ValueError):
        group_split(["g"], ["a"], (1.0,), seed=1)
