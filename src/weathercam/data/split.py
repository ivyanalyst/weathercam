import math
import random


def _validate(groups, classes, fractions):
    if len(fractions) < 2:
        raise ValueError("need at least two parts")
    if len(groups) != len(classes):
        raise ValueError(f"length mismatch: {len(groups)} groups, {len(classes)} classes")
    if any(f <= 0 for f in fractions):
        raise ValueError(f"fractions must be positive: {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1: {fractions}")


def _group_classes(groups, classes):
    group_class = {}
    for group, cls in zip(groups, classes):
        if group_class.setdefault(group, cls) != cls:
            raise ValueError(f"group {group!r} has more than one class")
    return group_class


def _counts(n, fractions):
    parts = len(fractions)
    if n < parts:
        return [n] + [0] * (parts - 1)
    exact = [n * f for f in fractions]
    counts = [math.floor(x + 1e-9) for x in exact]
    remainders = [x - c for x, c in zip(exact, counts)]
    order = sorted(range(parts), key=lambda p: (-remainders[p], p))
    for p in order[: n - sum(counts)]:
        counts[p] += 1
    for p in range(parts):
        if counts[p] == 0:
            donor = max(range(parts), key=lambda q: (counts[q], -q))
            counts[donor] -= 1
            counts[p] += 1
    return counts


def group_split(groups, classes, fractions, seed):
    _validate(groups, classes, fractions)
    group_class = _group_classes(groups, classes)
    by_class = {}
    for group, cls in group_class.items():
        by_class.setdefault(cls, []).append(group)
    rng = random.Random(seed)
    part_of = {}
    for cls in sorted(by_class, key=str):
        members = by_class[cls]
        rng.shuffle(members)
        start = 0
        for p, count in enumerate(_counts(len(members), fractions)):
            for group in members[start : start + count]:
                part_of[group] = p
            start += count
    parts = [[] for _ in fractions]
    for index, group in enumerate(groups):
        parts[part_of[group]].append(index)
    return parts
