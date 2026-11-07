import random
import re

from weathercam.data.neardup import UnionFind

_AD = re.compile(r"_[0-9A-Za-z]{5}_([0-9A-Za-z]{11})_\d+x\d+\.")


def make_of(path):
    return path.split("/", 1)[0].split("_", 1)[0]


def ad_id(path):
    found = _AD.search(path.split("/", 1)[-1])
    return found.group(1) if found else None


def make_list(counts, min_images):
    kept = sorted(make for make, count in counts.items() if count >= min_images)
    return kept + ["unknown"]


def encode_make(make, makes):
    if make in makes:
        return makes.index(make)
    return makes.index("unknown")


def _positions(keys):
    seen = {}
    for index, key in enumerate(keys):
        seen.setdefault(key, []).append(index)
    return seen


def cross_make_drop(members):
    seen = _positions([(crc, size) for _, crc, size in members])
    dropped = []
    for indices in seen.values():
        if len({make_of(members[i][0]) for i in indices}) > 1:
            dropped.extend(indices)
    return sorted(dropped)


def group_ids(members):
    forest = UnionFind(len(members))
    for indices in _positions([(crc, size) for _, crc, size in members]).values():
        for i in indices[1:]:
            forest.union(indices[0], i)
    first = {}
    for index, (path, _, _) in enumerate(members):
        key = ad_id(path)
        if key is None:
            continue
        if key in first:
            forest.union(first[key], index)
        else:
            first[key] = index
    return [forest.find(i) for i in range(len(members))]


def capped_indices(makes, groups, cap, seed):
    if len(makes) != len(groups):
        raise ValueError(f"length mismatch: {len(makes)} makes, {len(groups)} groups")
    by_make = {}
    for index, (make, group) in enumerate(zip(makes, groups)):
        by_make.setdefault(make, {}).setdefault(group, []).append(index)
    rng = random.Random(seed)
    picked = []
    for make in sorted(by_make, key=str):
        order = list(by_make[make].values())
        rng.shuffle(order)
        room = cap
        for members in order:
            if len(members) <= room:
                picked.extend(members)
                room -= len(members)
    return sorted(picked)
