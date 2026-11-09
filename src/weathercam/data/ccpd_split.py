import random
from collections import Counter, defaultdict

from weathercam.data.ccpd import parse_name, subset_of
from weathercam.data.split import group_split

PARTS = ("train", "val", "test")
_REMOVED = (
    "unparsable",
    "name_overlap",
    "base_duplicate_plate",
    "base_not_sampled",
    "plate_in_base",
    "test_duplicate_plate",
    "test_over_cap",
)


def _name(path):
    return path.split("/")[-1]


def _one_per_plate(items, rng):
    by_plate = defaultdict(list)
    for path, plate in items:
        by_plate[plate].append(path)
    return [rng.choice(sorted(by_plate[plate])) for plate in sorted(by_plate)]


def ccpd_split(paths, seed, pool, val_fraction, test_cap):
    if pool < 1 or test_cap < 1:
        raise ValueError("pool and test_cap must be at least 1")
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    removed = dict.fromkeys(_REMOVED, 0)
    parsed = []
    for path in paths:
        try:
            parsed.append((path, subset_of(path), parse_name(path).plate))
        except ValueError:
            removed["unparsable"] += 1
    plate_of = {path: plate for path, _, plate in parsed}
    subsets_of_name = defaultdict(set)
    for path, subset, _ in parsed:
        subsets_of_name[_name(path)].add(subset)
    base_plates = {plate for _, subset, plate in parsed if subset == "ccpd_base"}
    kept = []
    for item in parsed:
        if len(subsets_of_name[_name(item[0])]) > 1:
            removed["name_overlap"] += 1
        else:
            kept.append(item)
    rng = random.Random(seed)
    base = [(path, plate) for path, subset, plate in kept if subset == "ccpd_base"]
    unique = _one_per_plate(base, rng)
    removed["base_duplicate_plate"] = len(base) - len(unique)
    candidates = sorted(unique)
    if pool < len(candidates):
        candidates = sorted(rng.sample(candidates, pool))
    removed["base_not_sampled"] = len(unique) - len(candidates)
    plates = [plate_of[path] for path in candidates]
    fractions = (1 - val_fraction, val_fraction)
    train_idx, val_idx = group_split(plates, [plate[0] for plate in plates], fractions, seed)
    other = [(path, plate) for path, subset, plate in kept if subset != "ccpd_base"]
    clean = [(path, plate) for path, plate in other if plate not in base_plates]
    removed["plate_in_base"] = len(other) - len(clean)
    chosen = _one_per_plate(clean, rng)
    removed["test_duplicate_plate"] = len(clean) - len(chosen)
    by_subset = defaultdict(list)
    for path in chosen:
        by_subset[subset_of(path)].append(path)
    test = []
    for subset in sorted(by_subset):
        members = sorted(by_subset[subset])
        if len(members) > test_cap:
            removed["test_over_cap"] += len(members) - test_cap
            members = rng.sample(members, test_cap)
        test.extend(members)
    return {
        "train": sorted(candidates[i] for i in train_idx),
        "val": sorted(candidates[i] for i in val_idx),
        "test": sorted(test),
        "removed": removed,
    }


def duplicates(split):
    everything = [path for part in PARTS for path in split[part]]
    names = Counter(_name(path) for path in everything)
    plates = Counter(parse_name(path).plate for path in everything)
    return {
        "names": sum(1 for n in names.values() if n > 1),
        "plates": sum(1 for n in plates.values() if n > 1),
    }
