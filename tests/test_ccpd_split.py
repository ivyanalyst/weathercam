from collections import Counter

import pytest

from weathercam.data.ccpd import parse_name, subset_of
from weathercam.data.ccpd_split import ccpd_split, duplicates

PARTS = ("train", "val", "test")


def _plate(province, k):
    return f"{province}_0_0_0_0_{k // 35}_{k % 35}"


def _path(subset, tag, province=0, k=0):
    return f"CCPD2019/{subset}/{tag}-0_0-1&2_3&4-3&4_1&4_1&2_3&2-{_plate(province, k)}-0-0.jpg"


def _base(n, province=0, start=0):
    return [_path("ccpd_base", f"b{province}x{k}", province, k) for k in range(start, start + n)]


def _mixed():
    base = _base(30) + [_path("ccpd_base", "dup", 0, 3)]
    blur = [_path("ccpd_blur", f"u{k}", 2, k) for k in range(10)]
    tilt = [_path("ccpd_tilt", f"u{k}", 2, k) for k in range(5)]
    return base + blur + tilt + [_path("ccpd_fn", "w", 0, 2), "CCPD2019/ccpd_np/2606.jpg"]


def _run(paths, seed=1, pool=20, val_fraction=0.25, test_cap=10):
    return ccpd_split(paths, seed=seed, pool=pool, val_fraction=val_fraction, test_cap=test_cap)


def _all(result):
    return [p for part in PARTS for p in result[part]]


def test_unannotated_names_are_dropped():
    result = _run(_base(4) + ["CCPD2019/ccpd_np/2606.jpg"], pool=10)
    assert result["removed"]["unparsable"] == 1
    assert all("ccpd_np" not in p for p in _all(result))


def test_name_in_two_subsets_is_dropped_everywhere():
    paths = _base(4) + [
        _path("ccpd_tilt", "t1", 1, 5),
        _path("ccpd_fn", "t1", 1, 5),
        _path("ccpd_fn", "t2", 1, 6),
    ]
    result = _run(paths, pool=10)
    assert result["removed"]["name_overlap"] == 2
    assert result["test"] == [_path("ccpd_fn", "t2", 1, 6)]


def test_base_keeps_one_image_per_plate():
    same = [_path("ccpd_base", f"s{i}", 0, 7) for i in range(3)]
    result = _run(same + _base(3, start=100), pool=10)
    assert len(result["train"]) + len(result["val"]) == 4
    assert result["removed"]["base_duplicate_plate"] == 2
    assert len([p for p in _all(result) if p in same]) == 1


def test_test_plate_found_in_base_is_dropped_even_if_not_sampled():
    test = [_path("ccpd_blur", f"t{k}", 0, k) for k in range(5)]
    result = _run(_base(5) + test, pool=1, val_fraction=0.5)
    assert result["test"] == []
    assert result["removed"]["plate_in_base"] == 5
    assert result["removed"]["base_not_sampled"] == 4
    assert len(result["train"]) + len(result["val"]) == 1


def test_test_plate_repeated_across_subsets_keeps_one():
    test = [
        _path("ccpd_blur", "a", 1, 3),
        _path("ccpd_tilt", "b", 1, 3),
        _path("ccpd_fn", "c", 1, 4),
    ]
    result = _run(_base(4) + test, pool=10)
    plates = [parse_name(p).plate for p in result["test"]]
    assert len(plates) == 2 and len(set(plates)) == 2
    assert result["removed"]["test_duplicate_plate"] == 1


def test_output_has_no_duplicate_names_or_plates():
    result = _run(_mixed())
    assert duplicates(result) == {"names": 0, "plates": 0}
    plates = [parse_name(p).plate for p in _all(result)]
    assert len(plates) == len(set(plates))


def test_pool_is_split_into_train_and_val():
    result = _run(_base(40), pool=20, val_fraction=0.25, test_cap=5)
    assert (len(result["train"]), len(result["val"])) == (15, 5)
    assert not set(result["train"]) & set(result["val"])
    assert result["removed"]["base_not_sampled"] == 20


def test_split_is_stratified_by_province():
    result = _run(_base(20, province=0) + _base(20, province=1), pool=40, seed=3)
    counts = Counter(parse_name(p).plate[0] for p in result["val"])
    assert counts == {"皖": 5, "沪": 5}


def test_pool_larger_than_base_uses_all_of_base():
    result = _run(_base(10), pool=1000)
    assert len(result["train"]) + len(result["val"]) == 10
    assert result["removed"]["base_not_sampled"] == 0


def test_test_cap_applies_per_subset():
    test = [_path("ccpd_blur", f"x{k}", 1, k) for k in range(10)]
    test += [_path("ccpd_tilt", f"y{k}", 2, k) for k in range(3)]
    result = _run(_base(4) + test, pool=10, test_cap=4)
    assert Counter(subset_of(p) for p in result["test"]) == {"ccpd_blur": 4, "ccpd_tilt": 3}
    assert result["removed"]["test_over_cap"] == 6


def test_every_input_is_kept_or_counted_once():
    paths = _mixed()
    result = _run(paths)
    assert len(paths) == len(_all(result)) + sum(result["removed"].values())


def test_same_seed_is_reproducible():
    assert _run(_mixed(), seed=5) == _run(_mixed(), seed=5)


def test_different_seed_changes_result():
    assert _run(_mixed(), seed=1, pool=10) != _run(_mixed(), seed=2, pool=10)


def test_parts_are_sorted():
    result = _run(_mixed())
    for part in PARTS:
        assert result[part] == sorted(result[part])


def test_duplicates_counts_shared_plate():
    fake = {
        "train": [_path("ccpd_base", "a", 0, 1)],
        "val": [],
        "test": [_path("ccpd_blur", "b", 0, 1)],
    }
    assert duplicates(fake) == {"names": 0, "plates": 1}


def test_duplicates_counts_shared_name():
    same = _path("ccpd_base", "a", 0, 1)
    assert duplicates({"train": [same], "val": [same], "test": []}) == {"names": 1, "plates": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pool": 10, "val_fraction": 0.0, "test_cap": 5},
        {"pool": 10, "val_fraction": 1.0, "test_cap": 5},
        {"pool": 0, "val_fraction": 0.25, "test_cap": 5},
        {"pool": 10, "val_fraction": 0.25, "test_cap": 0},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        ccpd_split(_base(4), seed=1, **kwargs)
