from collections import Counter

import pytest

from weathercam.review.queue import select_ccpd, select_veri


def veri(rows):
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def files(subset, count):
    return [f"images/{subset}_{i:03d}.jpg" for i in range(count)]


def subset_of(path):
    return path.split("/")[1].split("_")[0]


TEST_FILES = {"a": files("a", 20), "b": files("b", 20), "c": files("c", 20)}


def test_veri_returns_n_indices_from_requested_part():
    rows = [(v, v % 4, "test") for v in range(40)] + [(1000 + v, v % 4, "train") for v in range(40)]
    vids, cams, split = veri(rows)
    picked = select_veri(vids, cams, split, "test", 12, 42)
    assert len(picked) == 12 and len(set(picked)) == 12
    assert all(split[i] == "test" for i in picked)
    assert picked == sorted(picked)


def test_veri_one_image_per_vehicle():
    rows = [(v, c, "test") for v in range(50) for c in range(3)]
    vids, cams, split = veri(rows)
    picked = select_veri(vids, cams, split, "test", 30, 42)
    chosen = [vids[i] for i in picked]
    assert len(chosen) == 30 and len(set(chosen)) == 30


def test_veri_spread_over_cameras():
    rows = [(c * 100 + k, c, "test") for c in range(4) for k in range(10)]
    vids, cams, split = veri(rows)
    picked = select_veri(vids, cams, split, "test", 8, 42)
    assert Counter(cams[i] for i in picked) == {0: 2, 1: 2, 2: 2, 3: 2}


def test_veri_short_camera_lets_others_fill_up():
    rows = [(1, 0, "test")] + [(10 + k, 1, "test") for k in range(10)]
    vids, cams, split = veri(rows)
    picked = select_veri(vids, cams, split, "test", 5, 42)
    assert Counter(cams[i] for i in picked) == {0: 1, 1: 4}


def test_veri_fewer_vehicles_than_n_returns_one_per_vehicle():
    rows = [(v, c, "test") for v in range(5) for c in range(2)]
    vids, cams, split = veri(rows)
    picked = select_veri(vids, cams, split, "test", 20, 42)
    assert sorted(vids[i] for i in picked) == [0, 1, 2, 3, 4]


def test_veri_same_seed_same_result_other_seed_differs():
    rows = [(v, v % 5, "test") for v in range(300)]
    vids, cams, split = veri(rows)
    a = select_veri(vids, cams, split, "test", 20, 42)
    b = select_veri(vids, cams, split, "test", 20, 42)
    c = select_veri(vids, cams, split, "test", 20, 43)
    assert a == b
    assert a != c


def test_veri_zero_returns_empty():
    vids, cams, split = veri([(1, 0, "test"), (2, 1, "test")])
    assert select_veri(vids, cams, split, "test", 0, 42) == []


def test_veri_negative_n_raises():
    vids, cams, split = veri([(1, 0, "test")])
    with pytest.raises(ValueError):
        select_veri(vids, cams, split, "test", -1, 42)


def test_veri_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_veri([1, 2, 3], [0, 1], ["test", "test", "test"], "test", 2, 42)


def test_veri_part_absent_returns_empty():
    vids, cams, split = veri([(1, 0, "test"), (2, 1, "test")])
    assert select_veri(vids, cams, split, "val", 5, 42) == []


def test_ccpd_fixed_files_come_first_with_reasons():
    fixed = {"no_detection": ["images/a_000.jpg"], "iou_le_0.5": ["images/b_003.jpg", "images/b_001.jpg"]}
    out = select_ccpd(TEST_FILES, fixed, 2, 42)
    assert out[:3] == [
        ("images/a_000.jpg", ("no_detection",)),
        ("images/b_001.jpg", ("iou_le_0.5",)),
        ("images/b_003.jpg", ("iou_le_0.5",)),
    ]
    assert all(reasons == ("random",) for _file, reasons in out[3:])
    assert len(out) == 3 + 6


def test_ccpd_overlap_gives_one_entry_with_both_reasons():
    x, y = "images/a_005.jpg", "images/a_006.jpg"
    fixed = {"no_detection": [x], "iou_le_0.5": [x, y]}
    out = select_ccpd(TEST_FILES, fixed, 0, 42)
    assert out == [(x, ("no_detection", "iou_le_0.5")), (y, ("iou_le_0.5",))]


def test_ccpd_random_excludes_fixed_files():
    fixed = {"label_off_plate": files("a", 20)[:10]}
    out = select_ccpd(TEST_FILES, fixed, 5, 42)
    random_a = [f for f, reasons in out if reasons == ("random",) and subset_of(f) == "a"]
    assert len(random_a) == 5
    assert set(random_a) <= set(files("a", 20)[10:])


def test_ccpd_subset_with_fewer_free_files_gives_all_of_them():
    fixed = {"label_off_plate": files("a", 20)[:18]}
    out = select_ccpd(TEST_FILES, fixed, 5, 42)
    random_a = sorted(f for f, reasons in out if reasons == ("random",) and subset_of(f) == "a")
    assert random_a == ["images/a_018.jpg", "images/a_019.jpg"]


def test_ccpd_random_count_per_subset():
    out = select_ccpd(TEST_FILES, {}, 5, 42)
    assert len(out) == 15
    assert Counter(subset_of(f) for f, _reasons in out) == {"a": 5, "b": 5, "c": 5}


def test_ccpd_same_seed_same_result_other_seed_differs():
    a = select_ccpd(TEST_FILES, {}, 5, 42)
    b = select_ccpd(TEST_FILES, {}, 5, 42)
    c = select_ccpd(TEST_FILES, {}, 5, 43)
    assert a == b
    assert a != c


def test_ccpd_fixed_file_not_in_test_files_raises():
    with pytest.raises(ValueError):
        select_ccpd(TEST_FILES, {"no_detection": ["images/zzz.jpg"]}, 2, 42)


def test_ccpd_negative_per_subset_raises():
    with pytest.raises(ValueError):
        select_ccpd(TEST_FILES, {}, -1, 42)


def test_ccpd_per_subset_zero_gives_only_fixed():
    assert select_ccpd(TEST_FILES, {}, 0, 42) == []
    out = select_ccpd(TEST_FILES, {"no_detection": ["images/c_001.jpg"]}, 0, 42)
    assert out == [("images/c_001.jpg", ("no_detection",))]


def test_ccpd_random_section_sorted_by_file():
    out = select_ccpd(TEST_FILES, {}, 6, 42)
    names = [f for f, _reasons in out]
    assert names == sorted(names)
