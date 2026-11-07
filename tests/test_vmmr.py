import pytest

from weathercam.data.vmmr import (
    ad_id,
    capped_indices,
    cross_make_drop,
    encode_make,
    group_ids,
    make_list,
    make_of,
)


def test_make_of_simple():
    assert make_of("acura_cl_1997/a.jpg") == "acura"


def test_make_of_multi_word():
    assert make_of("mercedes benz_c class_2010/a.jpg") == "mercedes benz"


def test_make_of_bare_folder():
    assert make_of("landrover_defender_2005") == "landrover"


def test_ad_id_found():
    path = "acura_cl_1997/1997 acura cl_00p0p_lAPdeE6t8OG_600x450.jpg"
    assert ad_id(path) == "lAPdeE6t8OG"


def test_ad_id_none_for_numbered():
    assert ad_id("acura_cl_1997/acura_cl_1997_0.jpg") is None


def test_make_list_threshold():
    counts = {"ford": 150, "nash": 2, "audi": 100, "kia": 99}
    assert make_list(counts, 100) == ["audi", "ford", "unknown"]


def test_make_list_sorted_unknown_last():
    assert make_list({"b": 5, "a": 5}, 1) == ["a", "b", "unknown"]


def test_encode_make_known():
    assert encode_make("ford", ["audi", "ford", "unknown"]) == 1


def test_encode_make_unknown():
    assert encode_make("nash", ["audi", "ford", "unknown"]) == 2


def test_cross_make_drop_flags_shared_bytes():
    members = [
        ("acura_cl_1997/a.jpg", 1, 10),
        ("hyundai_x_2000/b.jpg", 1, 10),
        ("honda_x_2000/c.jpg", 2, 10),
        ("honda_y_2001/d.jpg", 2, 10),
        ("kia_z_2000/e.jpg", 1, 11),
    ]
    assert cross_make_drop(members) == [0, 1]


def test_cross_make_drop_none():
    members = [("a_x_2000/1.jpg", 1, 1), ("b_y_2000/2.jpg", 2, 1)]
    assert cross_make_drop(members) == []


def test_group_ids_identical_bytes():
    members = [
        ("a_x_2000/1.jpg", 5, 9),
        ("a_x_2000/2.jpg", 6, 9),
        ("b_y_2001/3.jpg", 7, 9),
        ("c_z_2002/4.jpg", 5, 9),
    ]
    assert group_ids(members) == [0, 1, 2, 0]


def test_group_ids_repeated_ad_id():
    members = [
        ("a_x_2000/2000 a x_00p0p_lAPdeE6t8OG_600x450.jpg", 1, 1),
        ("a_x_2000/2000 a x_01111_lAPdeE6t8OG_600x450.jpg", 2, 1),
        ("a_x_2000/a_x_2000_0.jpg", 3, 1),
    ]
    assert group_ids(members) == [0, 0, 2]


def test_group_ids_transitive():
    members = [
        ("a_x_2000/p.jpg", 1, 1),
        ("a_x_2000/2000 a_00p0p_AAAAAAAAAAA_600x450.jpg", 1, 1),
        ("a_x_2000/2000 a_01111_AAAAAAAAAAA_600x450.jpg", 2, 1),
    ]
    assert group_ids(members) == [0, 0, 0]


def test_group_ids_empty():
    assert group_ids([]) == []


def test_capped_all_fit():
    assert capped_indices(["a", "a", "b"], [0, 1, 2], 5, 42) == [0, 1, 2]


def test_capped_skips_group_that_does_not_fit():
    assert capped_indices(["a", "a", "a", "a"], [0, 0, 0, 1], 2, 42) == [3]


def test_capped_reaches_cap_with_singletons():
    picked = capped_indices(["a"] * 5, [0, 1, 2, 3, 4], 3, 42)
    assert len(picked) == 3
    assert picked == sorted(set(picked))
    assert set(picked) <= {0, 1, 2, 3, 4}


def test_capped_deterministic():
    makes = ["a"] * 10
    groups = list(range(10))
    assert capped_indices(makes, groups, 4, 42) == capped_indices(makes, groups, 4, 42)


def test_capped_keeps_groups_whole():
    picked = capped_indices(["a"] * 6, [0, 0, 1, 1, 2, 2], 4, 42)
    assert len(picked) == 4
    for group in (0, 1, 2):
        members = [i for i in picked if [0, 0, 1, 1, 2, 2][i] == group]
        assert len(members) in (0, 2)


def test_capped_per_make_independent():
    picked = capped_indices(["a", "a", "a", "b"], [0, 1, 2, 3], 2, 42)
    assert 3 in picked
    assert len([i for i in picked if i != 3]) == 2
    assert len(picked) == 3


def test_capped_length_mismatch():
    with pytest.raises(ValueError):
        capped_indices(["a", "a"], [0], 2, 42)
