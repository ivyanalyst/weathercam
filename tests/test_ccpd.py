import pytest

from weathercam.data.ccpd import (
    ADS,
    ALPHABETS,
    PROVINCES,
    PROVINCES_LATIN,
    decode_plate,
    from_latin_plate,
    hull_box,
    parse_name,
    subset_of,
    to_latin_plate,
)

OFFICIAL = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
BASE = "0111242816092-89_90-265&500_443&565-449&556_277&578_277&515_449&493-0_0_27_33_33_23_26-144-12.jpg"


def _name(box="1&2_3&4", verts="3&4_1&4_1&2_3&2", plate="0_0_0_0_0_0_0"):
    return f"0-0_0-{box}-{verts}-{plate}-0-0.jpg"


def test_table_sizes():
    assert (len(PROVINCES), len(ALPHABETS), len(ADS)) == (34, 25, 35)


def test_decode_plate_official_example():
    assert decode_plate([0, 0, 22, 27, 27, 33, 16]) == "皖AY339S"


def test_decode_plate_base_sample():
    assert decode_plate([0, 0, 27, 33, 33, 23, 26]) == "皖A399Z2"


def test_decode_plate_other_province():
    assert decode_plate([12, 1, 24, 24, 24, 24, 24]) == "京B00000"


def test_decode_plate_wrong_length_raises():
    with pytest.raises(ValueError):
        decode_plate([0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        decode_plate([0, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize(
    "indices",
    [
        [34, 0, 0, 0, 0, 0, 0],
        [0, 25, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 35],
        [-1, 0, 0, 0, 0, 0, 0],
    ],
)
def test_decode_plate_index_out_of_range_raises(indices):
    with pytest.raises(ValueError):
        decode_plate(indices)


def test_hull_box_from_vertices():
    assert hull_box([(449, 556), (277, 578), (277, 515), (449, 493)]) == (277, 493, 449, 578)


def test_hull_box_needs_four_vertices():
    with pytest.raises(ValueError):
        hull_box([(1, 2), (3, 4), (5, 6)])


def test_parse_name_official_example():
    parsed = parse_name(OFFICIAL)
    assert parsed.plate == "皖AY339S"
    assert parsed.vertices[0] == (386, 473)
    assert parsed.box == (154, 383, 386, 473)
    assert parsed.field_box == (154, 383, 386, 473)


def test_parse_name_base_box_differs_from_field():
    parsed = parse_name(BASE)
    assert parsed.plate == "皖A399Z2"
    assert parsed.field_box == (265, 500, 443, 565)
    assert parsed.box == (277, 493, 449, 578)


def test_parse_name_accepts_archive_path():
    assert parse_name("CCPD2019/ccpd_base/" + BASE) == parse_name(BASE)


def test_parse_name_unannotated_name_raises():
    with pytest.raises(ValueError):
        parse_name("CCPD2019/ccpd_np/2606.jpg")


def test_parse_name_wrong_field_count_raises():
    with pytest.raises(ValueError):
        parse_name("0-0_0-1&2_3&4-3&4_1&4_1&2_3&2-0_0_0_0_0_0_0-0.jpg")


def test_parse_name_non_integer_coordinate_raises():
    with pytest.raises(ValueError):
        parse_name(_name(box="a&2_3&4"))


def test_parse_name_wrong_vertex_count_raises():
    with pytest.raises(ValueError):
        parse_name(_name(verts="3&4_1&4_1&2"))


def test_parse_name_bad_plate_index_raises():
    with pytest.raises(ValueError):
        parse_name(_name(plate="34_0_0_0_0_0_0"))


def test_subset_of():
    assert subset_of("CCPD2019/ccpd_base/" + BASE) == "ccpd_base"
    assert subset_of("ccpd_tilt/" + OFFICIAL) == "ccpd_tilt"


def test_subset_of_bare_name_raises():
    with pytest.raises(ValueError):
        subset_of(BASE)


def test_latin_table_size():
    assert len(PROVINCES_LATIN) == len(PROVINCES) == 34


def test_latin_codes_are_unique_uppercase_ascii():
    assert len(set(PROVINCES_LATIN)) == len(PROVINCES_LATIN)
    assert all(code.isascii() and code.isalpha() and code.isupper() for code in PROVINCES_LATIN)


def test_to_latin_plate_official_example():
    assert to_latin_plate("皖AY339S") == "AH-AY339S"


@pytest.mark.parametrize(
    "plate, expected",
    [
        ("京B00000", "BJ-B00000"),
        ("警A12345", "JG-A12345"),
        ("学A12345", "XE-A12345"),
        ("OA12345", "O-A12345"),
    ],
)
def test_to_latin_plate_other_characters(plate, expected):
    assert to_latin_plate(plate) == expected


def test_latin_round_trip_for_every_province_character():
    for province in PROVINCES:
        plate = province + "A12345"
        assert from_latin_plate(to_latin_plate(plate)) == plate


def test_to_latin_plate_wrong_length_raises():
    with pytest.raises(ValueError):
        to_latin_plate("皖A1234")


def test_to_latin_plate_unknown_province_raises():
    with pytest.raises(ValueError):
        to_latin_plate("ZA12345")


@pytest.mark.parametrize("text", ["AH-A1234", "XX-A12345", "AHA12345"])
def test_from_latin_plate_bad_form_raises(text):
    with pytest.raises(ValueError):
        from_latin_plate(text)
