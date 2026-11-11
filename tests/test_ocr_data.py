from collections import Counter

import numpy as np
import pytest

from weathercam.data.ccpd import decode_plate
from weathercam.plates.ocr import POSITION_CLASSES
from weathercam.plates.ocr_data import draw_probabilities, encode_plate, non_wan_share

TRAIN_COUNTS = {
    "皖": 15315, "苏": 270, "浙": 118, "沪": 48, "豫": 34, "京": 33, "鄂": 30, "粤": 25,
    "鲁": 22, "闽": 18, "赣": 17, "冀": 15, "川": 9, "渝": 7, "晋": 6, "湘": 6, "陕": 6,
    "云": 4, "辽": 4, "津": 4, "甘": 2, "琼": 1, "吉": 1, "新": 1, "黑": 1, "贵": 1,
    "蒙": 1, "青": 1, "宁": 1,
}


def _chars(counts):
    return [char for char, count in counts.items() for _ in range(count)]


def test_encode_matches_indices_in_a_real_file_name():
    # CCPD name field 0_0_24_21_30_29_28 belongs to plate 皖A0X654
    assert encode_plate("皖A0X654") == (0, 0, 24, 21, 30, 29, 28)


def test_encode_second_known_plate():
    assert encode_plate("苏EZ6W75") == (10, 4, 23, 30, 20, 31, 29)


def test_round_trip_with_decode_and_index_range():
    plates = ["皖A0X654", "苏EZ6W75", "皖AX760S", "皖A5L588", "皖AOOOOO", "OAOOOOO", "OOOOOOO"]
    for plate in plates:
        indices = encode_plate(plate)
        assert decode_plate(indices) == plate
        assert all(0 <= i < size for i, size in zip(indices, POSITION_CLASSES))


def test_encode_rejects_wrong_length():
    for plate in ("皖A0X65", "皖A0X6543", ""):
        with pytest.raises(ValueError):
            encode_plate(plate)


def test_encode_rejects_character_not_allowed_at_position():
    for plate in ("皖I12345", "皖1ABCDE", "皖a0X654", "AHA0X65", "皖A0X65I"):
        with pytest.raises(ValueError):
            encode_plate(plate)


def test_encode_rejects_non_string():
    for value in (None, 1234567, ("皖", "A", "0", "X", "6", "5", "4")):
        with pytest.raises(ValueError):
            encode_plate(value)


def test_draw_probabilities_without_cap_follow_the_weights():
    chars = _chars({"皖": 1000, "苏": 10, "黑": 1})
    p = draw_probabilities(chars, exponent=0.5, cap_draws=1e9)
    assert p.sum() == pytest.approx(1.0)
    assert p[-1] / p[0] == pytest.approx(np.sqrt(1000))
    assert p[0] == p[999]


def test_draw_probabilities_cap_holds_and_singleton_hits_it():
    chars = _chars({"皖": 1000, "苏": 10, "黑": 1})
    n = len(chars)
    p = draw_probabilities(chars, exponent=0.5, cap_draws=16.0)
    assert p.sum() == pytest.approx(1.0)
    assert (p * n).max() <= 16.0 * (1 + 1e-9)
    assert p[-1] * n == pytest.approx(16.0)
    assert non_wan_share(p, chars) > 11 / n


def test_draw_probabilities_cap_one_is_uniform():
    chars = _chars({"皖": 1000, "苏": 10, "黑": 1})
    p = draw_probabilities(chars, exponent=0.5, cap_draws=1.0)
    assert np.allclose(p, 1 / len(chars))


def test_draw_probabilities_exponent_zero_is_uniform():
    chars = _chars({"皖": 50, "苏": 5, "黑": 1})
    p = draw_probabilities(chars, exponent=0.0, cap_draws=16.0)
    assert np.allclose(p, 1 / len(chars))


def test_draw_probabilities_on_the_real_train_counts():
    chars = _chars(TRAIN_COUNTS)
    n = len(chars)
    assert n == 16001
    p = draw_probabilities(chars)
    assert p.sum() == pytest.approx(1.0)
    assert (p * n).max() <= 16.0 * (1 + 1e-9)
    singles = [i for i, char in enumerate(chars) if TRAIN_COUNTS[char] == 1]
    assert len(singles) == 8
    assert np.allclose(p[singles] * n, 16.0)
    share = non_wan_share(p, chars)
    assert 0.30 < share < 0.45
    assert share > 686 / 16001


def test_draw_probabilities_reject_bad_arguments():
    with pytest.raises(ValueError):
        draw_probabilities([])
    with pytest.raises(ValueError):
        draw_probabilities(["皖", "苏"], exponent=-0.1)
    with pytest.raises(ValueError):
        draw_probabilities(["皖", "苏"], cap_draws=0.5)


def test_non_wan_share_sums_the_other_provinces():
    chars = ["皖", "苏", "皖", "浙"]
    assert non_wan_share([0.1, 0.2, 0.3, 0.4], chars) == pytest.approx(0.6)
    assert Counter(chars)["皖"] == 2


def test_non_wan_share_rejects_length_mismatch():
    with pytest.raises(ValueError):
        non_wan_share([0.5, 0.5], ["皖"])
