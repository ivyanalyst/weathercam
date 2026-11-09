import math

import numpy as np
import pytest

from weathercam.data.ccpd import ADS, ALPHABETS, PROVINCES
from weathercam.plates.ocr import (
    NUM_CLASSES,
    POSITION_CLASSES,
    PlateReading,
    decode_batch,
    decode_positions,
    read_all,
)

INDICES = [0, 0, 22, 27, 27, 33, 16]


def one_hot(indices, value=10.0):
    logits = np.zeros((7, NUM_CLASSES))
    for position, index in enumerate(indices):
        logits[position, index] = value
    return logits


def test_position_classes():
    assert POSITION_CLASSES == (len(PROVINCES), len(ALPHABETS)) + (len(ADS),) * 5
    assert POSITION_CLASSES == (34, 25, 35, 35, 35, 35, 35)
    assert NUM_CLASSES == 35


def test_decode_positions_reads_plate():
    reading = decode_positions(one_hot(INDICES))
    assert reading.text == "皖AY339S"
    assert len(reading.positions) == 7


def test_decode_positions_confidence_uses_only_valid_classes():
    reading = decode_positions(one_hot(INDICES))
    e = math.exp(10.0)
    assert reading.positions[0] == pytest.approx(e / (e + 33))
    assert reading.positions[1] == pytest.approx(e / (e + 24))
    assert reading.positions[2] == pytest.approx(e / (e + 34))


def test_decode_positions_masks_invalid_province_class():
    logits = np.zeros((7, NUM_CLASSES))
    logits[0, 34] = 100.0
    logits[0, 5] = 1.0
    reading = decode_positions(logits)
    assert reading.text[0] == PROVINCES[5]
    assert reading.positions[0] == pytest.approx(math.e / (math.e + 33))


def test_decode_positions_masks_invalid_letter_class():
    logits = np.zeros((7, NUM_CLASSES))
    logits[1, 30] = 50.0
    logits[1, 3] = 1.0
    assert decode_positions(logits).text[1] == ALPHABETS[3]


def test_decode_positions_confidence_is_minimum():
    reading = decode_positions(one_hot(INDICES))
    assert reading.confidence == min(reading.positions)


def test_decode_positions_large_logits_do_not_overflow():
    reading = decode_positions(one_hot(INDICES, value=1000.0))
    assert reading.text == "皖AY339S"
    assert reading.confidence == pytest.approx(1.0)


def test_decode_positions_accepts_float32():
    assert decode_positions(one_hot(INDICES).astype(np.float32)).text == "皖AY339S"


@pytest.mark.parametrize("shape", [(6, NUM_CLASSES), (7, NUM_CLASSES - 1), (1, 7, NUM_CLASSES), (NUM_CLASSES,)])
def test_decode_positions_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        decode_positions(np.zeros(shape))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_decode_positions_rejects_non_finite(bad):
    logits = one_hot(INDICES)
    logits[3, 4] = bad
    with pytest.raises(ValueError):
        decode_positions(logits)


def test_decode_positions_rejects_non_array():
    with pytest.raises(ValueError):
        decode_positions([[0.0] * NUM_CLASSES] * 7)


def test_decode_batch_matches_single():
    batch = np.stack([one_hot(INDICES), one_hot([12, 1, 24, 24, 24, 24, 24])])
    readings = decode_batch(batch)
    assert readings == [decode_positions(batch[0]), decode_positions(batch[1])]
    assert readings[1].text == "京B00000"


def test_decode_batch_empty():
    assert decode_batch(np.zeros((0, 7, NUM_CLASSES))) == []


@pytest.mark.parametrize("shape", [(7, NUM_CLASSES), (2, 7, NUM_CLASSES - 1), (2, 6, NUM_CLASSES)])
def test_decode_batch_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        decode_batch(np.zeros(shape))


class Recorder:
    def __init__(self):
        self.sizes = []

    def read(self, crops):
        self.sizes.append(len(crops))
        return [PlateReading(str(crop), 1.0, ()) for crop in crops]


class Short:
    def read(self, crops):
        return []


class Wrong:
    def read(self, crops):
        return ["abc" for _ in crops]


def test_read_all_keeps_order_across_batches():
    reader = Recorder()
    readings = read_all(reader, [0, 1, 2, 3, 4], batch_size=2)
    assert [r.text for r in readings] == ["0", "1", "2", "3", "4"]
    assert reader.sizes == [2, 2, 1]


def test_read_all_empty_does_not_call_reader():
    reader = Recorder()
    assert read_all(reader, [], batch_size=4) == []
    assert reader.sizes == []


@pytest.mark.parametrize("batch_size", [0, -3])
def test_read_all_rejects_bad_batch_size(batch_size):
    with pytest.raises(ValueError):
        read_all(Recorder(), [0], batch_size=batch_size)


def test_read_all_wrong_count_raises():
    with pytest.raises(ValueError):
        read_all(Short(), [0, 1])


def test_read_all_wrong_type_raises():
    with pytest.raises(ValueError):
        read_all(Wrong(), [0])
