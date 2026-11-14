import threading

import pytest

from weathercam.service.buffer import StreamBuffer


def test_empty_buffer_pop_returns_none():
    buffer = StreamBuffer(3, 100)
    assert buffer.pop() is None
    assert len(buffer) == 0


def test_push_then_pop_returns_item():
    buffer = StreamBuffer(3, 100)
    buffer.push("a", 10)
    assert len(buffer) == 1
    assert buffer.pop() == "a"
    assert len(buffer) == 0


def test_pop_order_is_fifo():
    buffer = StreamBuffer(5, 100)
    for name in ["a", "b", "c"]:
        buffer.push(name, 1)
    assert [buffer.pop(), buffer.pop(), buffer.pop()] == ["a", "b", "c"]


def test_push_returns_zero_when_there_is_room():
    buffer = StreamBuffer(3, 100)
    assert buffer.push("a", 10) == 0
    assert buffer.push("b", 10) == 0


def test_item_cap_drops_oldest():
    buffer = StreamBuffer(2, 100)
    assert buffer.push("a", 1) == 0
    assert buffer.push("b", 1) == 0
    assert buffer.push("c", 1) == 1
    assert buffer.pop() == "b"
    assert buffer.pop() == "c"
    assert buffer.pop() is None


def test_byte_cap_allows_exact_fit_then_drops_oldest():
    buffer = StreamBuffer(10, 10)
    assert buffer.push("a", 5) == 0
    assert buffer.push("b", 5) == 0
    assert buffer.bytes_held == 10
    assert buffer.push("c", 1) == 1
    assert buffer.bytes_held == 6
    assert buffer.pop() == "b"
    assert buffer.pop() == "c"
    assert buffer.pop() is None


def test_byte_cap_can_drop_several_items():
    buffer = StreamBuffer(10, 10)
    for name in ["a", "b", "c"]:
        buffer.push(name, 3)
    assert buffer.push("d", 8) == 3
    assert len(buffer) == 1
    assert buffer.bytes_held == 8
    assert buffer.pop() == "d"


def test_counters_track_pushes_and_drops():
    buffer = StreamBuffer(2, 100)
    for name in ["a", "b", "c"]:
        buffer.push(name, 1)
    assert buffer.pushed == 3
    assert buffer.dropped == 1
    buffer.pop()
    assert buffer.pushed == 3
    assert buffer.dropped == 1


def test_bytes_held_follows_pop():
    buffer = StreamBuffer(5, 100)
    buffer.push("x", 5)
    buffer.push("y", 7)
    assert buffer.bytes_held == 12
    buffer.pop()
    assert buffer.bytes_held == 7
    buffer.pop()
    assert buffer.bytes_held == 0


def test_item_larger_than_byte_cap_raises_and_changes_nothing():
    buffer = StreamBuffer(10, 10)
    buffer.push("a", 3)
    with pytest.raises(ValueError):
        buffer.push("big", 11)
    assert len(buffer) == 1
    assert buffer.pushed == 1
    assert buffer.dropped == 0
    assert buffer.bytes_held == 3


def test_negative_size_raises():
    buffer = StreamBuffer(3, 100)
    with pytest.raises(ValueError):
        buffer.push("a", -1)
    assert len(buffer) == 0


def test_non_positive_limits_raise():
    with pytest.raises(ValueError):
        StreamBuffer(0, 10)
    with pytest.raises(ValueError):
        StreamBuffer(10, 0)


def test_zero_size_items_are_limited_by_item_cap():
    buffer = StreamBuffer(2, 10)
    for name in ["a", "b", "c"]:
        buffer.push(name, 0)
    assert len(buffer) == 2
    assert buffer.dropped == 1
    assert buffer.bytes_held == 0


def test_concurrent_pushes_keep_counts_consistent():
    buffer = StreamBuffer(10, 1000)

    def worker(start):
        for i in range(250):
            buffer.push(start + i, 1)

    threads = [threading.Thread(target=worker, args=(n * 1000,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert buffer.pushed == 1000
    assert len(buffer) == 10
    assert buffer.dropped == 990
    assert buffer.bytes_held == 10
