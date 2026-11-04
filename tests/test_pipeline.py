import threading
import time

import pytest

from weathercam.core.memory import MemoryLimitError
from weathercam.data.pipeline import BatchLoader

THREAD_NAME = "batch-loader"


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def loader_threads():
    return [t for t in threading.enumerate() if t.name == THREAD_NAME and t.is_alive()]


class Counter:
    def __init__(self):
        self.calls = []
        self.threads = set()
        self.lock = threading.Lock()

    def __call__(self, key):
        with self.lock:
            self.calls.append(key)
            self.threads.add(threading.current_thread().name)
        return key * 10


class FakeGuard:
    def __init__(self, fail_on=None):
        self.count = 0
        self.fail_on = fail_on

    def check(self):
        self.count += 1
        if self.count == self.fail_on:
            raise MemoryLimitError("too much")
        return 1.0


def collect(loader):
    return [batch for batch in loader]


def test_yields_all_items_in_order():
    with BatchLoader(Counter(), list(range(10)), 4) as loader:
        batches = collect(loader)
    assert [k for keys, _ in batches for k in keys] == list(range(10))
    assert [i for _, items in batches for i in items] == [k * 10 for k in range(10)]


def test_short_last_batch():
    with BatchLoader(Counter(), list(range(10)), 4) as loader:
        sizes = [len(keys) for keys, _ in loader]
    assert sizes == [4, 4, 2]


def test_exact_multiple_has_no_empty_batch():
    with BatchLoader(Counter(), list(range(8)), 4) as loader:
        sizes = [len(keys) for keys, _ in loader]
    assert sizes == [4, 4]


def test_batch_larger_than_input():
    with BatchLoader(Counter(), list(range(3)), 10) as loader:
        sizes = [len(keys) for keys, _ in loader]
    assert sizes == [3]


def test_empty_keys_yield_nothing():
    with BatchLoader(Counter(), [], 4) as loader:
        assert collect(loader) == []


def test_accepts_generator_keys():
    with BatchLoader(Counter(), (k for k in range(5)), 2) as loader:
        batches = collect(loader)
    assert [k for keys, _ in batches for k in keys] == list(range(5))


def test_batches_pair_keys_with_loaded_items():
    with BatchLoader(Counter(), list(range(6)), 4) as loader:
        batches = collect(loader)
    assert batches[0] == ([0, 1, 2, 3], [0, 10, 20, 30])
    assert isinstance(batches[0][0], list)
    assert isinstance(batches[0][1], list)


def test_load_runs_on_one_named_thread():
    counter = Counter()
    with BatchLoader(counter, list(range(10)), 4) as loader:
        collect(loader)
    assert counter.threads == {THREAD_NAME}


def test_load_called_once_per_key():
    counter = Counter()
    with BatchLoader(counter, list(range(10)), 4) as loader:
        collect(loader)
    assert sorted(counter.calls) == list(range(10))


def test_thread_finishes_after_exhaustion():
    with BatchLoader(Counter(), list(range(10)), 4) as loader:
        collect(loader)
        assert wait_until(lambda: not loader_threads())


def test_producer_blocks_when_queue_is_full():
    counter = Counter()
    with BatchLoader(counter, list(range(100)), 4, queue_depth=2) as loader:
        iter(loader)
        assert wait_until(lambda: len(counter.calls) >= 12)
        time.sleep(0.1)
        assert len(counter.calls) == 12


def test_bound_slides_with_consumption():
    counter = Counter()
    with BatchLoader(counter, list(range(100)), 4, queue_depth=2) as loader:
        iterator = iter(loader)
        assert wait_until(lambda: len(counter.calls) >= 12)
        next(iterator)
        assert wait_until(lambda: len(counter.calls) >= 16)
        time.sleep(0.1)
        assert len(counter.calls) == 16


def test_close_stops_producer_early():
    counter = Counter()
    with BatchLoader(counter, list(range(1000)), 4, queue_depth=2) as loader:
        next(iter(loader))
    assert wait_until(lambda: not loader_threads())
    seen = len(counter.calls)
    time.sleep(0.05)
    assert len(counter.calls) == seen
    assert seen <= 20


def test_close_is_idempotent():
    loader = BatchLoader(Counter(), list(range(10)), 4)
    assert not loader.closed
    loader.close()
    loader.close()
    assert loader.closed


def test_iter_after_close_raises():
    loader = BatchLoader(Counter(), list(range(10)), 4)
    loader.close()
    with pytest.raises(ValueError):
        iter(loader)


def test_second_iteration_raises():
    with BatchLoader(Counter(), list(range(10)), 4) as loader:
        collect(loader)
        with pytest.raises(ValueError):
            iter(loader)


def test_exception_in_load_reaches_consumer():
    def load(key):
        if key == 5:
            raise RuntimeError("bad item")
        return key

    with BatchLoader(load, list(range(10)), 4) as loader:
        iterator = iter(loader)
        keys, items = next(iterator)
        assert keys == [0, 1, 2, 3]
        with pytest.raises(RuntimeError, match="bad item"):
            next(iterator)
        assert wait_until(lambda: not loader_threads())


def test_exception_in_first_batch_reaches_consumer():
    def load(key):
        if key == 1:
            raise RuntimeError("bad first")
        return key

    with BatchLoader(load, list(range(10)), 4) as loader:
        with pytest.raises(RuntimeError, match="bad first"):
            next(iter(loader))


def test_guard_checked_once_per_batch():
    guard = FakeGuard()
    with BatchLoader(Counter(), list(range(10)), 4, guard=guard) as loader:
        collect(loader)
    assert guard.count == 3


def test_guard_error_reaches_consumer():
    guard = FakeGuard(fail_on=2)
    with BatchLoader(Counter(), list(range(10)), 4, guard=guard) as loader:
        iterator = iter(loader)
        keys, _ = next(iterator)
        assert keys == [0, 1, 2, 3]
        with pytest.raises(MemoryLimitError):
            next(iterator)


@pytest.mark.parametrize("batch_size", [0, -3])
def test_invalid_batch_size_raises(batch_size):
    with pytest.raises(ValueError):
        BatchLoader(Counter(), [1, 2], batch_size)


@pytest.mark.parametrize("queue_depth", [0, -1])
def test_invalid_queue_depth_raises(queue_depth):
    with pytest.raises(ValueError):
        BatchLoader(Counter(), [1, 2], 2, queue_depth=queue_depth)


def test_exception_in_with_body_closes_loader():
    counter = Counter()
    with pytest.raises(KeyError):
        with BatchLoader(counter, list(range(1000)), 4) as loader:
            next(iter(loader))
            raise KeyError("stop")
    assert loader.closed
    assert wait_until(lambda: not loader_threads())
