import pytest

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard, MemoryLimitError, rss_mb, track_memory

BLOCK_BYTES = 50 * 1024 * 1024
PAGE_BYTES = 4096


def touch_block():
    block = bytearray(BLOCK_BYTES)
    for i in range(0, BLOCK_BYTES, PAGE_BYTES):
        block[i] = 1
    return block


def test_rss_mb_is_positive_float():
    value = rss_mb()
    assert isinstance(value, float)
    assert value > 0


def test_rss_mb_grows_after_allocation():
    before = rss_mb()
    block = touch_block()
    after = rss_mb()
    assert after - before > 30
    del block


def test_guard_passes_under_limit():
    MemoryGuard(limit_mb=rss_mb() + 1000).check()


def test_guard_raises_over_limit():
    with pytest.raises(MemoryLimitError):
        MemoryGuard(limit_mb=1).check()


def test_guard_error_message_names_limit():
    with pytest.raises(MemoryLimitError, match="limit"):
        MemoryGuard(limit_mb=1).check()


@pytest.mark.parametrize("bad_limit", [0, -5])
def test_guard_rejects_nonpositive_limit(bad_limit):
    with pytest.raises(ValueError):
        MemoryGuard(limit_mb=bad_limit)


def test_guard_accepts_config_limit():
    guard = MemoryGuard(limit_mb=load_config().memory_limit_mb)
    guard.check()


def test_track_memory_reports_growth():
    with track_memory() as stats:
        block = touch_block()
    assert stats.delta_mb > 30
    assert stats.end_mb > stats.start_mb
    del block
