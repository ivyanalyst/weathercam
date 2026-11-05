import json

import numpy as np
import pytest

from weathercam.core.memory import rss_mb
from weathercam.detection.latency import (
    LatencyStats,
    measure_latency,
    peak_rss_mb,
    percentile,
)


def timeline(durations):
    times = []
    now = 0.0
    for duration in durations:
        times.append(now)
        now += duration
        times.append(now)
    return iter(times)


@pytest.mark.parametrize(
    "values,q,expected",
    [
        ([1, 2, 3, 4, 5], 50, 3.0),
        ([1, 2, 3, 4, 5], 95, 4.8),
        ([10, 20, 30, 40], 50, 25.0),
        ([10, 20, 30, 40], 95, 38.5),
        ([5, 1, 4, 2, 3], 50, 3.0),
        ([7], 50, 7.0),
        ([7], 95, 7.0),
        ([3, 9], 0, 3.0),
        ([3, 9], 100, 9.0),
        ([3, 9], 25, 4.5),
        (list(range(1, 22)), 95, 20.0),
    ],
)
def test_percentile_values(values, q, expected):
    assert percentile(values, q) == pytest.approx(expected)


def test_percentile_does_not_change_input():
    values = [5, 1, 4, 2, 3]
    percentile(values, 50)
    assert values == [5, 1, 4, 2, 3]


def test_percentile_accepts_tuple():
    assert percentile((1, 2, 3, 4, 5), 50) == pytest.approx(3.0)


@pytest.mark.parametrize("values", [[], ()])
def test_percentile_empty_raises(values):
    with pytest.raises(ValueError):
        percentile(values, 50)


@pytest.mark.parametrize("q", [-1, 100.5, float("nan"), float("inf"), "50", True])
def test_percentile_bad_q_raises(q):
    with pytest.raises(ValueError):
        percentile([1, 2, 3], q)


@pytest.mark.parametrize(
    "values",
    [
        [1, float("nan")],
        [1, float("inf")],
        [1, -float("inf")],
        [1, "a"],
        [True, 2],
    ],
)
def test_percentile_bad_value_raises(values):
    with pytest.raises(ValueError):
        percentile(values, 50)


def test_results_follow_item_order():
    results, _ = measure_latency(lambda x: x * 2, [1, 2, 3, 4], warmup=1)
    assert results == [2, 4, 6, 8]


def test_func_called_once_per_item_including_warmup():
    calls = []
    measure_latency(calls.append, ["a", "b", "c", "d"], warmup=2)
    assert calls == ["a", "b", "c", "d"]


def test_known_timings_give_known_statistics():
    ticks = timeline([4.0, 0.25, 0.5, 0.75])
    _, stats = measure_latency(
        lambda x: x, [1, 2, 3, 4], warmup=1, clock=ticks.__next__
    )
    assert isinstance(stats, LatencyStats)
    assert list(ticks) == []
    assert stats.count == 3
    assert stats.warmup == 1
    assert stats.timings_ms == (250.0, 500.0, 750.0)
    assert stats.mean_ms == pytest.approx(500.0)
    assert stats.median_ms == pytest.approx(500.0)
    assert stats.p95_ms == pytest.approx(725.0)
    assert stats.min_ms == pytest.approx(250.0)
    assert stats.max_ms == pytest.approx(750.0)


def test_zero_warmup_times_every_call():
    ticks = timeline([1.0, 3.0])
    _, stats = measure_latency(lambda x: x, [1, 2], warmup=0, clock=ticks.__next__)
    assert stats.count == 2
    assert stats.timings_ms == (1000.0, 3000.0)
    assert stats.median_ms == pytest.approx(2000.0)
    assert stats.mean_ms == pytest.approx(2000.0)


def test_generator_input_is_accepted():
    results, stats = measure_latency(
        lambda x: x, (i for i in range(3)), warmup=0
    )
    assert results == [0, 1, 2]
    assert stats.count == 3


def test_func_exception_propagates():
    def boom(item):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        measure_latency(boom, [1, 2], warmup=0)


def test_backward_clock_raises():
    ticks = iter([1.0, 0.5])
    with pytest.raises(ValueError):
        measure_latency(lambda x: x, [1], warmup=0, clock=ticks.__next__)


@pytest.mark.parametrize("warmup", [-1, True, 1.5, "1"])
def test_bad_warmup_raises(warmup):
    with pytest.raises(ValueError):
        measure_latency(lambda x: x, [1, 2, 3], warmup=warmup)


@pytest.mark.parametrize("items,warmup", [([], 0), ([1], 1), ([1, 2], 2)])
def test_not_enough_items_raises(items, warmup):
    with pytest.raises(ValueError):
        measure_latency(lambda x: x, items, warmup=warmup)


def test_non_callable_raises():
    with pytest.raises(ValueError):
        measure_latency(3, [1, 2], warmup=0)


def test_real_clock_gives_ordered_statistics():
    _, stats = measure_latency(lambda x: sum(range(x)), [1000] * 5, warmup=2)
    assert stats.count == 3
    assert 0 <= stats.min_ms <= stats.median_ms <= stats.p95_ms <= stats.max_ms


def test_as_dict_is_json_ready():
    ticks = timeline([4.0, 0.25, 0.5, 0.75])
    _, stats = measure_latency(
        lambda x: x, [1, 2, 3, 4], warmup=1, clock=ticks.__next__
    )
    data = stats.as_dict()
    assert set(data) == {
        "count",
        "warmup",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "timings_ms",
    }
    assert data["timings_ms"] == [250.0, 500.0, 750.0]
    assert json.loads(json.dumps(data)) == data


def test_stats_are_frozen():
    _, stats = measure_latency(lambda x: x, [1, 2], warmup=0)
    with pytest.raises(AttributeError):
        stats.count = 5


def test_peak_rss_is_positive_float():
    value = peak_rss_mb()
    assert isinstance(value, float)
    assert value > 0


def test_peak_rss_reflects_a_known_allocation():
    before = rss_mb()
    block = np.ones(256 * 1024 * 1024, dtype=np.uint8)
    during = rss_mb()
    peak = peak_rss_mb()
    assert block[0] == 1
    assert during - before >= 200
    assert 0.9 * during <= peak <= 4 * during
    del block
