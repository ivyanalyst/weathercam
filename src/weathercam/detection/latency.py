import math
import resource
import time
from dataclasses import dataclass


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def percentile(values, q):
    if not _is_number(q) or not math.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("q must be a finite number in [0, 100]")
    ordered = list(values)
    if not ordered:
        raise ValueError("values must not be empty")
    for value in ordered:
        if not _is_number(value) or not math.isfinite(value):
            raise ValueError("values must be finite numbers")
    ordered.sort()
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    count: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    timings_ms: tuple

    def as_dict(self):
        return {
            "count": self.count,
            "warmup": self.warmup,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "timings_ms": list(self.timings_ms),
        }


def measure_latency(func, items, warmup=1, clock=time.perf_counter):
    if not callable(func):
        raise ValueError("func must be callable")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    items = list(items)
    if len(items) <= warmup:
        raise ValueError("need more items than warmup calls")
    results = []
    timings = []
    for index, item in enumerate(items):
        start = clock()
        result = func(item)
        end = clock()
        if end < start:
            raise ValueError("clock went backwards")
        results.append(result)
        if index >= warmup:
            timings.append((end - start) * 1000.0)
    stats = LatencyStats(
        count=len(timings),
        warmup=warmup,
        mean_ms=sum(timings) / len(timings),
        median_ms=percentile(timings, 50),
        p95_ms=percentile(timings, 95),
        min_ms=min(timings),
        max_ms=max(timings),
        timings_ms=tuple(timings),
    )
    return results, stats


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
