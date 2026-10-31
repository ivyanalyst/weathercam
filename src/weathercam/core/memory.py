from contextlib import contextmanager
from dataclasses import dataclass

import psutil

BYTES_PER_MB = 1024 * 1024


class MemoryLimitError(MemoryError):
    pass


def rss_mb():
    return float(psutil.Process().memory_info().rss) / BYTES_PER_MB


class MemoryGuard:
    def __init__(self, limit_mb):
        if limit_mb <= 0:
            raise ValueError(f"limit_mb must be positive, got {limit_mb}")
        self.limit_mb = limit_mb

    def check(self):
        current = rss_mb()
        if current > self.limit_mb:
            raise MemoryLimitError(
                f"RSS {current:.0f} MB exceeds limit {self.limit_mb} MB"
            )
        return current


@dataclass
class MemoryStats:
    start_mb: float = 0.0
    end_mb: float = 0.0
    delta_mb: float = 0.0


@contextmanager
def track_memory():
    stats = MemoryStats(start_mb=rss_mb())
    try:
        yield stats
    finally:
        stats.end_mb = rss_mb()
        stats.delta_mb = stats.end_mb - stats.start_mb
