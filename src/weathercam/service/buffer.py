import threading
from collections import deque


class StreamBuffer:
    def __init__(self, max_items, max_bytes):
        if max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._items = deque()
        self._bytes = 0
        self._pushed = 0
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def pushed(self):
        with self._lock:
            return self._pushed

    @property
    def dropped(self):
        with self._lock:
            return self._dropped

    @property
    def bytes_held(self):
        with self._lock:
            return self._bytes

    def __len__(self):
        with self._lock:
            return len(self._items)

    def push(self, item, size):
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        if size > self._max_bytes:
            raise ValueError(f"size {size} exceeds max_bytes {self._max_bytes}")
        with self._lock:
            dropped = 0
            while self._items and (
                len(self._items) + 1 > self._max_items
                or self._bytes + size > self._max_bytes
            ):
                _, old = self._items.popleft()
                self._bytes -= old
                dropped += 1
            self._items.append((item, size))
            self._bytes += size
            self._pushed += 1
            self._dropped += dropped
            return dropped

    def pop(self):
        with self._lock:
            if not self._items:
                return None
            item, size = self._items.popleft()
            self._bytes -= size
            return item
