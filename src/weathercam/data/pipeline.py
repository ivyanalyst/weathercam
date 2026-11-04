import queue
import threading

THREAD_NAME = "batch-loader"
POLL_SECONDS = 0.05
_DONE = object()


class _Failure:
    def __init__(self, error):
        self.error = error


class BatchLoader:
    def __init__(self, load, keys, batch_size, queue_depth=2, guard=None):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if queue_depth < 1:
            raise ValueError(f"queue_depth must be >= 1, got {queue_depth}")
        self._load = load
        self._keys = keys
        self._batch_size = batch_size
        self._guard = guard
        self._queue = queue.Queue(maxsize=queue_depth)
        self._stop_event = threading.Event()
        self._thread = None
        self._closed = False
        self._finished = False

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self):
        if self._closed:
            raise ValueError("loader is closed")
        if self._thread is not None:
            raise ValueError("loader can only be iterated once")
        self._thread = threading.Thread(target=self._produce, name=THREAD_NAME, daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        if self._closed:
            raise ValueError("loader is closed")
        if self._thread is None:
            raise ValueError("call iter() before next()")
        if self._finished:
            raise StopIteration
        while True:
            try:
                item = self._queue.get(timeout=POLL_SECONDS)
            except queue.Empty:
                if self._closed:
                    raise ValueError("loader is closed")
                if not self._thread.is_alive() and self._queue.empty():
                    self._finished = True
                    raise RuntimeError("producer stopped without finishing")
                continue
            if item is _DONE:
                self._finished = True
                raise StopIteration
            if isinstance(item, _Failure):
                self._finished = True
                raise item.error
            return item

    def _put(self, item) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _emit(self, keys, items) -> bool:
        if self._guard is not None:
            self._guard.check()
        return self._put((keys, items))

    def _produce(self):
        try:
            batch_keys = []
            batch_items = []
            for key in self._keys:
                if self._stop_event.is_set():
                    return
                batch_keys.append(key)
                batch_items.append(self._load(key))
                if len(batch_keys) == self._batch_size:
                    if not self._emit(batch_keys, batch_items):
                        return
                    batch_keys = []
                    batch_items = []
            if batch_keys and not self._emit(batch_keys, batch_items):
                return
            self._put(_DONE)
        except BaseException as error:
            self._put(_Failure(error))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
