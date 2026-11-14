import base64
import json
import queue
import subprocess
import threading
from typing import Protocol, runtime_checkable

import numpy as np

STOP_SECONDS = 5.0
JOIN_SECONDS = 2.0


class WorkerError(RuntimeError):
    pass


@runtime_checkable
class Encoder(Protocol):
    def encode_text(self, texts) -> np.ndarray: ...

    def encode_images(self, blobs) -> np.ndarray: ...


def _check_texts(texts):
    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must be a sequence of strings")
    items = list(texts)
    if not items:
        raise ValueError("texts must not be empty")
    if not all(isinstance(t, str) for t in items):
        raise ValueError("every text must be a string")
    return items


def _check_blobs(blobs):
    if isinstance(blobs, (str, bytes, bytearray)):
        raise ValueError("blobs must be a sequence of byte strings")
    items = list(blobs)
    if not items:
        raise ValueError("blobs must not be empty")
    if not all(isinstance(b, (bytes, bytearray)) for b in items):
        raise ValueError("every image must be bytes")
    return items


class WorkerClient:
    def __init__(self, command, timeout_seconds, startup_timeout_seconds):
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        if startup_timeout_seconds <= 0:
            raise ValueError(
                f"startup_timeout_seconds must be > 0, got {startup_timeout_seconds}"
            )
        self._timeout = float(timeout_seconds)
        self._lock = threading.Lock()
        self._lines = queue.Queue()
        self._next_id = 0
        self._closed = False
        try:
            self._process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as error:
            raise WorkerError(f"cannot start worker: {error}") from error
        self._reader = threading.Thread(
            target=self._read_stdout, name="worker-reader", daemon=True
        )
        self._reader.start()
        try:
            ready = self._parse(self._next_line(float(startup_timeout_seconds)))
            self._check_ready(ready)
        except WorkerError:
            self._shutdown()
            raise
        self._model = str(ready.get("model", ""))
        self._pretrained = str(ready.get("pretrained", ""))

    def _check_ready(self, ready):
        dim = ready.get("dim")
        if (
            ready.get("ready") is not True
            or isinstance(dim, bool)
            or not isinstance(dim, int)
            or dim < 1
        ):
            raise WorkerError("invalid ready line")
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def model(self):
        return self._model

    @property
    def pretrained(self):
        return self._pretrained

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        with self._lock:
            self._shutdown()

    def _shutdown(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass
        process.terminate()
        try:
            process.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        self._reader.join(timeout=JOIN_SECONDS)
        try:
            process.stdout.close()
        except (OSError, ValueError):
            pass

    def _read_stdout(self):
        try:
            for line in self._process.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def _next_line(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise WorkerError(f"no reply within {timeout} seconds") from None
        if line is None:
            raise WorkerError("worker exited")
        return line

    @staticmethod
    def _parse(line):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise WorkerError(f"reply is not JSON: {error}") from error
        if not isinstance(value, dict):
            raise WorkerError("reply is not an object")
        return value

    def _exchange(self, message):
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError) as error:
            raise WorkerError(f"cannot write to worker: {error}") from error
        reply = self._parse(self._next_line(self._timeout))
        if reply.get("id") != message["id"]:
            raise WorkerError("reply id does not match request")
        return reply

    def _decode(self, reply, count):
        try:
            if list(reply["shape"]) != [count, self._dim]:
                raise WorkerError(f"unexpected shape {reply['shape']}")
            raw = base64.b64decode(reply["data"], validate=True)
            if len(raw) != count * self._dim * 4:
                raise WorkerError("data length does not match shape")
            flat = np.frombuffer(raw, dtype="<f4")
            array = flat.reshape(count, self._dim).astype(np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerError(f"malformed reply: {error}") from error
        if not np.all(np.isfinite(array)):
            raise WorkerError("reply contains non-finite values")
        return array

    def _request(self, op, field, values):
        with self._lock:
            if self._closed:
                raise WorkerError("client is closed")
            self._next_id += 1
            message = {"id": self._next_id, "op": op, field: values}
            try:
                reply = self._exchange(message)
            except WorkerError:
                self._shutdown()
                raise
            if reply.get("ok") is not True:
                raise WorkerError(f"worker reported an error: {reply.get('error')}")
            try:
                return self._decode(reply, len(values))
            except WorkerError:
                self._shutdown()
                raise

    def encode_text(self, texts):
        return self._request("text", "texts", _check_texts(texts))

    def encode_images(self, blobs):
        items = _check_blobs(blobs)
        encoded = [base64.b64encode(bytes(b)).decode("ascii") for b in items]
        return self._request("image", "images", encoded)
