import base64
import math
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

from weathercam.plates.ocr import NUM_CLASSES, PlateReading, decode_positions
from weathercam.service.worker_client import WorkerClient, WorkerError, _check_blobs


class PlateResult(NamedTuple):
    width: int
    height: int
    box: tuple | None
    score: float | None
    reading: PlateReading | None


@runtime_checkable
class PlateReader(Protocol):
    def read_plates(self, blobs) -> list: ...


def _dimension(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkerError(f"invalid image size {value!r}")
    return value


def _parse_entry(entry):
    width = _dimension(entry["width"])
    height = _dimension(entry["height"])
    box, score = entry["box"], entry["score"]
    if box is None:
        if score is not None:
            raise WorkerError("score given without a box")
        return width, height, None, None
    if not isinstance(box, list) or len(box) != 4:
        raise WorkerError("box must hold four numbers")
    box = tuple(float(v) for v in box)
    score = float(score)
    if not all(math.isfinite(v) for v in box + (score,)):
        raise WorkerError("box or score is not finite")
    return width, height, box, score


class PlateClient(WorkerClient):
    def _check_ready(self, ready):
        if ready.get("ready") is not True:
            raise WorkerError("invalid ready line")
        for key in ("detector_sha256", "ocr_sha256"):
            value = ready.get(key)
            if not isinstance(value, str) or not value:
                raise WorkerError(f"ready line lacks {key}")
        self._detector_sha256 = ready["detector_sha256"]
        self._ocr_sha256 = ready["ocr_sha256"]

    @property
    def detector_sha256(self):
        return self._detector_sha256

    @property
    def ocr_sha256(self):
        return self._ocr_sha256

    def _build(self, reply, count):
        try:
            results = reply["results"]
            if not isinstance(results, list) or len(results) != count:
                raise WorkerError("results do not match the images")
            parsed = [_parse_entry(entry) for entry in results]
            hits = sum(1 for item in parsed if item[2] is not None)
            if list(reply["shape"]) != [hits, 7, NUM_CLASSES]:
                raise WorkerError(f"unexpected shape {reply['shape']}")
            raw = base64.b64decode(reply["data"], validate=True)
            if len(raw) != hits * 7 * NUM_CLASSES * 4:
                raise WorkerError("data length does not match shape")
            flat = np.frombuffer(raw, dtype="<f4")
            logits = flat.reshape(hits, 7, NUM_CLASSES).astype(np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerError(f"malformed reply: {error}") from error
        if not np.all(np.isfinite(logits)):
            raise WorkerError("reply contains non-finite values")
        output, row = [], 0
        for width, height, box, score in parsed:
            if box is None:
                output.append(PlateResult(width, height, None, None, None))
            else:
                reading = decode_positions(logits[row])
                output.append(PlateResult(width, height, box, score, reading))
                row += 1
        return output

    def read_plates(self, blobs):
        items = _check_blobs(blobs)
        encoded = [base64.b64encode(bytes(b)).decode("ascii") for b in items]
        with self._lock:
            if self._closed:
                raise WorkerError("client is closed")
            self._next_id += 1
            message = {"id": self._next_id, "op": "plate", "images": encoded}
            try:
                reply = self._exchange(message)
            except WorkerError:
                self._shutdown()
                raise
            if reply.get("ok") is not True:
                raise WorkerError(f"worker reported an error: {reply.get('error')}")
            try:
                return self._build(reply, len(items))
            except WorkerError:
                self._shutdown()
                raise
