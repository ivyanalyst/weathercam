import sys
import time

import numpy as np
import pytest

from weathercam.plates.ocr import decode_positions
from weathercam.service.plate_client import PlateClient, PlateReader, PlateResult
from weathercam.service.worker_client import WorkerError

FAKE_WORKER = '''
import base64
import json
import struct
import sys
import time

mode = sys.argv[1]
if mode == "nostart":
    time.sleep(30)
ready = {"ready": True, "model": "fake", "detector_sha256": "det", "ocr_sha256": "ocr"}
if mode == "nohash":
    del ready["ocr_sha256"]
print(json.dumps(ready), flush=True)


def logits_for(size):
    return [[((p * 7 + c * 3 + size) % 11) / 2.0 for c in range(35)] for p in range(7)]


for line in sys.stdin:
    request = json.loads(line)
    if mode == "exit":
        sys.exit(0)
    if mode == "hang":
        time.sleep(30)
    if mode == "badline":
        print("not json", flush=True)
        continue
    if mode == "error":
        print(json.dumps({"id": request["id"], "ok": False, "error": "boom"}), flush=True)
        continue
    blobs = [base64.b64decode(b) for b in request["images"]]
    found = [not b.startswith(b"none") for b in blobs]
    results = []
    for blob, hit in zip(blobs, found):
        size = len(blob)
        box = [float(size), 1.0, float(size) + 10.0, 11.0] if hit else None
        score = size / 10 if hit else None
        results.append({"width": 720, "height": 1160, "box": box, "score": score})
    if mode == "badbox":
        results[0]["box"] = [1.0, 2.0, 3.0]
    if mode == "shortresults":
        results = results[:-1]
    rows = [logits_for(len(b)) for b, hit in zip(blobs, found) if hit]
    flat = [v for plane in rows for row in plane for v in row]
    if mode == "nonfinite":
        flat[0] = float("nan")
    count = len(rows) + 1 if mode == "badshape" else len(rows)
    reply_id = request["id"] + 1 if mode == "wrongid" else request["id"]
    data = base64.b64encode(struct.pack("<%df" % len(flat), *flat)).decode()
    reply = {"id": reply_id, "ok": True, "results": results, "shape": [count, 7, 35], "data": data}
    print(json.dumps(reply), flush=True)
'''


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER)
    return str(path)


def make(script, mode, timeout=5.0, startup=10.0):
    return PlateClient([sys.executable, script, mode], timeout, startup)


def fake_logits(size):
    rows = [[((p * 7 + c * 3 + size) % 11) / 2.0 for c in range(35)] for p in range(7)]
    return np.array(rows, dtype=np.float32)


def expected_reading(size):
    return decode_positions(fake_logits(size))


def test_ready_line_sets_hashes(script):
    with make(script, "ok") as client:
        assert client.detector_sha256 == "det"
        assert client.ocr_sha256 == "ocr"


def test_read_plates_returns_box_score_and_reading(script):
    with make(script, "ok") as client:
        first, second = client.read_plates([b"abc", b"hello"])
    assert (first.width, first.height) == (720, 1160)
    assert first.box == (3.0, 1.0, 13.0, 11.0)
    assert first.score == pytest.approx(0.3)
    assert first.reading == expected_reading(3)
    assert second.box == (5.0, 1.0, 15.0, 11.0)
    assert second.score == pytest.approx(0.5)
    assert second.reading == expected_reading(5)


def test_undetected_image_in_the_middle_keeps_alignment(script):
    with make(script, "ok") as client:
        results = client.read_plates([b"abc", b"none1", b"hello"])
    assert results[1] == PlateResult(720, 1160, None, None, None)
    assert results[0].reading == expected_reading(3)
    assert results[2].reading == expected_reading(5)


def test_no_detection_at_all(script):
    with make(script, "ok") as client:
        results = client.read_plates([b"none", b"none2"])
    assert results == [PlateResult(720, 1160, None, None, None)] * 2


def test_repeated_calls_keep_working(script):
    with make(script, "ok") as client:
        first = client.read_plates([b"abc"])
        second = client.read_plates([b"abc"])
    assert first == second


def test_empty_input_raises_value_error(script):
    with make(script, "ok") as client:
        with pytest.raises(ValueError):
            client.read_plates([])


def test_wrong_input_types_raise_value_error(script):
    with make(script, "ok") as client:
        with pytest.raises(ValueError):
            client.read_plates(["abc"])


def test_error_reply_raises_with_message(script):
    with make(script, "error") as client:
        with pytest.raises(WorkerError, match="boom"):
            client.read_plates([b"abc"])


def test_wrong_id_raises(script):
    with make(script, "wrongid") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_bad_json_line_raises(script):
    with make(script, "badline") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_worker_exit_raises(script):
    with make(script, "exit") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_reply_timeout_raises(script):
    start = time.monotonic()
    with make(script, "hang", timeout=0.5) as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])
    assert time.monotonic() - start < 10.0


def test_startup_timeout_raises(script):
    with pytest.raises(WorkerError):
        make(script, "nostart", startup=0.5)


def test_missing_ready_hash_raises(script):
    with pytest.raises(WorkerError):
        make(script, "nohash")


def test_result_count_mismatch_raises(script):
    with make(script, "shortresults") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_bad_shape_raises(script):
    with make(script, "badshape") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_non_finite_values_raise(script):
    with make(script, "nonfinite") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_bad_box_raises(script):
    with make(script, "badbox") as client:
        with pytest.raises(WorkerError):
            client.read_plates([b"abc"])


def test_close_twice_and_call_after_close(script):
    client = make(script, "ok")
    client.close()
    client.close()
    with pytest.raises(WorkerError):
        client.read_plates([b"abc"])


def test_missing_command_raises():
    with pytest.raises(WorkerError):
        PlateClient(["/nonexistent/worker_cmd"], 1.0, 1.0)


def test_invalid_timeouts_raise_value_error(script):
    with pytest.raises(ValueError):
        make(script, "ok", timeout=0)
    with pytest.raises(ValueError):
        make(script, "ok", startup=0)


def test_client_satisfies_reader_protocol(script):
    with make(script, "ok") as client:
        assert isinstance(client, PlateReader)
