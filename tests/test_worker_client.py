import subprocess
import sys
import time

import numpy as np
import pytest

from weathercam.service import worker_client
from weathercam.service.worker_client import Encoder, WorkerClient, WorkerError

FAKE_WORKER = '''
import base64
import json
import struct
import sys
import time

mode = sys.argv[1]
if mode == "nostart":
    time.sleep(30)
print(json.dumps({"ready": True, "model": "fake", "pretrained": "none", "dim": 4}), flush=True)
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
    if request["op"] == "text":
        sizes = [len(t) for t in request["texts"]]
    else:
        sizes = [len(base64.b64decode(b)) for b in request["images"]]
    rows = [[float(s), float(i), 0.0, 1.0] for i, s in enumerate(sizes)]
    if mode == "nonfinite":
        rows[0][0] = float("nan")
    flat = [v for row in rows for v in row]
    count = len(rows) + 1 if mode == "badshape" else len(rows)
    reply_id = request["id"] + 1 if mode == "wrongid" else request["id"]
    data = base64.b64encode(struct.pack("<%df" % len(flat), *flat)).decode()
    print(json.dumps({"id": reply_id, "ok": True, "shape": [count, 4], "data": data}), flush=True)
'''


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER)
    return str(path)


def make(script, mode, timeout=5.0, startup=10.0):
    return WorkerClient([sys.executable, script, mode], timeout, startup)


def test_ready_line_sets_model_and_dim(script):
    with make(script, "ok") as client:
        assert client.dim == 4
        assert client.model == "fake"


def test_encode_text_returns_float32_rows(script):
    with make(script, "ok") as client:
        result = client.encode_text(["ab", "cde"])
    assert result.dtype == np.float32
    assert result.shape == (2, 4)
    assert result.tolist() == [[2.0, 0.0, 0.0, 1.0], [3.0, 1.0, 0.0, 1.0]]


def test_encode_images_sends_bytes(script):
    with make(script, "ok") as client:
        result = client.encode_images([b"xyz", b"12345"])
    assert result.tolist() == [[3.0, 0.0, 0.0, 1.0], [5.0, 1.0, 0.0, 1.0]]


def test_repeated_calls_keep_working(script):
    with make(script, "ok") as client:
        first = client.encode_text(["a"])
        second = client.encode_text(["a"])
    assert np.array_equal(first, second)


def test_empty_inputs_raise_value_error(script):
    with make(script, "ok") as client:
        with pytest.raises(ValueError):
            client.encode_text([])
        with pytest.raises(ValueError):
            client.encode_images([])


def test_wrong_input_types_raise_value_error(script):
    with make(script, "ok") as client:
        with pytest.raises(ValueError):
            client.encode_text([b"x"])
        with pytest.raises(ValueError):
            client.encode_images(["x"])


def test_error_reply_raises_with_message(script):
    with make(script, "error") as client:
        with pytest.raises(WorkerError, match="boom"):
            client.encode_text(["a"])


def test_wrong_id_raises(script):
    with make(script, "wrongid") as client:
        with pytest.raises(WorkerError):
            client.encode_text(["a"])


def test_bad_json_line_raises(script):
    with make(script, "badline") as client:
        with pytest.raises(WorkerError):
            client.encode_text(["a"])


def test_worker_exit_raises(script):
    with make(script, "exit") as client:
        with pytest.raises(WorkerError):
            client.encode_text(["a"])


def test_reply_timeout_raises(script):
    start = time.monotonic()
    with make(script, "hang", timeout=0.5) as client:
        with pytest.raises(WorkerError):
            client.encode_text(["a"])
    assert time.monotonic() - start < 10.0


def test_startup_timeout_raises(script):
    with pytest.raises(WorkerError):
        make(script, "nostart", startup=0.5)


def test_interrupt_during_startup_stops_the_worker(script, monkeypatch):
    started = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    def interrupted(self, timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_client.subprocess, "Popen", spy)
    monkeypatch.setattr(WorkerClient, "_next_line", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            make(script, "nostart")
        assert len(started) == 1
        assert started[0].poll() is not None
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_bad_shape_raises(script):
    with make(script, "badshape") as client:
        with pytest.raises(WorkerError):
            client.encode_text(["a"])


def test_non_finite_values_raise(script):
    with make(script, "nonfinite") as client:
        with pytest.raises(WorkerError):
            client.encode_text(["a"])


def test_close_twice_and_call_after_close(script):
    client = make(script, "ok")
    client.close()
    client.close()
    with pytest.raises(WorkerError):
        client.encode_text(["a"])


def test_missing_command_raises():
    with pytest.raises(WorkerError):
        WorkerClient(["/nonexistent/worker_cmd"], 1.0, 1.0)


def test_invalid_timeouts_raise_value_error(script):
    with pytest.raises(ValueError):
        make(script, "ok", timeout=0)
    with pytest.raises(ValueError):
        make(script, "ok", startup=0)


def test_client_satisfies_encoder_protocol(script):
    with make(script, "ok") as client:
        assert isinstance(client, Encoder)
