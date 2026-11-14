import argparse
import base64
import hashlib
import io
import json
import os
import sys

import numpy as np

BYTES_PER_MB = 1024 * 1024
OUT_H, OUT_W = 96, 192
MARGIN = 0.15
IMGSZ = 640
CONF = 0.001
IOU = 0.5
MAX_DET = 10
DETECTOR_SHA256 = "59b2521b4c0b239272373e4085d2ef854eb0a78dc4566dc5cb51e808e971666d"
OCR_SHA256 = "3dd0c9d2110c79a77e05ac521815449a35eb120481e02b242c60a1f2a58f7a57"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--detector-sha256", default=DETECTOR_SHA256)
    ap.add_argument("--ocr-sha256", default=OCR_SHA256)
    ap.add_argument("--det-batch", type=int, default=8)
    ap.add_argument("--ocr-batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-limit-mb", type=float, default=3000)
    return ap.parse_args()


def rss_mb():
    with open("/proc/self/statm") as handle:
        resident = int(handle.read().split()[1])
    return resident * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_MB


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def send(stream, payload):
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def pack(array):
    data = np.ascontiguousarray(array, dtype="<f4")
    encoded = base64.b64encode(data.tobytes()).decode("ascii")
    return {"shape": list(data.shape), "data": encoded}


def require_list(value):
    if not isinstance(value, list) or not value:
        raise ValueError("request needs a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("every entry must be a string")
    return value


def main():
    args = parse_args()
    if args.det_batch < 1 or args.ocr_batch < 1 or args.threads < 1 or args.memory_limit_mb <= 0:
        raise SystemExit("batch sizes, threads and memory limit must be positive")
    if file_sha256(args.detector) != args.detector_sha256:
        raise SystemExit("detector weights hash differs from the expected value")
    if file_sha256(args.ocr) != args.ocr_sha256:
        raise SystemExit("OCR weights hash differs from the expected value")
    real = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from PIL import Image
    from torch import nn
    from ultralytics import YOLO

    from weathercam.core.boxes import clip_box
    from weathercam.core.types import Box
    from weathercam.plates.crop import crop_array, expand_box, resize_rgb
    from weathercam.plates.ocr import NUM_CLASSES

    torch.set_num_threads(args.threads)

    class PlateNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c_in = [], 3
            for c_out in (32, 64, 128, 256):
                for _ in range(2):
                    layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
                    c_in = c_out
                layers.append(nn.MaxPool2d(2))
            self.features = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * (OUT_H // 16) * (OUT_W // 16), 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 7 * NUM_CLASSES),
            )

        def forward(self, x):
            x = (x - 0.5) / 0.5
            return self.head(self.features(x)).view(-1, 7, NUM_CLASSES)

    detector = YOLO(args.detector)
    if list(detector.names.values()) != ["plate"]:
        raise SystemExit("unexpected detector classes: %s" % (detector.names,))
    net = PlateNet()
    net.load_state_dict(torch.load(args.ocr, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    net.eval()

    def detect(arrays):
        found = []
        for start in range(0, len(arrays), args.det_batch):
            chunk = [np.ascontiguousarray(a[..., ::-1]) for a in arrays[start : start + args.det_batch]]
            results = detector.predict(chunk, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET, device="cpu", verbose=False)
            if len(results) != len(chunk):
                raise RuntimeError("detector returned %d results for %d images" % (len(results), len(chunk)))
            for image, result in zip(chunk, results):
                scores = result.boxes.conf.cpu().numpy()
                if len(scores) == 0:
                    found.append((None, None))
                    continue
                best = int(scores.argmax())
                height, width = image.shape[:2]
                raw = [float(v) for v in result.boxes.xyxy[best].cpu().tolist()]
                box = clip_box(Box(*raw), width, height)
                found.append((box, min(max(float(scores[best]), 0.0), 1.0)))
        return found

    def crop_input(rgb, box):
        height, width = rgb.shape[:2]
        try:
            crop = crop_array(rgb, expand_box(box, MARGIN, width, height))
        except ValueError:
            return None
        return resize_rgb(crop.astype(np.float32) / 255.0, OUT_H, OUT_W).transpose(2, 0, 1)

    def ocr_logits(inputs):
        if not inputs:
            return np.zeros((0, 7, NUM_CLASSES), dtype=np.float32)
        parts = []
        with torch.no_grad():
            for start in range(0, len(inputs), args.ocr_batch):
                batch = torch.from_numpy(np.stack(inputs[start : start + args.ocr_batch]))
                parts.append(net(batch).float().cpu().numpy())
        return np.concatenate(parts)

    def read_plates(blobs):
        arrays = []
        for text in blobs:
            raw = base64.b64decode(text, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                arrays.append(np.asarray(image.convert("RGB")))
        entries, inputs = [], []
        for rgb, (box, score) in zip(arrays, detect(arrays)):
            height, width = rgb.shape[:2]
            tensor = None if box is None else crop_input(rgb, box)
            if tensor is None:
                entries.append({"width": width, "height": height, "box": None, "score": None})
            else:
                entries.append({"width": width, "height": height, "box": list(box.as_tuple()), "score": score})
                inputs.append(tensor)
        return entries, ocr_logits(inputs)

    send(real, {"ready": True, "model": "plate-yolo11m+ocr", "detector_sha256": args.detector_sha256, "ocr_sha256": args.ocr_sha256})
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            used = rss_mb()
            if used > args.memory_limit_mb:
                raise MemoryError("rss %.0f MB exceeds limit %.0f MB" % (used, args.memory_limit_mb))
            if request.get("op") != "plate":
                raise ValueError("unknown op")
            entries, logits = read_plates(require_list(request.get("images")))
            reply = {"id": request_id, "ok": True, "results": entries}
            reply.update(pack(logits))
        except Exception as error:
            reply = {"id": request_id, "ok": False, "error": "%s: %s" % (type(error).__name__, error)}
        send(real, reply)


if __name__ == "__main__":
    main()
