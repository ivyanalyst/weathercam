import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from weathercam.core.boxes import clip_box
from weathercam.core.types import Box, Detection
from weathercam.detection.records import ImagePrediction, PredictionSet, save_predictions

WIDTH, HEIGHT = 720, 1160


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part", choices=("train", "val", "test"), default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"EXISTS: {out}")
    src = Path(args.src)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["part"] == args.part]
    if args.limit:
        items = items[: args.limit]
    model = YOLO(args.weights)
    if list(model.names.values()) != ["plate"]:
        sys.exit(f"unexpected class names: {model.names}")
    device = int(args.device) if args.device.isdigit() else args.device
    if args.batch < 1:
        sys.exit("--batch must be at least 1")
    paths = [str(src / item["file"]) for item in items]

    def iter_results():
        for start in range(0, len(paths), args.batch):
            yield from model.predict(
                paths[start : start + args.batch],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                verbose=False,
            )

    results = iter_results()
    predictions = []
    empty = 0
    for item, result in zip(items, results):
        if tuple(result.orig_shape) != (HEIGHT, WIDTH):
            raise ValueError(f"{item['file']}: unexpected shape {tuple(result.orig_shape)}")
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        detections = [
            Detection(clip_box(Box(*b), WIDTH, HEIGHT), 0, min(max(s, 0.0), 1.0))
            for b, s in zip(boxes, scores)
        ]
        empty += int(not detections)
        predictions.append(ImagePrediction(name=item["file"], width=WIDTH, height=HEIGHT, detections=detections))
        if len(predictions) % 1000 == 0:
            print("predicted", len(predictions), flush=True)
    if len(predictions) != len(items):
        sys.exit(f"got {len(predictions)} results for {len(items)} images")
    settings = {
        "part": args.part,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": str(args.device),
        "images": len(items),
        "images_without_detection": empty,
        "weights_sha256": file_sha256(args.weights),
        "split_sha256": manifest["split_sha256"],
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(out, PredictionSet(model=Path(args.weights).name, settings=settings, images=predictions))
    print("images", len(items), "| without any detection", empty)
    print("wrote", out, "| sha256", file_sha256(out))


if __name__ == "__main__":
    main()
