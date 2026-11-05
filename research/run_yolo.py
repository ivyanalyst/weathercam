import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import ultralytics
from ultralytics import YOLO

from weathercam.core.memory import MemoryGuard, track_memory
from weathercam.data.dawn import load_annotation
from weathercam.detection.classmap import model_label_map
from weathercam.detection.convert import to_detections
from weathercam.detection.latency import measure_latency, peak_rss_mb, percentile
from weathercam.detection.records import ImagePrediction, PredictionSet, save_predictions

EXIF_ORIENTATION = 0x0112
CONF = 0.01
IOU = 0.5
MAX_DET = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--weights-dir", default="outputs/yolo_weights")
    parser.add_argument("--dawn-dir", default="data/DAWN")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--names-file", default=None)
    parser.add_argument("--memory-limit-mb", type=float, default=3000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--latency-out", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_items(dawn_dir):
    jpgs = sorted(Path(dawn_dir).glob("*/*/*.jpg"))
    items = []
    missing = []
    for jpg in jpgs:
        weather = jpg.parent.parent.name
        xml = jpg.parent / f"{weather}_PASCAL_VOC" / f"{jpg.stem}.xml"
        if not xml.exists():
            missing.append(str(xml))
            continue
        items.append((f"{weather}/{jpg.name}", str(jpg), str(xml)))
    if missing:
        print("missing xml files", len(missing), missing[:3])
        sys.exit(1)
    names = [item[0] for item in items]
    if len(set(names)) != len(names):
        print("duplicate image names")
        sys.exit(1)
    return items


def select(items, limit):
    if limit is None or limit >= len(items):
        return items
    if limit <= 0:
        print("limit must be positive")
        sys.exit(1)
    return [items[i * len(items) // limit] for i in range(limit)]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def main():
    args = parse_args()
    weights = Path(args.weights_dir) / args.model
    if not weights.exists():
        print("weights not found", weights)
        sys.exit(1)
    stem = weights.stem
    out = Path(args.out) if args.out else Path("outputs/detections") / f"{stem}_{args.imgsz}_predictions.json"
    latency_out = Path(args.latency_out) if args.latency_out else Path("outputs/detections") / f"{stem}_{args.imgsz}_latency.json"
    for target in (out, latency_out):
        if target.exists() and not args.overwrite:
            print("EXISTS, use --overwrite", target)
            sys.exit(1)
        target.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    all_items = build_items(args.dawn_dir)
    if args.names_file:
        if args.limit is not None:
            print("use either --limit or --names-file")
            sys.exit(1)
        wanted = [line.strip() for line in Path(args.names_file).read_text().splitlines() if line.strip()]
        by_name = {item[0]: item for item in all_items}
        absent = [name for name in wanted if name not in by_name]
        if absent:
            print("names not found", len(absent), absent[:3])
            sys.exit(1)
        items = [by_name[name] for name in wanted]
    else:
        items = select(all_items, args.limit)
    print("images found", len(all_items), "| selected", len(items))

    model = YOLO(str(weights))
    label_map = model_label_map(model.names)
    print("label_map", label_map)
    print("torch threads", torch.get_num_threads())

    guard = MemoryGuard(args.memory_limit_mb)
    warm = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(warm, imgsz=args.imgsz, conf=CONF, iou=IOU, max_det=MAX_DET, device="cpu", verbose=False)
    print("torch threads after first predict", torch.get_num_threads())
    torch.set_num_threads(args.threads)
    print("torch threads reset to", torch.get_num_threads())

    def process(item):
        name, jpg, xml = item
        annotation = load_annotation(xml)
        with Image.open(jpg) as image:
            orientation = image.getexif().get(EXIF_ORIENTATION)
            width, height = image.size
            rgb = np.asarray(image.convert("RGB"))
        if (annotation.width, annotation.height) != (width, height):
            raise ValueError(f"{name}: xml size {(annotation.width, annotation.height)} != decoded {(width, height)}")
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        threads_before = torch.get_num_threads()
        result = model.predict(
            bgr, imgsz=args.imgsz, conf=CONF, iou=IOU, max_det=MAX_DET, device="cpu", verbose=False
        )[0]
        threads_after = torch.get_num_threads()
        if threads_before != args.threads or threads_after != args.threads:
            raise RuntimeError(f"{name}: torch threads {threads_before} -> {threads_after}, expected {args.threads}")
        if tuple(result.orig_shape) != (height, width):
            raise ValueError(f"{name}: model shape {tuple(result.orig_shape)} != decoded {(height, width)}")
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        scores = result.boxes.conf.cpu().numpy().tolist()
        labels = result.boxes.cls.cpu().numpy().astype(int).tolist()
        detections = to_detections(boxes, scores, labels, label_map, width, height)
        guard.check()
        prediction = ImagePrediction(name=name, width=width, height=height, detections=detections)
        return prediction, orientation, float(result.speed["inference"])

    with track_memory() as memory:
        outputs, stats = measure_latency(process, items, warmup=1)
    peak = peak_rss_mb()

    if len(outputs) != len(items):
        print("result count", len(outputs), "!= items", len(items))
        sys.exit(1)

    predictions = [entry[0] for entry in outputs]
    exif = {}
    for entry in outputs:
        if entry[1] not in (None, 1):
            exif[entry[0].name] = int(entry[1])
    inference = [entry[2] for entry in outputs[stats.warmup:]]

    settings = {
        "imgsz": args.imgsz,
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "input": "PIL decode, uint8 RGB flipped to BGR, EXIF orientation not applied",
        "label_map": {str(key): value for key, value in label_map.items()},
        "exif_orientation": exif,
        "images_total": len(all_items),
        "limit": args.limit,
    }
    save_predictions(out, PredictionSet(model=args.model, settings=settings, images=predictions))

    report = {
        "model": args.model,
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": file_sha256(weights),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "imgsz": args.imgsz,
        "torch_threads": torch.get_num_threads(),
        "images": len(items),
        "latency_end_to_end": stats.as_dict(),
        "latency_note": "decode + BGR flip + predict + convert + size checks per image; first image is the untimed warm-up",
        "ultralytics_inference_ms_median": percentile(inference, 50) if inference else None,
        "ultralytics_inference_ms_p95": percentile(inference, 95) if inference else None,
        "rss_start_mb": memory.start_mb,
        "rss_end_mb": memory.end_mb,
        "rss_delta_mb": memory.delta_mb,
        "peak_rss_mb": peak,
        "memory_limit_mb": args.memory_limit_mb,
    }
    write_json(latency_out, report)

    total = sum(len(p.detections) for p in predictions)
    print("images", len(predictions), "| detections", total, "| tagged exif", len(exif))
    print("end-to-end median ms", round(stats.median_ms, 1), "| p95 ms", round(stats.p95_ms, 1), "| timed", stats.count)
    print("rss start/end/delta mb", round(memory.start_mb, 1), round(memory.end_mb, 1), round(memory.delta_mb, 1))
    print("peak rss mb", round(peak, 1))
    print("wrote", out)
    print("wrote", latency_out)


if __name__ == "__main__":
    main()
