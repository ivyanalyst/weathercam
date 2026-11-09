import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from weathercam.attributes.weather import wilson_interval
from weathercam.core.types import Box
from weathercam.detection.records import load_predictions
from weathercam.plates.detector import hit_rate, top1_ious


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.5])
    return p.parse_args()


def row(name, ious, empty, thresholds):
    n = len(ious)
    cells = []
    for t in thresholds:
        rate = hit_rate(ious, t)
        k = round(rate * n)
        low, high = wilson_interval(k, n)
        cells.append("IoU>%.2f %d/%d = %.4f (wilson95 %.4f-%.4f)" % (t, k, n, rate, low, high))
    return "%-15s %s  median IoU %.3f  no detection %d" % (name, "  ".join(cells), statistics.median(ious), empty)


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"EXISTS: {out}")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["part"] == "test"]
    predictions = load_predictions(args.predictions)
    if predictions.settings.get("split_sha256") != manifest["split_sha256"]:
        sys.exit("split_sha256 differs between predictions and manifest")
    by_name = {image.name: image for image in predictions.images}
    wanted = {item["file"] for item in items}
    if set(by_name) != wanted:
        sys.exit(f"name mismatch: {len(wanted - set(by_name))} missing, {len(set(by_name) - wanted)} extra")
    detections = [by_name[item["file"]].detections for item in items]
    ious = top1_ious(detections, [Box(*item["box"]) for item in items])
    groups = collections.defaultdict(list)
    for item, value, found in zip(items, ious, detections):
        groups[item["subset"]].append((value, not found))
    lines = [
        "model %s | weights_sha256 %s" % (predictions.model, predictions.settings.get("weights_sha256")),
        "split_sha256 %s | test images %d | top-1 detection per image, hit if IoU > threshold" % (manifest["split_sha256"], len(items)),
        "pooled row weights subsets by their capped sizes, not by natural frequency",
    ]
    for name in sorted(groups):
        lines.append(row(name, [v for v, _ in groups[name]], sum(e for _, e in groups[name]), args.thresholds))
    lines.append(row("ALL (pooled)", ious, sum(1 for found in detections if not found), args.thresholds))
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
