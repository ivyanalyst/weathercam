import argparse
import json
from pathlib import Path

from weathercam.core.types import Box, Detection
from weathercam.data.dawn import load_annotation
from weathercam.data.splitfile import load_split_paths
from weathercam.detection.classmap import CLASS_NAMES, dawn_class_map
from weathercam.detection.dawn_eval import (
    VEHICLE_LABEL,
    ground_truth_detections,
    score_by_weather,
    score_images,
)
from weathercam.detection.records import load_predictions


def mirror(detection, width):
    box = detection.box
    return Detection(
        Box(width - box.x2, box.y1, width - box.x1, box.y2), detection.label, detection.score
    )


def check_mirror():
    original = Detection(Box(10.0, 20.0, 30.0, 40.0), 2, 1.0)
    flipped = mirror(original, 100.0)
    if flipped.box != Box(70.0, 20.0, 90.0, 40.0):
        raise SystemExit(f"mirror self-check failed: {flipped.box}")
    if mirror(flipped, 100.0) != original:
        raise SystemExit("mirror twice did not return the original")


def label_name(label):
    return "vehicle (any)" if label == VEHICLE_LABEL else CLASS_NAMES[label]


def print_score(title, score):
    print(title, "| images", score.images)
    rows = list(score.classes)
    if score.vehicle is not None:
        rows.append(score.vehicle)
    print("  {:<14}{:>7}{:>9}{:>11}".format("class", "boxes", "AP50", "AP50-95"))
    for row in rows:
        print("  {:<14}{:>7}{:>9.4f}{:>11.4f}".format(label_name(row.label), row.boxes, row.ap50, row.ap50_95))


def car_row(score):
    for row in score.classes:
        if CLASS_NAMES[row.label] == "car":
            return row
    return None


def xml_path(dawn_dir, name):
    weather, file = name.split("/", 1)
    return Path(dawn_dir) / weather / weather / f"{weather}_PASCAL_VOC" / f"{Path(file).stem}.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/detections/yolo11m_640_predictions.json")
    parser.add_argument("--latency", default="outputs/detections/yolo11m_640_latency.json")
    parser.add_argument("--dawn-dir", default="data/DAWN")
    parser.add_argument("--display-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--part", choices=("train", "val", "test"), default=None)
    args = parser.parse_args()
    if (args.split_file is None) != (args.part is None):
        raise SystemExit("--split-file and --part must be given together")

    check_mirror()
    print("mirror self-check passed")

    latency = json.load(open(args.latency))
    stats = latency["latency_end_to_end"]
    print("model", latency["model"], "| ultralytics", latency["ultralytics"], "| torch", latency["torch"])
    print("imgsz", latency["imgsz"], "| threads", latency["torch_threads"], "| images", latency["images"])
    print(
        "end-to-end ms median/p95/max",
        round(stats["median_ms"], 1),
        round(stats["p95_ms"], 1),
        round(stats["max_ms"], 1),
        "| timed",
        stats["count"],
    )
    print(
        "ultralytics inference ms median/p95",
        round(latency["ultralytics_inference_ms_median"], 1),
        round(latency["ultralytics_inference_ms_p95"], 1),
    )
    print("rss start/end/peak mb", round(latency["rss_start_mb"], 1), round(latency["rss_end_mb"], 1), round(latency["peak_rss_mb"], 1))
    print("note: end-to-end latency, 2 threads, one run, first image untimed")

    prediction_set = load_predictions(args.predictions)
    mapping = dawn_class_map()
    wanted = None
    if args.split_file:
        wanted = {}
        for rel in load_split_paths(args.split_file, args.part):
            key = (rel.split("/", 1)[0], Path(rel).stem)
            if key in wanted:
                raise SystemExit(f"duplicate key in split: {key}")
            wanted[key] = rel
    found = set()
    names, weathers, predictions, truths, widths = [], [], [], [], []
    for image in prediction_set.images:
        key = (image.name.split("/", 1)[0], Path(image.name).stem)
        if wanted is not None:
            if key not in wanted:
                continue
            found.add(key)
        annotation = load_annotation(xml_path(args.dawn_dir, image.name))
        if (annotation.width, annotation.height) != (image.width, image.height):
            raise SystemExit(f"size mismatch for {image.name}")
        names.append(image.name)
        weathers.append(image.name.split("/", 1)[0])
        predictions.append(list(image.detections))
        truths.append(ground_truth_detections(annotation, mapping))
        widths.append(float(image.width))
    if wanted is not None:
        missing = sorted(set(wanted) - found)
        if missing:
            raise SystemExit(f"{len(missing)} split images have no prediction, first: {missing[:3]}")
        print("split part", args.part, "|", len(names), "images kept of", len(prediction_set.images))
        print("note: latency lines above come from the full prediction run, not this part")
    tags = prediction_set.settings.get("exif_orientation", {})
    kept = set(names)
    tagged = {name: tag for name, tag in tags.items() if tag == 2 and name in kept}
    index = {name: i for i, name in enumerate(names)}
    raw_truths = truths
    if args.display_view:
        truths = [
            [mirror(d, widths[i]) for d in t] if names[i] in tagged else t
            for i, t in enumerate(truths)
        ]
    print("tagged-2 ground truth:", "mirrored to display view" if args.display_view else "as stored", "|", len(tagged), "images")
    print("images", len(names), "| predicted boxes", sum(len(p) for p in predictions), "| ground-truth boxes", sum(len(t) for t in truths))
    print("empty prediction lists", sum(1 for p in predictions if not p))

    print()
    pooled = score_images(predictions, truths)
    print_score("ALL IMAGES POOLED", pooled)
    for weather, score in score_by_weather(weathers, predictions, truths).items():
        print()
        print_score(weather, score)
    print()
    print("DAWN labels are incomplete or loose in places, so these scores understate detector quality.")
    print("'vehicle (any)' means any vehicle found after class-aware NMS, not car accuracy.")

    chosen = [index[name] for name in tagged]
    print()
    print("EXIF EXPERIMENT: images tagged 2:", len(chosen))
    if chosen:
        sub_preds = [predictions[i] for i in chosen]
        stored = [raw_truths[i] for i in chosen]
        flipped = [[mirror(d, widths[i]) for d in raw_truths[i]] for i in chosen]
        score_stored = score_images(sub_preds, stored)
        score_flipped = score_images(sub_preds, flipped)
        print_score("ground truth as stored", score_stored)
        print_score("ground truth mirrored left-right", score_flipped)
        car_a, car_b = car_row(score_stored), car_row(score_flipped)
        if car_a and car_b:
            print("car AP50 stored vs mirrored", round(car_a.ap50, 4), round(car_b.ap50, 4))
            print("car AP50-95 stored vs mirrored", round(car_a.ap50_95, 4), round(car_b.ap50_95, 4))
        print("one measurement on a few dozen images, not proof")


if __name__ == "__main__":
    main()
