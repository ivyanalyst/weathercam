import argparse
import dataclasses
import json
import resource
import time
from pathlib import Path

import numpy as np

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.data.imagecodec import decode_rgb
from weathercam.data.pipeline import BatchLoader
from weathercam.data.split import group_split
from weathercam.data.veri import VeriReader, named_items, read_labels, read_names
from weathercam.retrieval.evaluate import (
    cross_vehicle_scores,
    group_scores,
    overall_score,
    query_scores,
)
from weathercam.retrieval.features import color_histograms
from weathercam.retrieval.index import BruteForceIndex

BINS = 8
CROSS_PROTOCOL = (
    "colour and type only: the gallery excludes every image of the query vehicle; "
    "relevant means the same attribute on a different vehicle"
)
PROTOCOL = (
    "within-split leave-one-out: every image is a query, the gallery is the whole split "
    "minus the query itself; not the VeRi standard protocol"
)


def select_rows(source, split, seed):
    colors, types = read_names(source)
    if split == "test":
        return "test", list(named_items(read_labels(source, "test"), colors, types))
    train = named_items(read_labels(source, "train"), colors, types)
    groups = [row.vehicle_id for row in train]
    classes = [(row.color, row.type) for row in train]
    _, validation = group_split(groups, classes, (0.8, 0.2), seed)
    return "train", [train[i] for i in validation]


def thin(rows, limit):
    if limit is None or limit >= len(rows):
        return rows
    stride = len(rows) // limit
    return rows[::stride][:limit]


def extract(source, split, rows, batch_size, guard):
    names = [row.image_name for row in rows]
    blocks = []
    with VeriReader(source) as reader:

        def load(name):
            return decode_rgb(reader.read_image(split, name))

        with BatchLoader(load, names, batch_size, queue_depth=2, guard=guard) as loader:
            for _, images in loader:
                blocks.append(color_histograms(images, BINS))
    return np.concatenate(blocks)


def evaluate(index, features, rows, guard):
    vehicles = [row.vehicle_id for row in rows]
    colors = [row.color for row in rows]
    types = [row.type for row in rows]
    kinds = {
        "identity": (vehicles, colors, "color", False),
        "color": (colors, colors, "color", False),
        "type": (types, types, "type", False),
        "color_cross_vehicle": (colors, colors, "color", True),
        "type_cross_vehicle": (types, types, "type", True),
    }
    result = {}
    for kind, (labels, groups, grouped_by, cross) in kinds.items():
        start = time.perf_counter()
        if cross:
            scores = cross_vehicle_scores(index, features, labels, vehicles)
        else:
            scores = query_scores(index, features, labels)
        table = group_scores(scores, groups, vehicles)
        result[kind] = {
            "grouped_by": grouped_by,
            "overall": dataclasses.asdict(overall_score(scores)),
            "groups": {name: dataclasses.asdict(table[name]) for name in sorted(table)},
            "seconds": time.perf_counter() - start,
        }
        guard.check()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    config = load_config()
    source = config.paths.veri_zip
    guard = MemoryGuard(config.memory_limit_mb)
    folder, rows = select_rows(source, args.split, config.seed)
    total = len(rows)
    rows = thin(rows, args.limit)
    start = time.perf_counter()
    features = extract(source, folder, rows, args.batch_size, guard)
    extract_seconds = time.perf_counter() - start
    index = BruteForceIndex(features.shape[1], chunk_size=256)
    index.add(features)
    guard.check()
    results = evaluate(index, features, rows, guard)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "protocol": PROTOCOL,
        "cross_vehicle_protocol": CROSS_PROTOCOL,
        "features": "joint RGB colour histogram, %d bins per channel, %d dimensions" % (BINS, features.shape[1]),
        "split": args.split,
        "slice": args.limit,
        "images": len(rows),
        "images_in_split": total,
        "vehicles": len({row.vehicle_id for row in rows}),
        "seed": config.seed,
        "batch_size": args.batch_size,
        "extract_seconds": extract_seconds,
        "peak_rss_mb": peak,
        "results": results,
    }
    out_dir = Path(config.paths.outputs_dir) / "retrieval_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.split if args.limit is None else "%s_slice%d" % (args.split, args.limit)
    path = out_dir / (name + ".json")
    path.write_text(json.dumps(payload, indent=2))
    print("images %d of %d, vehicles %d" % (len(rows), total, payload["vehicles"]))
    print("extract seconds %.1f" % extract_seconds)
    for kind, entry in results.items():
        overall = entry["overall"]
        print(
            "%-8s P@1 %.4f mAP@R %.4f scored %d skipped %d seconds %.1f"
            % (kind, overall["precision_at_1"], overall["map_at_r"], overall["scored"], overall["skipped"], entry["seconds"])
        )
    print("peak rss %.0f MB" % peak)
    print(path)


main()
