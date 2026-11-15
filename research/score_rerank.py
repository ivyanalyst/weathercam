import argparse
import hashlib
import sys

import numpy as np

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.retrieval.evaluate import camera_aware_scores, overall_score, query_scores
from weathercam.retrieval.index import BruteForceIndex
from weathercam.retrieval.innersplit import veri_inner_split
from weathercam.retrieval.metric_head import MetricHead, project
from weathercam.retrieval.rerank import expand_queries

NEIGHBOURS = (1, 2, 3, 5, 10, 20, 50)
WEIGHTS = (0.5, 1.0, 2.0, 4.0, 8.0)
VAL_IMAGES = 7828
VAL_VEHICLES = 119


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--mode", choices=("validation", "test"), required=True)
    p.add_argument("--neighbours", type=int)
    p.add_argument("--weight", type=float)
    args = p.parse_args()
    given = args.neighbours is not None or args.weight is not None
    if args.mode == "validation" and given:
        p.error("validation mode takes no --neighbours or --weight")
    if args.mode == "test" and (args.neighbours is None or args.weight is None):
        p.error("test mode needs --neighbours and --weight")
    return args


def validation_rows(data, seed):
    _, rows = veri_inner_split(data, seed)
    vehicles = len(set(data["vehicle_ids"][rows].tolist()))
    if len(rows) != VAL_IMAGES or vehicles != VAL_VEHICLES:
        sys.exit("validation rows are " + str(len(rows)) + " images, " + str(vehicles) + " vehicles")
    return rows


def head_features(data, head, rows):
    raw = data["embeddings"][rows].astype(np.float64)
    standardised = (raw - head["mean"]) / head["std"]
    weights = head["weights"]
    model = MetricHead(weights, np.empty((0, weights.shape[1])), float(head["scale"]))
    return project(model, standardised)


def evaluate(features, vehicles, cameras, neighbours, weight):
    queries = expand_queries(features, features, neighbours, weight, exclude_self=True)
    index = BruteForceIndex(features.shape[1])
    index.add(features)
    plain = overall_score(query_scores(index, queries, vehicles))
    camera = overall_score(camera_aware_scores(index, queries, vehicles, cameras))
    return plain, camera


def show(neighbours, weight, plain, camera):
    print(
        "k", neighbours, "weight", weight,
        "| plain P@1", round(plain.precision_at_1, 4), "mAP@R", round(plain.map_at_r, 4),
        "| camera-aware P@1", round(camera.precision_at_1, 4), "mAP@R", round(camera.map_at_r, 4),
        "| scored", camera.scored, "skipped", camera.skipped,
        flush=True,
    )


def main():
    args = parse_args()
    config = load_config()
    guard = MemoryGuard(config.memory_limit_mb)
    head = np.load(args.head)
    data = np.load(args.embeddings, allow_pickle=True)
    print("head", args.head, "sha256", hashlib.sha256(open(args.head, "rb").read()).hexdigest())
    print("head files", sorted(head.files))
    if args.mode == "validation":
        rows = validation_rows(data, config.seed)
        print("stored val camera-aware mAP@R", round(float(head["val_camera_map"]), 4))
        configs = [(0, 0.0)] + [(k, w) for k in NEIGHBOURS for w in WEIGHTS]
    else:
        rows = np.flatnonzero(data["split"] == "test")
        configs = [(0, 0.0), (args.neighbours, args.weight)]
    vehicles = data["vehicle_ids"][rows]
    cameras = data["camera_ids"][rows]
    features = head_features(data, head, rows)
    print("mode", args.mode, "images", len(rows), "vehicles", len(set(vehicles.tolist())))
    best = None
    for neighbours, weight in configs:
        plain, camera = evaluate(features, vehicles, cameras, neighbours, weight)
        show(neighbours, weight, plain, camera)
        guard.check()
        if neighbours > 0 and (best is None or camera.map_at_r > best[0]):
            best = (camera.map_at_r, neighbours, weight)
    if args.mode == "validation":
        print("best camera-aware mAP@R", round(best[0], 4), "k", best[1], "weight", best[2])


if __name__ == "__main__":
    main()
