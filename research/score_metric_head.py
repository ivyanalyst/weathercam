import argparse
import hashlib

import numpy as np

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.retrieval.evaluate import camera_aware_scores, overall_score, query_scores
from weathercam.retrieval.index import BruteForceIndex
from weathercam.retrieval.metric_head import MetricHead, project


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--head", required=True)
    return p.parse_args()


def score(features, vehicles, cameras):
    index = BruteForceIndex(features.shape[1])
    index.add(features)
    plain = overall_score(query_scores(index, features, vehicles))
    camera = overall_score(camera_aware_scores(index, features, vehicles, cameras))
    return plain, camera


def show(name, plain, camera):
    print(
        name,
        "| plain P@1", round(plain.precision_at_1, 4), "mAP@R", round(plain.map_at_r, 4),
        "| camera-aware P@1", round(camera.precision_at_1, 4), "mAP@R", round(camera.map_at_r, 4),
        "| scored", camera.scored, "skipped", camera.skipped,
    )


def main():
    args = parse_args()
    guard = MemoryGuard(load_config().memory_limit_mb)
    digest = hashlib.sha256(open(args.head, "rb").read()).hexdigest()
    head = np.load(args.head)
    data = np.load(args.embeddings, allow_pickle=True)
    test = data["split"] == "test"
    train = data["split"] == "train"
    vehicles = data["vehicle_ids"][test]
    cameras = data["camera_ids"][test]
    overlap = set(vehicles.tolist()) & set(data["vehicle_ids"][train].tolist())
    print("head", args.head, "sha256", digest)
    print(
        "head lr", float(head["lr"]), "scale", float(head["scale"]), "epochs", int(head["epochs"]),
        "val camera-aware mAP@R", round(float(head["val_camera_map"]), 4),
    )
    print(
        "test images", int(test.sum()), "test vehicles", len(set(vehicles.tolist())),
        "vehicles also in train", len(overlap),
    )
    raw = data["embeddings"][test].astype(np.float64)
    standardised = (raw - head["mean"]) / head["std"]
    weights = head["weights"]
    model = MetricHead(weights, np.empty((0, weights.shape[1])), float(head["scale"]))
    plain, camera = score(raw, vehicles, cameras)
    show("test raw 512-d", plain, camera)
    plain, camera = score(standardised, vehicles, cameras)
    show("test standardised 512-d", plain, camera)
    plain, camera = score(project(model, standardised), vehicles, cameras)
    show("test head 128-d", plain, camera)
    guard.check()


if __name__ == "__main__":
    main()
