import argparse
import hashlib
import sys

import numpy as np

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.data.split import group_split
from weathercam.retrieval.evaluate import camera_aware_scores, overall_score, query_scores
from weathercam.retrieval.index import BruteForceIndex
from weathercam.retrieval.metric_head import MetricHead, fit, project
from weathercam.retrieval.mining import k_center_greedy

BUDGETS = (1000, 3000, 6000, 12000)
STARTS = (0, 1, 2)
SEEDS = (0, 1, 2)
FIT_IMAGES = 29918
VAL_IMAGES = 7828
VAL_VEHICLES = 119


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--head", required=True)
    return p.parse_args()


def parts(data, seed):
    train_rows = np.flatnonzero(data["split"] == "train")
    vehicles = data["vehicle_ids"][train_rows]
    classes = list(
        zip(data["color_ids"][train_rows].tolist(), data["type_ids"][train_rows].tolist())
    )
    fit_part, val_part = group_split(vehicles.tolist(), classes, (0.8, 0.2), seed)
    fit_rows = train_rows[np.asarray(fit_part)]
    val_rows = train_rows[np.asarray(val_part)]
    val_vehicles = len(set(data["vehicle_ids"][val_rows].tolist()))
    if len(fit_rows) != FIT_IMAGES or len(val_rows) != VAL_IMAGES or val_vehicles != VAL_VEHICLES:
        sys.exit(
            "split sizes are fit " + str(len(fit_rows)) + " val " + str(len(val_rows))
            + " val vehicles " + str(val_vehicles)
        )
    return fit_rows, val_rows


def score(features, vehicles, cameras):
    index = BruteForceIndex(features.shape[1])
    index.add(features)
    plain = overall_score(query_scores(index, features, vehicles))
    camera = overall_score(camera_aware_scores(index, features, vehicles, cameras))
    return plain, camera


def fit_subset(x, vehicles, head, seed):
    _, y = np.unique(vehicles, return_inverse=True)
    n_classes = int(y.max()) + 1
    return fit(
        x, y, n_classes, int(head["dim"]), float(head["scale"]), float(head["lr"]),
        int(head["epochs"]), int(head["batch_size"]), seed, momentum=float(head["momentum"]),
    )


def show(name, budget, kept, plain, camera):
    print(
        name, "budget", budget, "vehicles", kept,
        "| camera-aware P@1", round(camera.precision_at_1, 4), "mAP@R", round(camera.map_at_r, 4),
        "| plain P@1", round(plain.precision_at_1, 4), "mAP@R", round(plain.map_at_r, 4),
        flush=True,
    )


def main():
    args = parse_args()
    config = load_config()
    guard = MemoryGuard(config.memory_limit_mb)
    head = np.load(args.head)
    data = np.load(args.embeddings, allow_pickle=True)
    print("head", args.head, "sha256", hashlib.sha256(open(args.head, "rb").read()).hexdigest())
    print("embeddings", args.embeddings, "sha256", hashlib.sha256(open(args.embeddings, "rb").read()).hexdigest())
    fit_rows, val_rows = parts(data, config.seed)
    print("fit", len(fit_rows), "val", len(val_rows), "seed", config.seed)
    fit_x = (data["embeddings"][fit_rows].astype(np.float64) - head["mean"]) / head["std"]
    val_x = (data["embeddings"][val_rows].astype(np.float64) - head["mean"]) / head["std"]
    fit_vehicles = data["vehicle_ids"][fit_rows]
    val_vehicles = data["vehicle_ids"][val_rows]
    val_cameras = data["camera_ids"][val_rows]
    results = {}

    def record(method, budget, model, selected):
        kept = len(set(fit_vehicles[selected].tolist()))
        plain, camera = score(project(model, val_x), val_vehicles, val_cameras)
        show(method, budget, kept, plain, camera)
        results.setdefault((method, budget), []).append(
            (camera.precision_at_1, camera.map_at_r, plain.precision_at_1, plain.map_at_r, kept)
        )
        guard.check()

    weights = head["weights"]
    stored = MetricHead(weights, np.empty((0, weights.shape[1])), float(head["scale"]))
    plain, camera = score(project(stored, val_x), val_vehicles, val_cameras)
    show("stored head", len(fit_rows), len(set(fit_vehicles.tolist())), plain, camera)
    print("stored val camera-aware mAP@R", round(float(head["val_camera_map"]), 4))
    everything = np.arange(len(fit_rows))
    record("refit all", len(fit_rows), fit_subset(fit_x, fit_vehicles, head, config.seed), everything)
    for start in STARTS:
        order = k_center_greedy(fit_x, max(BUDGETS), start)
        for budget in BUDGETS:
            selected = np.sort(order[:budget])
            record("kcenter start " + str(start), budget, fit_subset(fit_x[selected], fit_vehicles[selected], head, config.seed), selected)
    for seed in SEEDS:
        for budget in BUDGETS:
            rng = np.random.default_rng([seed, budget])
            selected = np.sort(rng.choice(len(fit_rows), size=budget, replace=False))
            assert len(selected) == budget
            record("random seed " + str(seed), budget, fit_subset(fit_x[selected], fit_vehicles[selected], head, config.seed), selected)
    print("summary (mean over 3 starts or seeds; camera-aware mAP@R min-max in brackets)")
    for budget in BUDGETS:
        for method in ("kcenter", "random"):
            rows = []
            for key, values in results.items():
                if key[1] == budget and key[0].startswith(method):
                    rows.extend(values)
            table = np.array(rows)
            print(
                method, "budget", budget, "runs", len(rows),
                "| camera-aware P@1", round(float(table[:, 0].mean()), 4),
                "mAP@R", round(float(table[:, 1].mean()), 4),
                "[" + str(round(float(table[:, 1].min()), 4)) + "-" + str(round(float(table[:, 1].max()), 4)) + "]",
                "| plain P@1", round(float(table[:, 2].mean()), 4),
                "mAP@R", round(float(table[:, 3].mean()), 4),
                "| vehicles", round(float(table[:, 4].mean()), 1),
            )


if __name__ == "__main__":
    main()
