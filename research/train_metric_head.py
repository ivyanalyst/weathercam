import argparse
import time
from pathlib import Path

import numpy as np

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.data.split import group_split
from weathercam.retrieval.evaluate import camera_aware_scores, overall_score, query_scores
from weathercam.retrieval.index import BruteForceIndex
from weathercam.retrieval.metric_head import fit, loss_and_grads, project


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default="outputs/heads_metric")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.2])
    p.add_argument("--scales", type=float, nargs="+", default=[16.0, 32.0])
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


def mean_loss(model, x, y, scale, chunk=4096):
    total = 0.0
    for start in range(0, len(y), chunk):
        stop = min(start + chunk, len(y))
        value = loss_and_grads(model.weights, model.prototypes, x[start:stop], y[start:stop], scale)[0]
        total += value * (stop - start)
    return total / len(y)


def main():
    args = parse_args()
    config = load_config()
    seed = config.seed
    guard = MemoryGuard(config.memory_limit_mb)
    data = np.load(args.embeddings, allow_pickle=True)
    rows = np.nonzero(data["split"] == "train")[0]
    vehicles = data["vehicle_ids"][rows]
    cameras = data["camera_ids"][rows]
    classes = list(zip(data["color_ids"][rows].tolist(), data["type_ids"][rows].tolist()))
    fit_part, val_part = group_split(vehicles.tolist(), classes, (0.8, 0.2), seed)
    fit_part = np.asarray(fit_part)
    val_part = np.asarray(val_part)
    raw = data["embeddings"][rows].astype(np.float64)
    fit_vehicles = set(vehicles[fit_part].tolist())
    val_vehicles = set(vehicles[val_part].tolist())
    print(
        "tag", args.tag, "seed", seed,
        "fit images", len(fit_part), "fit vehicles", len(fit_vehicles),
        "val images", len(val_part), "val vehicles", len(val_vehicles),
        "overlap", len(fit_vehicles & val_vehicles),
    )
    _, y_fit = np.unique(vehicles[fit_part], return_inverse=True)
    y_fit = y_fit.reshape(-1)
    n_classes = int(y_fit.max()) + 1
    mean = raw[fit_part].mean(axis=0)
    std = raw[fit_part].std(axis=0) + 1e-6
    x_fit = (raw[fit_part] - mean) / std
    x_val = (raw[val_part] - mean) / std
    val_vehicle = vehicles[val_part]
    val_camera = cameras[val_part]
    plain, camera = score(raw[val_part], val_vehicle, val_camera)
    show("baseline raw 512-d", plain, camera)
    plain, camera = score(x_val, val_vehicle, val_camera)
    show("baseline standardised 512-d", plain, camera)
    guard.check()
    best = None
    for lr in args.lrs:
        for scale in args.scales:
            start = time.perf_counter()
            model = fit(
                x_fit, y_fit, n_classes, args.dim, scale, lr, args.epochs,
                args.batch_size, seed, l2=0.0, momentum=args.momentum,
            )
            seconds = time.perf_counter() - start
            loss = mean_loss(model, x_fit, y_fit, scale)
            plain, camera = score(project(model, x_val), val_vehicle, val_camera)
            show("head lr %s scale %s train_loss %.4f fit_s %.1f" % (lr, scale, loss, seconds), plain, camera)
            guard.check()
            if best is None or camera.map_at_r > best[0]:
                best = (camera.map_at_r, lr, scale, model, plain, camera)
    _, lr, scale, model, plain, camera = best
    print("best lr", lr, "scale", scale, "camera-aware val mAP@R", round(camera.map_at_r, 4))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{args.tag}_metric.npz",
        weights=model.weights,
        mean=mean,
        std=std,
        dim=args.dim,
        scale=scale,
        lr=lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        momentum=args.momentum,
        seed=seed,
        val_plain_p1=plain.precision_at_1,
        val_plain_map=plain.map_at_r,
        val_camera_p1=camera.precision_at_1,
        val_camera_map=camera.map_at_r,
    )


if __name__ == "__main__":
    main()
