import argparse
import hashlib
import itertools
import sys

import numpy as np

from weathercam.core.config import load_config
from weathercam.core.memory import MemoryGuard
from weathercam.core.roc_metrics import roc_auc
from weathercam.retrieval.innersplit import veri_inner_split
from weathercam.retrieval.metric_head import MetricHead, project
from weathercam.retrieval.ood import knn_distances

KS = (1, 5, 10, 20, 50, 100)
SPACES = ("standardised", "head")
FIT_IMAGES = 29918
VAL_IMAGES = 7828
VAL_VEHICLES = 119


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--veri", required=True)
    p.add_argument("--dawn", required=True)
    p.add_argument("--vmmr", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--mode", choices=("validation", "test"), required=True)
    p.add_argument("--space", choices=SPACES)
    p.add_argument("--k", type=int)
    args = p.parse_args()
    given = args.space is not None or args.k is not None
    if args.mode == "validation" and given:
        p.error("validation mode takes no --space or --k")
    if args.mode == "test" and (args.space is None or args.k is None):
        p.error("test mode needs --space and --k")
    return args


def veri_parts(data, seed):
    fit_rows, val_rows = veri_inner_split(data, seed)
    val_vehicles = len(set(data["vehicle_ids"][val_rows].tolist()))
    if len(fit_rows) != FIT_IMAGES or len(val_rows) != VAL_IMAGES or val_vehicles != VAL_VEHICLES:
        sys.exit(
            "split sizes are fit " + str(len(fit_rows)) + " val " + str(len(val_rows))
            + " val vehicles " + str(val_vehicles)
        )
    return fit_rows, val_rows


def transform(raw, head, model, space):
    standardised = (raw.astype(np.float64) - head["mean"]) / head["std"]
    if space == "standardised":
        return standardised
    return project(model, standardised)


def ood_auroc(bank, id_features, ood_features, k):
    scores = np.concatenate(
        (knn_distances(bank, id_features, k), knn_distances(bank, ood_features, k))
    )
    labels = np.concatenate((np.zeros(len(id_features)), np.ones(len(ood_features))))
    return roc_auc(scores, labels)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    args = parse_args()
    config = load_config()
    guard = MemoryGuard(config.memory_limit_mb)
    head = np.load(args.head)
    veri = np.load(args.veri, allow_pickle=True)
    dawn = np.load(args.dawn, allow_pickle=True)
    vmmr = np.load(args.vmmr, allow_pickle=True)
    print("head", args.head, "sha256", sha(args.head))
    print("veri", args.veri, "sha256", sha(args.veri))
    print("dawn", args.dawn, "sha256", sha(args.dawn))
    print("vmmr", args.vmmr, "sha256", sha(args.vmmr))
    fit_rows, val_rows = veri_parts(veri, config.seed)
    if args.mode == "validation":
        id_rows = val_rows
        wanted = ("train", "val")
    else:
        id_rows = np.flatnonzero(veri["split"] == "test")
        wanted = ("test",)
    dawn_rows = np.flatnonzero(np.isin(dawn["split"], wanted))
    vmmr_rows = np.flatnonzero(np.isin(vmmr["split"], wanted))
    print(
        "mode", args.mode, "bank", len(fit_rows), "id", len(id_rows),
        "dawn", len(dawn_rows), "vmmr", len(vmmr_rows),
    )
    weights = head["weights"]
    model = MetricHead(weights, np.empty((0, weights.shape[1])), float(head["scale"]))
    if args.mode == "validation":
        configs = list(itertools.product(SPACES, KS))
    else:
        configs = [(args.space, args.k)]
    spaces = sorted({space for space, _ in configs})
    features = {}
    for space in spaces:
        features[space] = (
            transform(veri["embeddings"][fit_rows], head, model, space),
            transform(veri["embeddings"][id_rows], head, model, space),
            transform(dawn["embeddings"][dawn_rows], head, model, space),
            transform(vmmr["embeddings"][vmmr_rows], head, model, space),
        )
    best = None
    for space, k in configs:
        bank, id_features, dawn_features, vmmr_features = features[space]
        dawn_auc = ood_auroc(bank, id_features, dawn_features, k)
        vmmr_auc = ood_auroc(bank, id_features, vmmr_features, k)
        mean_auc = (dawn_auc + vmmr_auc) / 2.0
        print(
            "space", space, "k", k,
            "| DAWN AUROC", round(dawn_auc, 4), "| VMMR AUROC", round(vmmr_auc, 4),
            "| mean", round(mean_auc, 4),
            flush=True,
        )
        guard.check()
        if best is None or mean_auc > best[0]:
            best = (mean_auc, space, k)
    if args.mode == "validation":
        print("best mean AUROC", round(best[0], 4), "space", best[1], "k", best[2])


if __name__ == "__main__":
    main()
