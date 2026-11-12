import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import fit, predict, predict_proba
from weathercam.attributes.weather import load_weather, part_arrays
from weathercam.core.cls_metrics import macro_f1
from weathercam.core.memory import MemoryGuard
from weathercam.core.roc_metrics import roc_auc
from weathercam.retrieval.ood import knn_distances

L2 = 0.01
LR = 1.0
EPOCHS = 300
SEED = 42
KS = (1, 5, 10, 20, 50)
SIZES = {"train": 716, "val": 161, "test": 150}
MEMORY_LIMIT_MB = 3000
N_CLASSES = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="outputs/embeddings/dawn_vitb16_laion2b.npz")
    p.add_argument("--expect-embeddings-sha256", required=True)
    p.add_argument("--expect-val-f1", type=float, required=True)
    p.add_argument("--stage", choices=("validation", "test"), required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def standardise(x, mean, std):
    return (x - mean) / std


def fit_on(x_all, y_all, keep):
    mapping = {old: new for new, old in enumerate(keep)}
    rows = np.isin(y_all, keep)
    x = x_all[rows]
    y = np.array([mapping[int(v)] for v in y_all[rows]], dtype=np.int64)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = standardise(x, mean, std)
    model = fit(z, y, len(keep), L2, LR, EPOCHS, SEED)
    return model, mean, std, z


def evaluate(model, mean, std, z, x_eval, y_eval, held_out, ks):
    z_eval = standardise(x_eval, mean, std)
    is_ood = (y_eval == held_out).astype(np.int64)
    proba = predict_proba(model, z_eval)
    baseline = roc_auc(1.0 - proba.max(axis=1), is_ood)
    knn = {k: roc_auc(knn_distances(z, z_eval, k), is_ood) for k in ks}
    assigned = np.bincount(proba[is_ood == 1].argmax(axis=1), minlength=proba.shape[1])
    return knn, baseline, int(is_ood.sum()), int(len(is_ood) - is_ood.sum()), assigned


def main():
    args = parse_args()
    if args.stage == "test" and args.k is None:
        raise SystemExit("--k is required for the test stage")
    if args.stage == "validation" and args.k is not None:
        raise SystemExit("--k is only for the test stage")
    if args.k is not None and args.k not in KS:
        raise SystemExit("--k must be one of %s" % (KS,))
    out = Path(args.out) if args.out else Path("outputs/holdout_weather/%s_v1.txt" % args.stage)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    emb_sha = hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest()
    if emb_sha != args.expect_embeddings_sha256:
        raise SystemExit("embeddings hash differs from the expected value")
    guard = MemoryGuard(MEMORY_LIMIT_MB)
    data = load_weather(args.embeddings)
    xt, yt = part_arrays(data, "train")
    xv, yv = part_arrays(data, "val")
    if (len(xt), len(xv)) != (SIZES["train"], SIZES["val"]):
        raise SystemExit("train or val size differs from 716/161")
    if len(data.weathers) != N_CLASSES:
        raise SystemExit("expected 4 weather classes")
    start = time.perf_counter()
    lines = []

    def emit(text):
        print(text, flush=True)
        lines.append(text)

    emit("stage %s" % args.stage)
    emit("embeddings sha256 %s" % emb_sha)
    emit("classes %s" % ", ".join(data.weathers))
    emit("l2 %s lr %s epochs %d seed %d" % (L2, LR, EPOCHS, SEED))
    keep_all = list(range(N_CLASSES))
    model, mean, std, _ = fit_on(xt, yt, keep_all)
    val_pred = predict(model, standardise(xv, mean, std)).tolist()
    val_f1 = macro_f1(yv.tolist(), val_pred, keep_all)
    emit("val macro-F1 4-class %.4f (expected %.4f)" % (val_f1, args.expect_val_f1))
    if abs(val_f1 - args.expect_val_f1) > 1e-4:
        raise SystemExit("val macro-F1 does not reproduce; nothing scored")
    if args.stage == "validation":
        x_eval, y_eval, ks = xv, yv, KS
    else:
        x_eval, y_eval = part_arrays(data, "test")
        if len(x_eval) != SIZES["test"]:
            raise SystemExit("test size differs from 150")
        ks = (args.k,)
    results = []
    for held_out in range(N_CLASSES):
        keep = [c for c in range(N_CLASSES) if c != held_out]
        model, mean, std, z = fit_on(xt, yt, keep)
        knn, baseline, n_ood, n_id, assigned = evaluate(model, mean, std, z, x_eval, y_eval, held_out, ks)
        results.append((knn, baseline))
        names = [data.weathers[c] for c in keep]
        emit("held_out %s n_ood %d n_id %d bank %d" % (data.weathers[held_out], n_ood, n_id, len(z)))
        emit("  max-softmax baseline auroc %.4f" % baseline)
        for k in ks:
            emit("  knn k %d auroc %.4f" % (k, knn[k]))
        emit("  head assigns held-out images to " + ", ".join("%s %d" % (n, int(a)) for n, a in zip(names, assigned)))
        guard.check()
    emit("mean over %d folds (unweighted)" % N_CLASSES)
    emit("  max-softmax baseline %.4f" % np.mean([r[1] for r in results]))
    means = {k: float(np.mean([r[0][k] for r in results])) for k in ks}
    for k in ks:
        emit("  knn k %d %.4f" % (k, means[k]))
    if args.stage == "validation":
        best = ks[0]
        for k in ks[1:]:
            if means[k] > means[best]:
                best = k
        emit("selected k %d" % best)
        if best in (KS[0], KS[-1]):
            emit("grid edge: not bracketed")
    emit("elapsed %.1fs" % (time.perf_counter() - start))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
