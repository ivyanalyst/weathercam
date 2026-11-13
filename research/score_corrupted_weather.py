import argparse
import hashlib
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict
from weathercam.attributes.weather import load_weather, majority_accuracy, part_arrays, wilson_interval
from weathercam.core.cls_metrics import macro_f1
from weathercam.core.memory import MemoryGuard

SIZES = {"validation": 161, "test": 150}
PART = {"validation": "val", "test": "test"}
GENERIC = ("blur", "noise", "darken", "jpeg")
WEATHER_LIKE = ("fog", "haze", "rain", "snow")
N_CONDITIONS = 17
MEMORY_LIMIT_MB = 3000
MAX_CLEAN_DIFF = 1e-4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="outputs/heads_weather/vitb16_laion2b_weather.npz")
    p.add_argument("--clean", default="outputs/embeddings/dawn_vitb16_laion2b.npz")
    p.add_argument("--robust", default="outputs/robustness/dawn_valtest_v1.npz")
    p.add_argument("--expect-head-sha256", required=True)
    p.add_argument("--expect-clean-sha256", required=True)
    p.add_argument("--stage", choices=("validation", "test"), required=True)
    p.add_argument("--expect-clean-correct", type=int, required=True)
    p.add_argument("--expect-clean-f1", type=float, required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(name, pred, y, clean_pred, labels):
    n = len(y)
    correct = int((pred == y).sum())
    low, high = wilson_interval(correct, n)
    f1 = macro_f1(y.tolist(), pred.tolist(), labels)
    flips = int((pred != clean_pred).sum())
    lost = int(((clean_pred == y) & (pred != y)).sum())
    gained = int(((clean_pred != y) & (pred == y)).sum())
    predicted = "/".join(str(int(c)) for c in np.bincount(pred, minlength=len(labels)))
    recalls = []
    for c in labels:
        members = y == c
        total = int(members.sum())
        recalls.append("%.3f" % ((pred[members] == c).sum() / total) if total else "n/a")
    return "%-10s acc %.4f (%.4f-%.4f) f1 %.4f correct %d/%d flips %d lost %d gained %d pred %s recall %s" % (
        name, correct / n, low, high, f1, correct, n, flips, lost, gained, predicted, "/".join(recalls)
    )


def main():
    args = parse_args()
    out = Path(args.out) if args.out else Path("outputs/robustness/%s_v1.txt" % args.stage)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    head_sha = sha256_of(args.head)
    clean_sha = sha256_of(args.clean)
    robust_sha = sha256_of(args.robust)
    if head_sha != args.expect_head_sha256 or clean_sha != args.expect_clean_sha256:
        raise SystemExit("head or clean embeddings hash differs from the expected value")
    guard = MemoryGuard(MEMORY_LIMIT_MB)
    head = np.load(args.head, allow_pickle=False)
    if str(head["embeddings_sha256"]) != clean_sha:
        raise SystemExit("head was not trained on this clean embeddings file")
    weathers = tuple(str(w) for w in head["weathers"])
    data = load_weather(args.clean)
    if data.weathers != weathers:
        raise SystemExit("class lists differ")
    with np.load(args.clean, allow_pickle=False) as stored:
        stored_split_sha = str(stored["split_sha256"])
    robust = np.load(args.robust, allow_pickle=False)
    if str(robust["clean_embeddings_sha256"]) != clean_sha:
        raise SystemExit("robustness file was made from a different clean file")
    if str(robust["split_sha256"]) != stored_split_sha:
        raise SystemExit("split hash differs")
    if tuple(str(w) for w in robust["weathers"]) != weathers:
        raise SystemExit("robustness class list differs")
    rows = robust["rows"]
    if [data.image_names[r] for r in rows] != [str(v) for v in robust["image_names"]]:
        raise SystemExit("image names differ from the clean file")
    if not np.array_equal(data.labels[rows], robust["weather_ids"]):
        raise SystemExit("weather ids differ from the clean file")
    if not np.array_equal(data.split[rows], robust["split"]):
        raise SystemExit("split labels differ from the clean file")
    conditions = [str(c) for c in robust["conditions"]]
    corruptions = [str(c) for c in robust["corruptions"]]
    embeddings = robust["embeddings"]
    if len(conditions) != N_CONDITIONS or conditions[0] != "clean" or embeddings.shape[0] != N_CONDITIONS:
        raise SystemExit("unexpected condition list")
    mask = robust["split"] == PART[args.stage]
    if int(mask.sum()) != SIZES[args.stage]:
        raise SystemExit("expected %d %s rows, found %d" % (SIZES[args.stage], PART[args.stage], int(mask.sum())))
    y = robust["weather_ids"][mask].astype(np.int64)
    x = embeddings[:, mask, :]
    diff = float(np.abs(x[0].astype(np.float64) - data.embeddings[rows][mask].astype(np.float64)).max())
    if diff > MAX_CLEAN_DIFF:
        raise SystemExit("clean condition differs from the stored embeddings")
    model = SoftmaxModel(head["weights"], head["bias"])
    mean, std = head["mean"], head["std"]
    labels = list(range(len(weathers)))
    preds = [predict(model, (x[i].astype(np.float64) - mean) / std) for i in range(N_CONDITIONS)]
    clean_pred = preds[0]
    clean_correct = int((clean_pred == y).sum())
    clean_f1 = macro_f1(y.tolist(), clean_pred.tolist(), labels)
    _, y_train = part_arrays(data, "train")
    lines = []

    def emit(text):
        print(text, flush=True)
        lines.append(text)

    emit("stage %s" % args.stage)
    emit("head sha256 %s" % head_sha)
    emit("clean embeddings sha256 %s" % clean_sha)
    emit("robustness sha256 %s" % robust_sha)
    emit("classes %s" % ", ".join(weathers))
    emit("images %d, class counts %s" % (len(y), "/".join(str(int(c)) for c in np.bincount(y, minlength=len(labels)))))
    emit("train-majority baseline %.4f" % majority_accuracy(y_train, y))
    emit("clean max abs diff to stored %.2e" % diff)
    emit("clean correct %d (expected %d), macro-F1 %.4f (expected %.4f)" % (clean_correct, args.expect_clean_correct, clean_f1, args.expect_clean_f1))
    if clean_correct != args.expect_clean_correct or abs(clean_f1 - args.expect_clean_f1) > 1e-4:
        raise SystemExit("clean condition does not reproduce; nothing scored")
    emit("pred and recall columns follow the class order above")
    emit("")
    emit(describe("clean", clean_pred, y, clean_pred, labels))
    for title, group in (("generic", GENERIC), ("weather-like", WEATHER_LIKE)):
        emit("")
        emit(title)
        for i, name in enumerate(corruptions):
            if name in group:
                emit(describe(conditions[i], preds[i], y, clean_pred, labels))
    guard.check()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
