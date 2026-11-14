import argparse
import hashlib
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict
from weathercam.attributes.weather import load_weather
from weathercam.core.cls_metrics import macro_f1
from weathercam.core.memory import MemoryGuard

SEED = 42
SIGMAS = (0.01, 0.02, 0.03, 0.05)
DRAWS = 20
SIZES = {"validation": 161, "test": 150}
PART = {"validation": "val", "test": "test"}
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
    p.add_argument("--expect-robust-sha256", required=True)
    p.add_argument("--stage", choices=("validation", "test"), required=True)
    p.add_argument("--expect-clean-correct", type=int, required=True)
    p.add_argument("--expect-clean-f1", type=float, required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mean_cosine(a, b):
    return float((a * b).sum(axis=1).mean())


def main():
    args = parse_args()
    out = Path(args.out) if args.out else Path("outputs/robustness/feature_noise_%s_v1.txt" % args.stage)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    head_sha = sha256_of(args.head)
    clean_sha = sha256_of(args.clean)
    robust_sha = sha256_of(args.robust)
    if (head_sha, clean_sha, robust_sha) != (args.expect_head_sha256, args.expect_clean_sha256, args.expect_robust_sha256):
        raise SystemExit("a file hash differs from the expected value")
    guard = MemoryGuard(MEMORY_LIMIT_MB)
    head = np.load(args.head, allow_pickle=False)
    if str(head["embeddings_sha256"]) != clean_sha:
        raise SystemExit("head was not trained on this clean embeddings file")
    weathers = tuple(str(w) for w in head["weathers"])
    data = load_weather(args.clean)
    if data.weathers != weathers:
        raise SystemExit("class lists differ")
    robust = np.load(args.robust, allow_pickle=False)
    if str(robust["clean_embeddings_sha256"]) != clean_sha:
        raise SystemExit("robustness file was made from a different clean file")
    rows = robust["rows"]
    if [data.image_names[r] for r in rows] != [str(v) for v in robust["image_names"]]:
        raise SystemExit("image names differ from the clean file")
    if not np.array_equal(data.labels[rows], robust["weather_ids"]):
        raise SystemExit("weather ids differ from the clean file")
    if not np.array_equal(data.split[rows], robust["split"]):
        raise SystemExit("split labels differ from the clean file")
    conditions = [str(c) for c in robust["conditions"]]
    embeddings = robust["embeddings"]
    if len(conditions) != N_CONDITIONS or conditions[0] != "clean" or embeddings.shape[0] != N_CONDITIONS:
        raise SystemExit("unexpected condition list")
    mask = robust["split"] == PART[args.stage]
    if int(mask.sum()) != SIZES[args.stage]:
        raise SystemExit("expected %d %s rows, found %d" % (SIZES[args.stage], PART[args.stage], int(mask.sum())))
    y = robust["weather_ids"][mask].astype(np.int64)
    x_clean = data.embeddings[rows][mask].astype(np.float64)
    diff = float(np.abs(embeddings[0][mask].astype(np.float64) - x_clean).max())
    if diff > MAX_CLEAN_DIFF:
        raise SystemExit("clean condition differs from the stored embeddings")
    model = SoftmaxModel(head["weights"], head["bias"])
    mean, std = head["mean"], head["std"]
    labels = list(range(len(weathers)))

    def classify(x):
        return predict(model, (x - mean) / std)

    clean_pred = classify(x_clean)
    clean_correct = int((clean_pred == y).sum())
    clean_f1 = macro_f1(y.tolist(), clean_pred.tolist(), labels)
    lines = []

    def emit(text):
        print(text, flush=True)
        lines.append(text)

    def counts(pred):
        lost = int(((clean_pred == y) & (pred != y)).sum())
        gained = int(((clean_pred != y) & (pred == y)).sum())
        return int((pred == y).sum()), lost, gained

    emit("stage %s" % args.stage)
    emit("head sha256 %s" % head_sha)
    emit("clean embeddings sha256 %s" % clean_sha)
    emit("robustness sha256 %s" % robust_sha)
    emit("images %d, class counts %s" % (len(y), "/".join(str(int(c)) for c in np.bincount(y, minlength=len(labels)))))
    emit("clean correct %d (expected %d), macro-F1 %.4f (expected %.4f)" % (clean_correct, args.expect_clean_correct, clean_f1, args.expect_clean_f1))
    if clean_correct != args.expect_clean_correct or abs(clean_f1 - args.expect_clean_f1) > 1e-4:
        raise SystemExit("clean condition does not reproduce; nothing scored")
    emit("")
    emit("gaussian noise on unit-norm embeddings, renormalised; %d draws per level; seed [%d, level index, draw]" % (DRAWS, SEED))
    for level, sigma in enumerate(SIGMAS):
        cosines, corrects, losts, gaineds = [], [], [], []
        for draw in range(DRAWS):
            rng = np.random.default_rng([SEED, level, draw])
            noisy = unit_rows(x_clean + rng.normal(0.0, sigma, size=x_clean.shape))
            correct, lost, gained = counts(classify(noisy))
            cosines.append(mean_cosine(x_clean, noisy))
            corrects.append(correct)
            losts.append(lost)
            gaineds.append(gained)
        emit("noise s %.2f cos mean %.4f correct mean %.1f [min %d max %d] lost mean %.1f gained mean %.1f" % (
            sigma, np.mean(cosines), np.mean(corrects), min(corrects), max(corrects), np.mean(losts), np.mean(gaineds)))
        guard.check()
    emit("")
    emit("image conditions (one render each), cosine to the stored clean embeddings")
    for i, name in enumerate(conditions):
        x = embeddings[i][mask].astype(np.float64)
        correct, lost, gained = counts(classify(x))
        emit("%-10s cos mean %.4f correct %d lost %d gained %d" % (name, mean_cosine(x_clean, x), correct, lost, gained))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
