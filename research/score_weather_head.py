import argparse
import hashlib
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict
from weathercam.attributes.weather import load_weather, majority_accuracy, part_arrays, wilson_interval
from weathercam.core.cls_metrics import accuracy, confusion_matrix, macro_f1, per_class_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="outputs/heads_weather/vitb16_laion2b_weather.npz")
    p.add_argument("--embeddings", default="outputs/embeddings/dawn_vitb16_laion2b.npz")
    p.add_argument("--out", default="outputs/weather/score_test.txt")
    p.add_argument("--expect-val-f1", type=float, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    emb_sha = hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest()
    head_sha = hashlib.sha256(Path(args.head).read_bytes()).hexdigest()
    head = np.load(args.head, allow_pickle=False)
    if str(head["embeddings_sha256"]) != emb_sha:
        raise SystemExit("embeddings file does not match the one the head was trained on")
    weathers = tuple(str(w) for w in head["weathers"])
    data = load_weather(args.embeddings)
    if data.weathers != weathers:
        raise SystemExit("class lists differ")
    model = SoftmaxModel(head["weights"], head["bias"])
    mean, std = head["mean"], head["std"]
    labels = list(range(len(weathers)))
    xt, yt = part_arrays(data, "train")
    xv, yv = part_arrays(data, "val")
    xs, ys = part_arrays(data, "test")

    def run(x, y):
        pred = predict(model, (x - mean) / std).tolist()
        truth = y.tolist()
        return truth, pred, accuracy(truth, pred), macro_f1(truth, pred, labels)

    _, _, train_acc, train_f1 = run(xt, yt)
    _, _, val_acc, val_f1 = run(xv, yv)
    print("val macro_f1 %.4f (expected %.4f)" % (val_f1, args.expect_val_f1))
    if abs(val_f1 - args.expect_val_f1) > 1e-4:
        raise SystemExit("val macro-F1 does not reproduce; test not scored")

    truth, pred, acc, f1 = run(xs, ys)
    n = len(truth)
    correct = sum(1 for t, p in zip(truth, pred) if t == p)
    low, high = wilson_interval(correct, n)
    baseline = majority_accuracy(yt, ys)
    matrix = confusion_matrix(truth, pred, labels)
    scores = per_class_scores(truth, pred, labels)
    names = np.array(data.image_names)[data.split == "test"]
    lines = [
        "head %s sha256 %s" % (args.head, head_sha),
        "embeddings sha256 %s" % emb_sha,
        "head lr %s l2 %s epochs %d seed %d" % (float(head["lr"]), float(head["l2"]), int(head["epochs"]), int(head["seed"])),
        "classes %s" % (weathers,),
        "train acc %.4f macro_f1 %.4f" % (train_acc, train_f1),
        "val   acc %.4f macro_f1 %.4f" % (val_acc, val_f1),
        "TEST  n %d correct %d acc %.4f wilson95 %.4f-%.4f macro_f1 %.4f" % (n, correct, acc, low, high, f1),
        "test majority baseline (train majority) %.4f" % baseline,
        "confusion rows=true cols=pred %s" % (weathers,),
    ]
    for name, row in zip(weathers, matrix):
        lines.append("  %-5s %s" % (name, row))
    for label, s in scores.items():
        lines.append(
            "  %-5s precision %.3f recall %.3f f1 %.3f support %d" % (weathers[label], s.precision, s.recall, s.f1, s.support)
        )
    lines.append("misclassified test images (true -> predicted)")
    for name, t, p in zip(names, truth, pred):
        if t != p:
            lines.append("  %s %s -> %s" % (name, weathers[t], weathers[p]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
