import argparse
import hashlib
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import fit, predict
from weathercam.attributes.weather import load_weather, majority_accuracy, part_arrays
from weathercam.core.cls_metrics import accuracy, macro_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="outputs/embeddings/dawn_vitb16_laion2b.npz")
    p.add_argument("--out", default="outputs/heads_weather/vitb16_laion2b_weather.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--l2s", type=float, nargs="+", default=[0.1, 0.01, 0.001, 0.0001, 0.0])
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    emb_sha = hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest()
    data = load_weather(args.embeddings)
    labels = list(range(len(data.weathers)))
    xt, yt = part_arrays(data, "train")
    xv, yv = part_arrays(data, "val")
    print("embeddings sha256", emb_sha)
    print("classes", data.weathers, "train", len(yt), "val", len(yv))
    print("val majority baseline", round(majority_accuracy(yt, yv), 4))
    mean = xt.mean(axis=0)
    std = xt.std(axis=0) + 1e-6
    xt = (xt - mean) / std
    xv = (xv - mean) / std
    truth = yv.tolist()
    best = None
    for l2 in args.l2s:
        for lr in args.lrs:
            model = fit(xt, yt, len(labels), l2, lr, args.epochs, args.seed)
            pred = predict(model, xv).tolist()
            acc = accuracy(truth, pred)
            f1 = macro_f1(truth, pred, labels)
            print("l2", l2, "lr", lr, "val_acc", round(acc, 4), "val_macro_f1", round(f1, 4))
            if best is None or f1 > best[0]:
                best = (f1, acc, lr, l2, model)
    f1, acc, lr, l2, model = best
    train_acc = accuracy(yt.tolist(), predict(model, xt).tolist())
    print("best l2", l2, "lr", lr, "val_acc", round(acc, 4), "val_macro_f1", round(f1, 4), "train_acc", round(train_acc, 4))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        weights=model.weights,
        bias=model.bias,
        class_ids=np.array(labels),
        weathers=np.array(data.weathers),
        lr=lr,
        l2=l2,
        epochs=args.epochs,
        seed=args.seed,
        mean=mean,
        std=std,
        embeddings_sha256=np.array(emb_sha),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
