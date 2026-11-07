import argparse
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import fit, predict
from weathercam.core.cls_metrics import accuracy, macro_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default="outputs/heads_brand")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--l2s", type=float, nargs="+", default=[0.0, 0.0001])
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.embeddings)
    split = data["split"]
    makes = data["makes"]
    labels = list(range(len(makes)))
    y_all = data["make_ids"]
    train = split == "train"
    val = split == "val"
    print("tag", args.tag, "train", int(train.sum()), "val", int(val.sum()), "classes", len(labels))
    print("classes missing in train:", sorted(set(labels) - set(y_all[train].tolist())))
    raw = data["embeddings"]
    mean = raw[train].mean(axis=0)
    std = raw[train].std(axis=0) + 1e-6
    x = (raw - mean) / std
    best = None
    for lr in args.lrs:
        for l2 in args.l2s:
            model = fit(x[train], y_all[train], len(labels), l2, lr, args.epochs, args.seed)
            pred = predict(model, x[val]).tolist()
            truth = y_all[val].tolist()
            acc = accuracy(truth, pred)
            f1 = macro_f1(truth, pred, labels)
            print("lr", lr, "l2", l2, "val_acc", round(acc, 4), "val_macro_f1", round(f1, 4))
            if best is None or f1 > best[0]:
                best = (f1, acc, lr, l2, model)
    f1, acc, lr, l2, model = best
    print("best lr", lr, "l2", l2, "val_acc", round(acc, 4), "val_macro_f1", round(f1, 4))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{args.tag}_brand.npz",
        weights=model.weights,
        bias=model.bias,
        class_ids=np.array(labels),
        makes=makes,
        lr=lr,
        l2=l2,
        epochs=args.epochs,
        seed=args.seed,
        mean=mean,
        std=std,
    )


if __name__ == "__main__":
    main()
