import argparse
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import fit, predict
from weathercam.core.cls_metrics import accuracy, macro_f1
from weathercam.data.split import group_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default="outputs/heads")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lrs", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p.add_argument("--l2s", type=float, nargs="+", default=[0.0, 0.0001])
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.embeddings)
    rows = data["split"] == "train"
    x = data["embeddings"][rows]
    vehicles = data["vehicle_ids"][rows].tolist()
    targets = {"color": data["color_ids"][rows], "type": data["type_ids"][rows]}
    classes = list(zip(targets["color"].tolist(), targets["type"].tolist()))
    train_idx, val_idx = group_split(vehicles, classes, (0.8, 0.2), args.seed)
    for name, idx in (("train", train_idx), ("val", val_idx)):
        print(name, "images", len(idx), "vehicles", len({vehicles[i] for i in idx}))
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0) + 1e-6
    x = (x - mean) / std
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for attr, ids in targets.items():
        class_ids = np.array(sorted(set(ids.tolist())))
        y = np.searchsorted(class_ids, ids)
        print(attr, "classes", len(class_ids), class_ids.tolist())
        best = None
        for lr in args.lrs:
            for l2 in args.l2s:
                model = fit(x[train_idx], y[train_idx], len(class_ids), l2, lr, args.epochs, args.seed)
                pred = predict(model, x[val_idx]).tolist()
                truth = y[val_idx].tolist()
                acc = accuracy(truth, pred)
                f1 = macro_f1(truth, pred)
                print(attr, "lr", lr, "l2", l2, "val_acc", round(acc, 4), "val_macro_f1", round(f1, 4))
                if best is None or f1 > best[0]:
                    best = (f1, acc, lr, l2, model)
        f1, acc, lr, l2, model = best
        print(attr, "best lr", lr, "l2", l2, "val_acc", round(acc, 4), "val_macro_f1", round(f1, 4))
        np.savez(
            out_dir / f"{args.tag}_{attr}.npz",
            weights=model.weights,
            bias=model.bias,
            class_ids=class_ids,
            lr=lr,
            l2=l2,
            epochs=args.epochs,
            seed=args.seed,
            train_idx=train_idx,
            val_idx=val_idx,
            mean=mean,
            std=std,
        )


if __name__ == "__main__":
    main()
