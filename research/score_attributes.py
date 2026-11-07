import argparse
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict
from weathercam.core.cls_metrics import accuracy, confusion_matrix, macro_f1, per_class_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--heads-dir", default="outputs/heads_std")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.embeddings)
    split = data["split"]
    test = split == "test"
    train = split == "train"
    print("tag", args.tag, "test images", int(test.sum()))
    for attr in ("color", "type"):
        head = np.load(Path(args.heads_dir) / f"{args.tag}_{attr}.npz")
        class_ids = head["class_ids"]
        ids = data[f"{attr}_ids"]
        x = (data["embeddings"][test] - head["mean"]) / head["std"]
        model = SoftmaxModel(head["weights"], head["bias"])
        pred = class_ids[predict(model, x)].tolist()
        truth = ids[test].tolist()
        labels = class_ids.tolist()
        values, counts = np.unique(ids[train][head["train_idx"]], return_counts=True)
        majority = int(values[counts.argmax()])
        print(attr, "classes", len(labels), "lr", float(head["lr"]), "l2", float(head["l2"]))
        print(attr, "accuracy", round(accuracy(truth, pred), 4), "macro_f1", round(macro_f1(truth, pred, labels), 4))
        print(attr, "train_majority_class", majority, "accuracy_if_always", round(accuracy(truth, [majority] * len(truth)), 4))
        scores = per_class_scores(truth, pred, labels)
        for label in labels:
            s = scores[label]
            print(attr, "class", label, "support", s.support, "P", round(s.precision, 4), "R", round(s.recall, 4), "F1", round(s.f1, 4))
        print(attr, "confusion (rows true, columns predicted, class order", labels, ")")
        for label, row in zip(labels, confusion_matrix(truth, pred, labels)):
            print(attr, label, row)


if __name__ == "__main__":
    main()
