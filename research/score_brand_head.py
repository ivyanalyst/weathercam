import argparse
from pathlib import Path

import numpy as np

from weathercam.attributes.softmax import SoftmaxModel, predict
from weathercam.core.cls_metrics import accuracy, confusion_matrix, macro_f1, per_class_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--heads-dir", default="outputs/heads_brand")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.embeddings)
    head = np.load(Path(args.heads_dir) / f"{args.tag}_brand.npz")
    split = data["split"]
    test = split == "test"
    train = split == "train"
    makes = [str(m) for m in head["makes"]]
    labels = head["class_ids"].tolist()
    x = (data["embeddings"][test] - head["mean"]) / head["std"]
    model = SoftmaxModel(head["weights"], head["bias"])
    pred = predict(model, x).tolist()
    truth = data["make_ids"][test].tolist()
    print("tag", args.tag, "test images", int(test.sum()), "classes", len(labels))
    print("lr", float(head["lr"]), "l2", float(head["l2"]), "epochs", int(head["epochs"]))
    values, counts = np.unique(data["make_ids"][train], return_counts=True)
    majority = int(values[counts.argmax()])
    print("accuracy", round(accuracy(truth, pred), 4), "macro_f1", round(macro_f1(truth, pred, labels), 4))
    print("train_majority", makes[majority], "accuracy_if_always", round(accuracy(truth, [majority] * len(truth)), 4))
    scores = per_class_scores(truth, pred, labels)
    for label in labels:
        s = scores[label]
        print("class", makes[label], "support", s.support, "P", round(s.precision, 4), "R", round(s.recall, 4), "F1", round(s.f1, 4))
    ranked = sorted((l for l in labels if scores[l].support > 0), key=lambda l: scores[l].f1)
    print("lowest F1:", [(makes[l], round(scores[l].f1, 3), scores[l].support) for l in ranked[:8]])
    matrix = np.array(confusion_matrix(truth, pred, labels))
    pairs = [(int(matrix[i, j]), makes[labels[i]], makes[labels[j]]) for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j] > 0]
    pairs.sort(reverse=True)
    print("top confusions (count, true, predicted):")
    for count, a, b in pairs[:15]:
        print(" ", count, a, "->", b)


if __name__ == "__main__":
    main()
