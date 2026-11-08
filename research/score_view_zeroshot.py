import argparse
from pathlib import Path

import numpy as np


def read_labels(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if p:
                rows.append((p[0], p[1], int(p[2])))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def acc_line(name, true, pred):
    n = len(true)
    k = int((true == pred).sum())
    lo, hi = wilson(k, n)
    return "%s: %d/%d = %.3f (95%% Wilson %.3f-%.3f)" % (name, k, n, k / max(n, 1), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/view/hand_labels.txt")
    ap.add_argument("--pred", default="outputs/view/sample_predictions.npz")
    ap.add_argument("--out", default="outputs/view/score_view.txt")
    args = ap.parse_args()

    rows = read_labels(args.labels)
    p = np.load(args.pred)
    tids = [str(t) for t in p["tids"]]
    classes = [str(c) for c in p["classes"]]
    assert [r[0] for r in rows] == tids
    pred_all = p["pred"]
    conf_all = p["conf"]
    mixed = np.array([r[1] == "mixed" for r in rows])
    obl = np.array([r[2] == 1 for r in rows])
    keep = ~mixed
    true = np.array([classes.index(r[1]) for r in rows if r[1] != "mixed"])
    pred = pred_all[keep]
    k = len(classes)

    out = []
    out.append("tracks %d scored %d mixed %d oblique %d clean %d" % (
        len(rows), keep.sum(), mixed.sum(), (obl & keep).sum(), (~obl & keep).sum()))
    out.append(acc_line("accuracy all scored", true, pred))
    out.append(acc_line("accuracy clean", true[~obl[keep]], pred[~obl[keep]]))
    out.append(acc_line("accuracy oblique", true[obl[keep]], pred[obl[keep]]))
    major = int(np.bincount(true, minlength=k).argmax())
    out.append("majority baseline (%s): %.3f" % (classes[major], float((true == major).mean())))

    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (true, pred), 1)
    out.append("confusion rows=hand, cols=pred: " + " ".join(classes))
    for i in range(k):
        out.append("%-6s %s" % (classes[i], " ".join("%4d" % v for v in cm[i])))
    for i in range(k):
        sup = int(cm[i].sum())
        npred = int(cm[:, i].sum())
        rec = cm[i, i] / sup if sup else float("nan")
        prec = cm[i, i] / npred if npred else float("nan")
        out.append("%s: support %d predicted %d recall %.3f precision %.3f" % (classes[i], sup, npred, rec, prec))

    ok = true == pred
    out.append("mean confidence: correct %.3f wrong %.3f" % (conf_all[keep][ok].mean(), conf_all[keep][~ok].mean() if (~ok).any() else float("nan")))
    mp = np.bincount(pred_all[mixed], minlength=k)
    out.append("mixed tracks predicted counts: " + str(dict(zip(classes, mp.tolist()))))
    out.append("mean confidence mixed %.3f non-mixed %.3f" % (conf_all[mixed].mean(), conf_all[keep].mean()))
    out.append("hand labels are one labeller's reading, at most 6 frames per track")

    text = "\n".join(out)
    print(text)
    Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
