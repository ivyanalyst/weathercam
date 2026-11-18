"""Evaluate POST /search/text against VeRi attribute ground truth."""
import argparse
import json
import urllib.request

import numpy as np

WEATHER = ["fog", "haze", "rain", "sand storm"]
BATCH = 16


def post(url, texts, k):
    body = json.dumps({"texts": texts, "k": k}).encode()
    req = urllib.request.Request(
        url + "/search/text", body,
        {"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    res = json.loads(raw)["results"]
    assert len(res) == len(texts)
    return res


def rank_all(url, texts, k):
    out = []
    for i in range(0, len(texts), BATCH):
        out += post(url, texts[i:i + BATCH], k)
    return out


def unpack(ranking, n):
    names = [r["name"] for r in ranking]
    sc = np.array([r["score"] for r in ranking])
    assert len(names) == n and len(set(names)) == n
    assert np.all(np.diff(sc) <= 1e-9), "not sorted by score"
    return names, sc


def score_query(q, ranking, n):
    names, sc = unpack(ranking, n)
    want = set(q["matches"])
    hit = np.array([x in want for x in names])
    m = int(hit.sum())
    assert m == q["n_match"]
    pos = np.flatnonzero(hit) + 1
    auc = 1 - (pos.sum() - m * (m + 1) / 2) / (m * (n - m))
    return {
        "text": q["text"], "n_match": m, "baseline": m / n,
        "p1": float(hit[:1].mean()), "p5": float(hit[:5].mean()),
        "p10": float(hit[:10].mean()), "mrr": float(1 / pos[0]),
        "first_rank": int(pos[0]), "auroc": float(auc),
        "vehicles_top10": len({x[:4] for x in names[:10]}),
        "score_hit_mean": float(sc[hit].mean()),
        "score_miss_mean": float(sc[~hit].mean()),
        "p10_too_small": m < 10,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="scripts/queries.json")
    ap.add_argument("--url", default="http://127.0.0.1:8014")
    ap.add_argument("--out", default="eval/text_metrics.json")
    a = ap.parse_args()
    d = json.load(open(a.queries))
    n = d["n_gallery"]
    texts = [q["text"] for q in d["queries"]] + WEATHER
    res = rank_all(a.url, texts, n)

    rows = [score_query(q, res[i], n)
            for i, q in enumerate(d["queries"])]
    weather = []
    for j, t in enumerate(WEATHER):
        _, sc = unpack(res[len(d["queries"]) + j], n)
        weather.append({
            "text": t, "status": "unscored", "top1": float(sc[0]),
            "rank10": float(sc[9]), "median": float(np.median(sc)),
            "min": float(sc[-1])})

    head = (f"{'query':14s} {'m':>4s} {'base':>5s} {'P@1':>4s} "
            f"{'P@5':>4s} {'P@10':>5s} {'MRR':>5s} {'AUROC':>5s} "
            f"{'veh10':>5s} {'hit':>6s} {'miss':>6s}")
    print(head)
    for r in rows:
        flag = " (m<10)" if r["p10_too_small"] else ""
        print(f"{r['text']:14s} {r['n_match']:4d} {r['baseline']:5.3f} "
              f"{r['p1']:4.1f} {r['p5']:4.1f} {r['p10']:5.2f} "
              f"{r['mrr']:5.3f} {r['auroc']:5.3f} "
              f"{r['vehicles_top10']:5d} {r['score_hit_mean']:6.3f} "
              f"{r['score_miss_mean']:6.3f}{flag}")
    print("\nweather (unscored): text top1 rank10 median min")
    for w in weather:
        print(f"  {w['text']:11s} {w['top1']:.3f} {w['rank10']:.3f} "
              f"{w['median']:.3f} {w['min']:.3f}")
    with open(a.out, "w") as f:
        json.dump({"n_gallery": n, "queries": rows,
                   "weather_unscored": weather}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
