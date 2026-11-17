"""Score every set in eval/sets.json through POST /ood and compute metrics.

Positive class for AUROC/AUPR = OOD (higher score = more OOD).
Two operating points are reported, because "FPR at 95% TPR" is used both ways:
  ood_accepted_at_95_id_kept : threshold keeps 95% of ID images; fraction of OOD scored below it.
  id_flagged_at_95_ood_caught: threshold catches 95% of OOD images; fraction of ID scored above it.
"""
import argparse, base64, csv, json, os, time, urllib.request, zipfile
import numpy as np
import corrupt

BATCH = 16


def post(url, images):
    req = urllib.request.Request(url + "/ood", json.dumps({"images": images}).encode(),
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    assert len(r["scores"]) == len(images), r
    return r["scores"], r["k"]


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auroc(neg, pos):
    r = ranks(np.concatenate([neg, pos]))
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def aupr(neg, pos):
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    y = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def compare(neg, pos, rng, boot):
    out = {"n_id": len(neg), "n_ood": len(pos), "auroc": auroc(neg, pos), "aupr": aupr(neg, pos),
           "aupr_chance": len(pos) / (len(pos) + len(neg)),
           "ood_accepted_at_95_id_kept": float(np.mean(pos <= np.quantile(neg, 0.95))),
           "id_flagged_at_95_ood_caught": float(np.mean(neg >= np.quantile(pos, 0.05)))}
    b = [auroc(rng.choice(neg, len(neg)), rng.choice(pos, len(pos))) for _ in range(boot)]
    out["auroc_ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="eval/sets.json")
    ap.add_argument("--veri-zip", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8014")
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--limit", type=int, default=0, help="first N entries per set (smoke test)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    man = json.load(open(a.sets))
    z = zipfile.ZipFile(a.veri_zip)

    def load(e):
        raw = z.read(e["ref"]) if e["kind"] == "veri_zip" else open(e["ref"], "rb").read()
        if "corruption" in e:
            raw = corrupt.apply(raw, e["corruption"], e["severity"], man["seed"], e["ref"])
        return base64.b64encode(raw).decode()

    scores, rows, k = {}, [], None
    t0 = time.time()
    for name, s in man["sets"].items():
        ents = s["entries"][: a.limit] if a.limit else s["entries"]
        vals = []
        for i in range(0, len(ents), BATCH):
            sc, k = post(a.url, [load(e) for e in ents[i:i + BATCH]])
            vals += sc
        scores[name] = np.array(vals)
        rows += [(name, s["role"], e["ref"], v) for e, v in zip(ents, vals)]
        print(f"{name:20s} n={len(vals):4d} median={np.median(vals):.4f}  [{time.time()-t0:.0f}s]", flush=True)

    with open(os.path.join(a.out_dir, "ood_scores.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "role", "ref", "score"]); w.writerows(rows)

    allv = np.concatenate(list(scores.values()))
    edges = np.linspace(allv.min(), allv.max(), 21)
    dist = {n: {"n": len(v), "mean": float(v.mean()),
                "quantiles": {str(q): float(np.quantile(v, q / 100)) for q in (0, 5, 25, 50, 75, 95, 100)},
                "hist": np.histogram(v, edges)[0].tolist()} for n, v in scores.items()}

    rng = np.random.default_rng(a.seed)
    refs = {n: [e["ref"] for e in man["sets"][n]["entries"][: a.limit or None]] for n in man["sets"]}
    comps = []
    for name, s in man["sets"].items():
        if s["role"] == "in":
            continue
        if s["role"] == "ood":
            targets = {"veri_id_new": scores["veri_id_new"], "veri_id_same": scores["veri_id_same"]}
        else:  # shift: reference = id_new minus the images that were corrupted (no pairing leak)
            used = set(refs[name])
            keep = [i for i, r in enumerate(refs["veri_id_new"]) if r not in used]
            targets = {"veri_id_new_minus_base": scores["veri_id_new"][keep]}
        for rname, neg in targets.items():
            if len(neg) < 10:
                print(f"skip {name} vs {rname}: reference n={len(neg)} < 10"); continue
            c = compare(neg, scores[name], rng, a.boot)
            c.update({"ood_set": name, "role": s["role"], "reference": rname})
            if s["role"] == "shift":
                clean = dict(zip(refs["veri_id_new"], scores["veri_id_new"]))
                d = [v - clean[r] for r, v in zip(refs[name], scores[name]) if r in clean]
                c["paired_median_delta"] = float(np.median(d)) if d else None
            comps.append(c)

    json.dump({"k": k, "seed": a.seed, "limit": a.limit, "boot": a.boot,
               "distributions": dist, "comparisons": comps},
              open(os.path.join(a.out_dir, "ood_metrics.json"), "w"), indent=1)
    print(f"\n{'set':16s} {'vs':22s} {'n_id':>5s} {'n_ood':>5s} {'AUROC':>6s} {'95% CI':>15s} {'AUPR':>6s} {'chance':>6s} {'OODacc':>6s} {'IDflag':>6s}")
    for c in comps:
        print(f"{c['ood_set']:16s} {c['reference']:22s} {c['n_id']:5d} {c['n_ood']:5d} {c['auroc']:6.3f} "
              f"[{c['auroc_ci95'][0]:.3f},{c['auroc_ci95'][1]:.3f}] {c['aupr']:6.3f} {c['aupr_chance']:6.3f} "
              f"{c['ood_accepted_at_95_id_kept']:6.3f} {c['id_flagged_at_95_ood_caught']:6.3f}")


if __name__ == "__main__":
    main()
