"""Read-only look at eval/ood_scores.csv: bands, extremes, thresholds."""
import collections
import csv

import numpy as np

BANDS = [(1.04, 1.07), (1.13, 1.24)]
CLEAN = ["veri_id_same", "veri_id_val"]
OOD = ["ccpd_base", "ccpd_weather", "ccpd_blur", "ccpd_challenge", "vmmr"]

by = collections.defaultdict(list)
with open("eval/ood_scores.csv") as f:
    for r in csv.DictReader(f):
        by[r["set"]].append((float(r["score"]), r["ref"]))

print("set  n  min p05 p50 p95 max  counts in", BANDS)
for name, v in by.items():
    s = np.array([x[0] for x in v])
    q = np.quantile(s, [0, 0.05, 0.5, 0.95, 1])
    cnt = [int(((s >= lo) & (s <= hi)).sum()) for lo, hi in BANDS]
    nums = " ".join(f"{x:.3f}" for x in q)
    print(f"{name:16s} n={len(s):4d} {nums} bands={cnt}")


def pick(names, n, rev):
    pool = [(s, nm, r) for nm in names for s, r in by[nm]]
    return sorted(pool, reverse=rev)[:n]


print("\nhighest 10 clean VeRi")
for s, nm, r in pick(CLEAN, 10, True):
    print(f"  {s:.4f} {nm} {r}")
print("\nlowest 10 CCPD/VMMR")
for s, nm, r in pick(OOD, 10, False):
    print(f"  {s:.4f} {nm} {r}")
print("\nhighest 10 anywhere")
for s, nm, r in pick(list(by), 10, True):
    print(f"  {s:.4f} {nm} {r}")

idv = np.array([x[0] for nm in CLEAN for x in by[nm]])
ov = np.array([x[0] for nm in OOD for x in by[nm]])
print("\nin-sample thresholds: n_id", len(idv), "n_ood", len(ov))
print("thr  id_flagged  ood_accepted")
for t in (0.95, 1.00, 1.03, 1.05, 1.08, 1.10):
    print(f"{t:.2f}  {np.mean(idv > t):.3f}  {np.mean(ov <= t):.3f}")
