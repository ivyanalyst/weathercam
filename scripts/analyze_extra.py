"""Read-only extras: deltas, per-set thresholds, image sizes."""
import collections
import csv
import io
import json
import zipfile

import numpy as np
from PIL import Image

ZIP = "/path/to/VeRi.zip"
CLEAN = ["veri_id_same", "veri_id_val"]
THR = (1.00, 1.03, 1.05, 1.08)

m = json.load(open("eval/ood_metrics.json"))
print("k", m["k"], "seed", m["seed"], "limit", m["limit"],
      "boot", m["boot"])
print("\nshift sets: AUROC and paired median delta (same image)")
for c in m["comparisons"]:
    if c["role"] == "shift":
        d = c["paired_median_delta"]
        print(f"{c['ood_set']:16s} auroc={c['auroc']:.3f} "
              f"delta={d:+.4f}")

by = collections.defaultdict(list)
with open("eval/ood_scores.csv") as f:
    for r in csv.DictReader(f):
        by[r["set"]].append(float(r["score"]))

print("\nper-set rates at each threshold", THR)
print("clean sets: fraction flagged (score > t)")
for n in CLEAN:
    s = np.array(by[n])
    print(f"  {n:16s}", [round(float((s > t).mean()), 3) for t in THR])
print("OOD sets: fraction accepted (score <= t)")
for n in ("ccpd_base", "ccpd_weather", "ccpd_blur",
          "ccpd_challenge", "vmmr"):
    s = np.array(by[n])
    print(f"  {n:16s}", [round(float((s <= t).mean()), 3) for t in THR])

sets = json.load(open("eval/sets.json"))["sets"]
z = zipfile.ZipFile(ZIP)
print("\nimage sizes (w x h): median w, median h, median aspect w/h")
for name, s in sets.items():
    if s["role"] == "shift":
        continue
    ws, hs = [], []
    for e in s["entries"]:
        if e["kind"] == "veri_zip":
            im = Image.open(io.BytesIO(z.read(e["ref"])))
        else:
            im = Image.open(e["ref"])
        ws.append(im.width)
        hs.append(im.height)
    ws, hs = np.array(ws), np.array(hs)
    print(f"  {name:16s} n={len(ws)} w={np.median(ws):.0f} "
          f"h={np.median(hs):.0f} aspect={np.median(ws / hs):.2f} "
          f"pixels min={int((ws * hs).min())} "
          f"max={int((ws * hs).max())}")
