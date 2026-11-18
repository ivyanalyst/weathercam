"""Read-only: inspect rankings of four weak text queries."""
import collections
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, "scripts")
import build_queries as bq
import eval_text as et

ZIP = "/path/to/VeRi.zip"
GAL = os.path.expanduser("/path/to/gallery_2000.npz")
Q = ["gray", "green", "hatchback", "white truck"]

names = [os.path.basename(str(n)) for n in np.load(GAL)["names"]]
z = zipfile.ZipFile(ZIP)
meta = {os.path.basename(i["imageName"]): i
        for i in bq.parse_items(z.read("VeRi/test_label.xml"))}
lab = {n: (bq.COLORS[int(meta[n]["colorID"])],
           bq.TYPES[int(meta[n]["typeID"])]) for n in names}

res = et.post("http://127.0.0.1:8014", Q, len(names))
for q, ranking in zip(Q, res):
    rn = [r["name"] for r in ranking]
    print(f"\n=== {q}: top 10 (name score colour type)")
    for r in ranking[:10]:
        c, t = lab[r["name"]]
        print(f"  {r['name']} {r['score']:.3f} {c} {t}")
    for pos, label in ((0, "colour"), (1, "type")):
        g = collections.defaultdict(list)
        for i, n in enumerate(rn):
            g[lab[n][pos]].append(i + 1)
        rows = sorted((np.mean(v), k, len(v)) for k, v in g.items())
        print(f"  mean rank by true {label} (1 = best):")
        print("   ", ", ".join(f"{k} {mr:.0f} (n={c})"
                                for mr, k, c in rows))
