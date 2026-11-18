"""Build text-query ground truth from VeRi test_label.xml. Read-only."""
import argparse
import collections
import json
import os
import re
import sys
import zipfile

import numpy as np

COLORS = {1: "yellow", 2: "orange", 3: "green", 4: "gray", 5: "red",
          6: "blue", 7: "white", 8: "golden", 9: "brown", 10: "black"}
TYPES = {1: "sedan", 2: "SUV", 3: "van", 4: "hatchback", 5: "MPV",
         6: "pickup", 7: "bus", 8: "truck", 9: "estate"}
EXPECT_COLOR = {"black": 681, "white": 434, "gray": 308, "red": 191,
                "green": 127, "yellow": 126, "blue": 55, "golden": 32,
                "brown": 25, "orange": 21}
EXPECT_TYPE = {"sedan": 887, "SUV": 434, "truck": 301, "hatchback": 139,
               "pickup": 122, "bus": 53, "van": 47, "estate": 17}
SINGLE = ["white", "black", "gray", "red", "yellow", "green", "blue",
          "sedan", "SUV", "truck", "pickup", "hatchback", "bus", "van"]
COMBINED = ["white SUV", "black sedan", "red hatchback", "green pickup",
            "yellow truck", "white truck", "blue bus"]


def parse_items(raw):
    text = raw.decode("gb2312", errors="ignore")
    return [dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            for m in re.finditer(r"<Item\s+([^>]*?)/?>", text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--veri-zip", required=True)
    ap.add_argument("--out", default="scripts/queries.json")
    a = ap.parse_args()

    names = [os.path.basename(str(n)) for n in np.load(a.npz)["names"]]
    z = zipfile.ZipFile(a.veri_zip)
    meta = {os.path.basename(i["imageName"]): i
            for i in parse_items(z.read("VeRi/test_label.xml"))}
    lab = {}
    for n in names:
        m = meta[n]
        lab[n] = (COLORS[int(m["colorID"])], TYPES[int(m["typeID"])])

    cc = collections.Counter(c for c, _ in lab.values())
    tc = collections.Counter(t for _, t in lab.values())
    print("colors:", dict(cc))
    print("types:", dict(tc))
    bad = [(k, v, cc.get(k, 0)) for k, v in EXPECT_COLOR.items()
           if cc.get(k, 0) != v]
    bad += [(k, v, tc.get(k, 0)) for k, v in EXPECT_TYPE.items()
            if tc.get(k, 0) != v]
    if bad:
        print("MISMATCH vs handoff counts (name, expected, got):", bad)
        sys.exit(1)

    queries = []
    for q in SINGLE + COMBINED:
        parts = q.split()
        want_c = parts[0] if parts[0] in EXPECT_COLOR else None
        want_t = parts[-1] if parts[-1] in EXPECT_TYPE else None
        hits = [n for n in names
                if (want_c is None or lab[n][0] == want_c)
                and (want_t is None or lab[n][1] == want_t)]
        queries.append({"text": q, "color": want_c, "type": want_t,
                        "n_match": len(hits), "matches": hits})
        print(f"{q:16s} matches={len(hits)}")

    out = {"n_gallery": len(names), "colors": dict(cc), "types": dict(tc),
           "queries": queries}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
