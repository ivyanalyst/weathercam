"""Read-only: show what labels exist and what is checkable. No measurements, writes nothing."""
import argparse, collections, json, os, re, zipfile
from pathlib import Path
import numpy as np

# Standard VeRi-776 ID maps. Verify against the label file's own README if present.
COLOR = {1: "yellow", 2: "orange", 3: "green", 4: "gray", 5: "red", 6: "blue",
         7: "white", 8: "golden", 9: "brown", 10: "black"}
TYPE = {1: "sedan", 2: "SUV", 3: "van", 4: "hatchback", 5: "MPV", 6: "pickup",
        7: "bus", 8: "truck", 9: "estate"}


def parse_items(raw: bytes):
    text = raw.decode("gb2312", errors="ignore")
    items = []
    for m in re.finditer(r"<Item\s+([^>]*?)/?>", text):
        items.append(dict(re.findall(r'(\w+)="([^"]*)"', m.group(1))))
    return items


ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--veri-zip", required=True)
ap.add_argument("--ccpd-manifest", required=True)
ap.add_argument("--vmmr-dir", required=True)
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
names = [os.path.basename(str(n)) for n in d["names"]]
print("gallery size:", len(names), "unique:", len(set(names)))

z = zipfile.ZipFile(a.veri_zip)
members = z.namelist()
print("zip top-level entries:", sorted({m.split("/")[0] for m in members})[:10])
split_names = {}
for s in ("train", "test", "query"):
    pre = f"VeRi/image_{s}/"
    split_names[s] = {m[len(pre):] for m in members if m.startswith(pre) and m.endswith(".jpg")}
    print(f"image_{s}: {len(split_names[s])} jpgs")
print("label files present:", [m for m in members if m.endswith("_label.xml")])

meta = {}
for s in ("train", "test"):
    p = f"VeRi/{s}_label.xml"
    if p in members:
        items = parse_items(z.read(p))
        print(p, "items:", len(items), "| attribute keys:", sorted(items[0].keys()) if items else None)
        for it in items:
            meta[os.path.basename(it.get("imageName", ""))] = {**it, "split": s}

in_split = collections.Counter(
    next((s for s in ("train", "test", "query") if n in split_names[s]), "none") for n in names)
print("gallery images by zip folder:", dict(in_split))
found = [n for n in names if n in meta]
print(f"gallery names with labels: {len(found)}/{len(names)}")
col = lambda n: COLOR.get(int(meta[n]["colorID"]))
typ = lambda n: TYPE.get(int(meta[n]["typeID"]))
print("gallery color:", dict(collections.Counter(col(n) for n in found)))
print("gallery type:", dict(collections.Counter(typ(n) for n in found)))
combo = collections.Counter((col(n), typ(n)) for n in found)
print("color+type pairs with >=10 gallery images:", {f"{c} {t}": v for (c, t), v in combo.items() if v >= 10})
gids = {meta[n]["vehicleID"] for n in found}
print("gallery distinct vehicle IDs:", len(gids))

gset = set(names)
for s in ("test", "train"):
    pool = [n for n in split_names[s] if n not in gset]
    pool_new = [n for n in pool if n in meta and meta[n]["vehicleID"] not in gids]
    print(f"held-out pool from image_{s}: {len(pool)} excluding gallery names, "
          f"{len(pool_new)} also excluding gallery IDs")

man = json.load(open(a.ccpd_manifest))
items = man["items"]
print("CCPD items:", len(items), "| keys:", sorted(items[0].keys()))
print("CCPD by subset:", dict(collections.Counter(i["subset"] for i in items)))
print("CCPD by part:", dict(collections.Counter(i["part"] for i in items)))

n = sum(1 for _, _, fs in os.walk(a.vmmr_dir) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png")))
print("VMMR images:", n, "| classes:", len([p for p in Path(a.vmmr_dir).iterdir() if p.is_dir()]))
