"""Build seeded labelled sets for phase 15. Writes a manifest only.

Roles: "in" = in-distribution, held out from the OOD bank,
"ood" = other-source images,
"shift" = corrupted or re-encoded VeRi test images (reported separately).
"""
import argparse
import json
import os
import random
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, "src")
from weathercam.retrieval.innersplit import veri_inner_split

BANK_SEED = 42  # must match scripts/run_service.py::load_ood
VERI_EMB = "outputs/embeddings/veri_vitb16_laion2b.npz"
CORRUPTIONS = {
    "noise": (1, 2),
    "blur": (1, 2),
    "jpeg": (1, 2),
    "dark": (1, 2),
    "fog": (1, 2),
    "recode": (1,),
}
CCPD_SUBSETS = ("ccpd_base", "ccpd_weather", "ccpd_blur", "ccpd_challenge")


def parse_items(raw):
    text = raw.decode("gb2312", errors="ignore")
    return [dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            for m in re.finditer(r"<Item\s+([^>]*?)/?>", text)]


def veri_entry(split, name, meta):
    m = meta[name]
    return {"kind": "veri_zip", "ref": f"VeRi/image_{split}/{name}",
            "vehicleID": m["vehicleID"], "cameraID": m["cameraID"],
            "colorID": m["colorID"], "typeID": m["typeID"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="gallery npz")
    ap.add_argument("--veri-emb", default=VERI_EMB)
    ap.add_argument("--veri-zip", required=True)
    ap.add_argument("--ccpd-root", required=True)
    ap.add_argument("--vmmr-root", required=True)
    ap.add_argument("--out", default="eval/sets.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-corrupt", type=int, default=200)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    gallery = np.load(a.npz)["names"]
    gset = {os.path.basename(str(n)) for n in gallery}
    emb = np.load(a.veri_emb, allow_pickle=False)
    data = {k: emb[k] for k in emb.files}
    fit, val = veri_inner_split(data, BANK_SEED)
    inames = [str(n) for n in data["image_names"]]
    bank = {inames[i] for i in fit}
    val_names = {inames[i] for i in val}

    z = zipfile.ZipFile(a.veri_zip)
    members = set(z.namelist())
    meta = {}
    for split in ("train", "test"):
        raw = z.read(f"VeRi/{split}_label.xml")
        meta[split] = {os.path.basename(i["imageName"]): i
                       for i in parse_items(raw)}
    present = {}
    for s in ("train", "test"):
        prefix = f"VeRi/image_{s}/"
        present[s] = {m[len(prefix):] for m in members
                      if m.startswith(prefix) and m.endswith(".jpg")}

    pool_same = sorted(n for n in present["test"]
                       if n in meta["test"] and n not in gset)
    pool_val = sorted(n for n in val_names
                      if n in present["train"] and n in meta["train"])
    id_same = rng.sample(pool_same, a.n)
    id_val = rng.sample(pool_val, a.n)

    sets = {
        "veri_id_same": {"role": "in", "entries": [
            veri_entry("test", n, meta["test"]) for n in id_same]},
        "veri_id_val": {"role": "in", "entries": [
            veri_entry("train", n, meta["train"]) for n in id_val]},
    }

    man = json.load(open(os.path.join(a.ccpd_root, "manifest.json")))
    for sub in CCPD_SUBSETS:
        files = sorted(i["file"] for i in man["items"] if i["subset"] == sub)
        pick = rng.sample(files, min(a.n, len(files)))
        sets[sub] = {"role": "ood", "entries": [
            {"kind": "file", "ref": os.path.join(a.ccpd_root, f)}
            for f in pick]}

    vroot = os.path.join(a.vmmr_root, "images")
    vfiles = sorted(os.path.join(d, f) for d, _, fs in os.walk(vroot)
                    for f in fs
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    sets["vmmr"] = {"role": "ood", "entries": [
        {"kind": "file", "ref": p} for p in rng.sample(vfiles, a.n)]}

    # base images = last n_corrupt of id_same; clean reference = the rest
    base = id_same[-a.n_corrupt:]
    for cname, sevs in CORRUPTIONS.items():
        for sev in sevs:
            sets[f"veri_{cname}_s{sev}"] = {"role": "shift", "entries": [
                {**veri_entry("test", n, meta["test"]),
                 "corruption": cname, "severity": sev} for n in base]}

    # sanity checks
    assert len(set(id_same)) == len(id_same) == a.n
    assert len(set(id_val)) == len(id_val) == a.n
    assert not gset & set(id_same)
    assert not bank & set(id_same)
    assert not bank & set(id_val)
    assert not set(id_same) & set(id_val)
    assert all(n in val_names for n in id_val)
    for s in sets.values():
        for e in s["entries"]:
            if e["kind"] == "file":
                assert os.path.exists(e["ref"]), e["ref"]
            else:
                assert e["ref"] in members, e["ref"]

    veh_same = {meta["test"][n]["vehicleID"] for n in id_same}
    veh_val = {meta["train"][n]["vehicleID"] for n in id_val}
    veh_base = {meta["test"][n]["vehicleID"] for n in base}
    out = {"seed": a.seed,
           "args": {"n": a.n, "n_corrupt": a.n_corrupt},
           "bank": {"seed": BANK_SEED, "fit": len(fit), "val": len(val)},
           "pools": {"id_same": len(pool_same), "id_val": len(pool_val),
                     "vmmr_files": len(vfiles)},
           "vehicles": {"id_same": len(veh_same), "id_val": len(veh_val),
                        "corrupt_base": len(veh_base)},
           "sets": sets}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}")
    for name, s in sets.items():
        print(f"{name:20s} role={s['role']:6s} n={len(s['entries'])}")
    for k in ("bank", "pools", "vehicles"):
        print(k + ":", out[k])


if __name__ == "__main__":
    main()
