"""Build seeded labelled sets for phase 15. Writes a manifest only; copies no images.

Roles: "in" = in-distribution held-out, "ood" = other-source images,
"shift" = corrupted held-out VeRi (a distribution shift, reported separately).
"""
import argparse, collections, json, os, random, re, zipfile
import numpy as np

CORRUPTIONS = {"noise": (1, 2), "blur": (1, 2), "jpeg": (1, 2), "dark": (1, 2), "fog": (1, 2)}
CCPD_SUBSETS = ("ccpd_base", "ccpd_weather", "ccpd_blur", "ccpd_challenge")


def parse_items(raw: bytes):
    text = raw.decode("gb2312", errors="ignore")
    return [dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            for m in re.finditer(r"<Item\s+([^>]*?)/?>", text)]


def veri_entry(split, name, meta):
    m = meta[name]
    return {"kind": "veri_zip", "ref": f"VeRi/image_{split}/{name}", "vehicleID": m["vehicleID"],
            "cameraID": m["cameraID"], "colorID": m["colorID"], "typeID": m["typeID"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--veri-zip", required=True)
    ap.add_argument("--ccpd-root", required=True)
    ap.add_argument("--vmmr-root", required=True)
    ap.add_argument("--out", default="eval/sets.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-corrupt", type=int, default=200)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    gallery = [os.path.basename(str(n)) for n in np.load(a.npz)["names"]]
    gset = set(gallery)
    z = zipfile.ZipFile(a.veri_zip)
    members = set(z.namelist())
    meta = {}
    for split in ("train", "test"):
        meta[split] = {os.path.basename(i["imageName"]): i
                       for i in parse_items(z.read(f"VeRi/{split}_label.xml"))}
    present = {s: {m.rsplit("/", 1)[1] for m in members
                   if m.startswith(f"VeRi/image_{s}/") and m.endswith(".jpg")} for s in ("train", "test")}

    gids = {meta["test"][n]["vehicleID"] for n in gallery}
    pool_same = sorted(n for n in present["test"] if n in meta["test"] and n not in gset)
    pool_new = sorted(n for n in present["train"] if n in meta["train"]
                      and meta["train"][n]["vehicleID"] not in gids)
    id_same = rng.sample(pool_same, a.n)
    id_new = rng.sample(pool_new, a.n)

    sets = {
        "veri_id_new": {"role": "in", "entries": [veri_entry("train", n, meta["train"]) for n in id_new]},
        "veri_id_same": {"role": "in", "entries": [veri_entry("test", n, meta["test"]) for n in id_same]},
    }

    man = json.load(open(os.path.join(a.ccpd_root, "manifest.json")))
    for sub in CCPD_SUBSETS:
        files = sorted(i["file"] for i in man["items"] if i["subset"] == sub)
        pick = rng.sample(files, min(a.n, len(files)))
        sets[sub] = {"role": "ood", "entries": [
            {"kind": "file", "ref": os.path.join(a.ccpd_root, f)} for f in pick]}

    vroot = os.path.join(a.vmmr_root, "images")
    vfiles = sorted(os.path.join(d, f) for d, _, fs in os.walk(vroot) for f in fs
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    sets["vmmr"] = {"role": "ood", "entries": [
        {"kind": "file", "ref": p} for p in rng.sample(vfiles, a.n)]}

    base = id_new[: a.n_corrupt]
    for cname, sevs in CORRUPTIONS.items():
        for sev in sevs:
            sets[f"veri_{cname}_s{sev}"] = {"role": "shift", "entries": [
                {**veri_entry("train", n, meta["train"]), "corruption": cname, "severity": sev}
                for n in base]}

    # sanity checks
    assert not gset & set(id_same) and not gset & set(id_new)
    assert not {meta["train"][n]["vehicleID"] for n in id_new} & gids
    for s in sets.values():
        for e in s["entries"]:
            if e["kind"] == "file":
                assert os.path.exists(e["ref"]), e["ref"]
            else:
                assert e["ref"] in members, e["ref"]

    out = {"seed": a.seed, "args": {"n": a.n, "n_corrupt": a.n_corrupt},
           "pools": {"id_same": len(pool_same), "id_new": len(pool_new), "vmmr_files": len(vfiles)},
           "sets": sets}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}")
    for name, s in sets.items():
        print(f"{name:20s} role={s['role']:6s} n={len(s['entries'])}")
    print("pools:", out["pools"])


if __name__ == "__main__":
    main()
