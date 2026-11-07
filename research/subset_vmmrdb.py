import argparse
import collections
import json
import os
import zipfile

from weathercam.data.split import group_split
from weathercam.data.vmmr import (
    capped_indices,
    cross_make_drop,
    encode_make,
    group_ids,
    make_list,
    make_of,
)

PARTS = ("train", "val", "test")
FRACTIONS = (0.7, 0.15, 0.15)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=800)
    p.add_argument("--min-images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    z = zipfile.ZipFile(args.zip)
    infos = [i for i in z.infolist() if not i.filename.endswith("/")]
    members = [(i.filename, i.CRC, i.file_size) for i in infos]
    gids = group_ids(members)
    dropped = set(cross_make_drop(members))
    makes_in = {}
    for index, (path, _, _) in enumerate(members):
        if index not in dropped:
            makes_in.setdefault(gids[index], set()).add(make_of(path))
    mixed = {g for g, found in makes_in.items() if len(found) > 1}
    keep = [i for i in range(len(members)) if i not in dropped and gids[i] not in mixed]
    print("files:", len(members), "cross-make dropped:", len(dropped))
    print("mixed groups dropped:", len(mixed), "files kept:", len(keep))
    counts = collections.Counter(make_of(members[i][0]) for i in keep)
    makes = make_list(counts, args.min_images)
    print("classes:", len(makes), "(incl. unknown)")
    names = [makes[encode_make(make_of(members[i][0]), makes)] for i in keep]
    kept_groups = [gids[i] for i in keep]
    chosen = capped_indices(names, kept_groups, args.cap, args.seed)
    sel = [keep[k] for k in chosen]
    sel_names = [names[k] for k in chosen]
    sel_groups = [kept_groups[k] for k in chosen]
    parts = group_split(sel_groups, sel_names, FRACTIONS, args.seed)
    part_of = {}
    for p, indices in enumerate(parts):
        for k in indices:
            part_of[k] = PARTS[p]
    print("selected:", len(sel), {PARTS[p]: len(v) for p, v in enumerate(parts)})
    os.makedirs(args.out, exist_ok=False)
    images = os.path.join(args.out, "images")
    items = []
    for k, index in enumerate(sel):
        z.extract(infos[index], images)
        items.append(
            {
                "path": infos[index].filename,
                "make_id": makes.index(sel_names[k]),
                "group": sel_groups[k],
                "split": part_of[k],
            }
        )
        if (k + 1) % 5000 == 0:
            print("extracted", k + 1)
    manifest = {
        "makes": makes,
        "cap": args.cap,
        "min_images": args.min_images,
        "seed": args.seed,
        "fractions": list(FRACTIONS),
        "items": items,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    table = collections.defaultdict(collections.Counter)
    for k in range(len(sel)):
        table[sel_names[k]][part_of[k]] += 1
    for make in makes:
        row = table[make]
        print(make, [row[name] for name in PARTS])
    print("extracted:", len(sel))


if __name__ == "__main__":
    main()
