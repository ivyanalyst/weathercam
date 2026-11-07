import argparse

import numpy as np

from weathercam.retrieval.evaluate import cross_vehicle_scores, overall_score
from weathercam.retrieval.index import BruteForceIndex


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--split", default="test")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.embeddings)
    rows = data["split"] == args.split
    x = data["embeddings"][rows]
    vehicles = data["vehicle_ids"][rows]
    index = BruteForceIndex(x.shape[1])
    index.add(x)
    print("split", args.split, "images", len(x), "vehicles", len(set(vehicles.tolist())))
    for attr in ("color", "type"):
        scores = cross_vehicle_scores(index, x, data[f"{attr}_ids"][rows], vehicles)
        s = overall_score(scores)
        print(attr, "P@1", round(s.precision_at_1, 4), "mAP@R", round(s.map_at_r, 4), "scored", s.scored, "skipped", s.skipped)


if __name__ == "__main__":
    main()
