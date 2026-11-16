import argparse
import sys
from pathlib import Path

import numpy as np

EMBED_DIM = 512
DEFAULT_EMBEDDINGS = "outputs/embeddings/veri_vitb16_laion2b.npz"
KEYS = ("image_names", "embeddings", "split")
NORM_TOLERANCE = 1e-3


def build(embeddings_path, split, count, seed):
    with np.load(embeddings_path, allow_pickle=False) as data:
        for key in KEYS:
            if key not in data.files:
                raise ValueError(f"{embeddings_path} has no {key}")
        names = data["image_names"]
        vectors = data["embeddings"]
        splits = data["split"]
    if vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
        raise ValueError(f"embeddings must have shape (n, {EMBED_DIM}), got {vectors.shape}")
    if not len(names) == len(vectors) == len(splits):
        raise ValueError(f"{len(names)} names, {len(vectors)} embeddings, {len(splits)} splits")
    if split == "all":
        rows = np.arange(len(names))
    else:
        rows = np.flatnonzero(splits == split)
    if rows.size == 0:
        raise ValueError(f"no rows with split {split}")
    if count is not None:
        if not 1 <= count <= rows.size:
            raise ValueError(f"count must be in [1, {rows.size}], got {count}")
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=count, replace=False))
    chosen_names = names[rows]
    if len(set(chosen_names.tolist())) != len(chosen_names):
        raise ValueError("duplicate names in the selected rows")
    chosen = vectors[rows].astype(np.float32)
    norms = np.linalg.norm(chosen.astype(np.float64), axis=1)
    if np.abs(norms - 1.0).max() > NORM_TOLERANCE:
        raise ValueError("embeddings are not unit length")
    return chosen_names, chosen


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    p.add_argument("--split", required=True, choices=("train", "test", "all"))
    p.add_argument("--count", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.suffix != ".npz":
        sys.exit("output must end with .npz")
    if output.exists():
        sys.exit(f"{output} exists")
    try:
        names, vectors = build(args.embeddings, args.split, args.count, args.seed)
    except ValueError as error:
        sys.exit(str(error))
    np.savez(output, names=names, embeddings=vectors)
    print(f"wrote {len(names)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
