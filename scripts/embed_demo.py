"""Build the demo gallery: embed demo images and pack them in a zip.

Embeddings come from research/embed_worker.py, run by a separate Python
that has torch and OpenCLIP (this environment has no torch). The zip
uses the layout the service expects for thumbnails:
VeRi/image_test/<name>. It holds the demo images only, no VeRi data.
Run from the repository root.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

from weathercam.data.veri import ZIP_ROOT
from weathercam.service.worker_client import WorkerClient

EMBED_DIM = 512
BATCH = 16
NORM_TOLERANCE = 1e-3


def list_images(folder):
    files = sorted(Path(folder).glob("demo_*.jpg"))
    if not files:
        sys.exit("no demo_*.jpg in " + str(folder))
    return files


def embed(files, python, timeout):
    command = ["nice", "-n", "10", python, "research/embed_worker.py"]
    with WorkerClient(command, timeout, timeout) as client:
        if client.dim != EMBED_DIM:
            sys.exit("worker dim %d, need %d" % (client.dim, EMBED_DIM))
        print("worker ready:", client.model, client.pretrained, flush=True)
        parts = []
        for start in range(0, len(files), BATCH):
            blobs = [p.read_bytes() for p in files[start:start + BATCH]]
            parts.append(client.encode_images(blobs))
            print("embedded", min(start + BATCH, len(files)), flush=True)
    vectors = np.concatenate(parts).astype(np.float32)
    if vectors.shape != (len(files), EMBED_DIM):
        sys.exit("bad shape " + str(vectors.shape))
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.abs(norms - 1.0).max() > NORM_TOLERANCE:
        sys.exit("embeddings are not unit length")
    return vectors


def pack_zip(files, path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, "%s/image_test/%s" % (ZIP_ROOT, p.name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="demo/images")
    ap.add_argument("--out-npz", default="demo/gallery_demo.npz")
    ap.add_argument("--out-zip", default="demo/demo_images.zip")
    ap.add_argument("--embed-worker-python")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--zip-only", action="store_true")
    args = ap.parse_args()
    if not args.zip_only and not args.embed_worker_python:
        sys.exit("give --embed-worker-python, or --zip-only")
    outputs = [args.out_zip] if args.zip_only else [args.out_npz, args.out_zip]
    for out in outputs:
        if Path(out).exists():
            sys.exit("refusing to overwrite " + out)
    files = list_images(args.images)
    if not args.zip_only:
        vectors = embed(files, args.embed_worker_python, args.timeout)
        names = np.array([p.name for p in files])
        np.savez(args.out_npz, names=names, embeddings=vectors)
        print("wrote", args.out_npz, vectors.shape)
    pack_zip(files, args.out_zip)
    print("wrote", args.out_zip, len(files), "images")


if __name__ == "__main__":
    main()
