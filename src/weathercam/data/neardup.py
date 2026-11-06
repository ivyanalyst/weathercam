import numpy as np
from PIL import Image


def hash_distance(a, b):
    return (a ^ b).bit_count()


def dhash(image, size=16):
    if size < 1:
        raise ValueError(f"size must be positive: {size}")
    small = image.convert("L").resize((size + 1, size), Image.Resampling.BOX)
    pixels = np.asarray(small, dtype=np.int16)
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | int(pixels[row, col] > pixels[row, col + 1])
    return bits


def hash_file(path, size=16):
    with Image.open(path) as image:
        image.draft("L", ((size + 1) * 4, size * 4))
        return dhash(image, size)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        low, high = min(root_a, root_b), max(root_a, root_b)
        self.parent[high] = low
        return True


def group_by_distance(hashes, threshold):
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative: {threshold}")
    forest = UnionFind(len(hashes))
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hash_distance(hashes[i], hashes[j]) <= threshold:
                forest.union(i, j)
    return [forest.find(i) for i in range(len(hashes))]
