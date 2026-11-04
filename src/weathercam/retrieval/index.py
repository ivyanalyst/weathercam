from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray


@runtime_checkable
class VectorIndex(Protocol):
    def add(self, vectors: np.ndarray) -> None: ...

    def search(
        self, queries: np.ndarray, k: int, exclude: np.ndarray | None = None
    ) -> SearchResult: ...

    def __len__(self) -> int: ...


def _normalised(vectors: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("vectors must be two-dimensional")
    if array.shape[1] != dim:
        raise ValueError(f"vectors must have {dim} columns")
    if not np.all(np.isfinite(array)):
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("vectors must not be zero")
    return (array / norms[:, None]).astype(np.float32)


class BruteForceIndex:
    def __init__(self, dim: int, chunk_size: int = 256) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self._dim = dim
        self._chunk_size = chunk_size
        self._blocks: list[np.ndarray] = []
        self._size = 0

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return self._size

    def add(self, vectors: np.ndarray) -> None:
        block = _normalised(vectors, self._dim)
        if block.shape[0] == 0:
            return
        self._blocks.append(block)
        self._size += block.shape[0]

    def _matrix(self) -> np.ndarray:
        if len(self._blocks) > 1:
            self._blocks = [np.concatenate(self._blocks, axis=0)]
        return self._blocks[0]

    def search(
        self, queries: np.ndarray, k: int, exclude: np.ndarray | None = None
    ) -> SearchResult:
        if k < 1:
            raise ValueError("k must be at least 1")
        if self._size == 0:
            raise ValueError("the index is empty")
        limit = self._size - 1 if exclude is not None else self._size
        if k > limit:
            raise ValueError("k is larger than the number of available results")
        normalised = _normalised(queries, self._dim)
        count = normalised.shape[0]
        skip = None
        if exclude is not None:
            skip = np.asarray(exclude)
            if not np.issubdtype(skip.dtype, np.integer):
                raise ValueError("exclude must contain integers")
            if skip.shape != (count,):
                raise ValueError("exclude must have one entry per query")
            if np.any(skip < -1) or np.any(skip >= self._size):
                raise ValueError("exclude entries must be -1 or a valid gallery index")
        matrix = self._matrix()
        indices = np.empty((count, k), dtype=np.int64)
        scores = np.empty((count, k), dtype=np.float32)
        for start in range(0, count, self._chunk_size):
            stop = min(start + self._chunk_size, count)
            block = normalised[start:stop] @ matrix.T
            if skip is not None:
                chunk_skip = skip[start:stop]
                rows = np.nonzero(chunk_skip >= 0)[0]
                block[rows, chunk_skip[rows]] = -np.inf
            order = np.argsort(-block, axis=1, kind="stable")[:, :k]
            indices[start:stop] = order
            scores[start:stop] = np.take_along_axis(block, order, axis=1)
        return SearchResult(indices, scores)
