from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from weathercam.retrieval.index import VectorIndex
from weathercam.retrieval.metrics import RetrievalScore, average_precision_at_r


@dataclass(frozen=True)
class QueryScores:
    precision_at_1: np.ndarray
    average_precision: np.ndarray
    relevant: np.ndarray


@dataclass(frozen=True)
class GroupScore:
    precision_at_1: float | None
    map_at_r: float | None
    queries: int
    scored: int
    skipped: int
    vehicles: int


def query_scores(
    index: VectorIndex,
    queries: np.ndarray,
    labels: Sequence | np.ndarray,
    block_size: int = 512,
) -> QueryScores:
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    queries = np.asarray(queries)
    if queries.ndim != 2:
        raise ValueError("queries must be two-dimensional")
    count = queries.shape[0]
    if count == 0:
        raise ValueError("no queries")
    if len(index) != count:
        raise ValueError("the index must hold exactly the queries")
    label_array = np.asarray(labels)
    if label_array.shape != (count,):
        raise ValueError("labels must have one entry per query")
    _, codes, sizes = np.unique(label_array, return_inverse=True, return_counts=True)
    codes = codes.reshape(-1)
    relevant = (sizes[codes] - 1).astype(np.int64)
    if relevant.max() == 0:
        raise ValueError("every query was skipped")
    depth = int(relevant.max())
    first = np.zeros(count, dtype=np.float64)
    average = np.zeros(count, dtype=np.float64)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        result = index.search(queries[start:stop], depth, exclude=np.arange(start, stop))
        hits = codes[result.indices] == codes[start:stop, None]
        for row in range(stop - start):
            needed = int(relevant[start + row])
            if needed == 0:
                continue
            first[start + row] = 1.0 if hits[row, 0] else 0.0
            average[start + row] = average_precision_at_r(hits[row], needed)
    return QueryScores(first, average, relevant)


def overall_score(scores: QueryScores) -> RetrievalScore:
    mask = scores.relevant > 0
    scored = int(mask.sum())
    if scored == 0:
        raise ValueError("every query was skipped")
    return RetrievalScore(
        float(scores.precision_at_1[mask].mean()),
        float(scores.average_precision[mask].mean()),
        scored,
        int(mask.shape[0] - scored),
    )


def group_scores(
    scores: QueryScores,
    groups: Sequence,
    vehicle_ids: Sequence,
) -> dict:
    count = scores.relevant.shape[0]
    group_list = list(groups)
    vehicle_list = list(vehicle_ids)
    if len(group_list) != count or len(vehicle_list) != count:
        raise ValueError("groups and vehicle_ids must have one entry per query")
    members: dict = {}
    for position, group in enumerate(group_list):
        members.setdefault(group, []).append(position)
    result = {}
    for group, positions in members.items():
        rows = np.asarray(positions)
        mask = scores.relevant[rows] > 0
        scored = int(mask.sum())
        if scored:
            first = float(scores.precision_at_1[rows][mask].mean())
            average = float(scores.average_precision[rows][mask].mean())
        else:
            first = None
            average = None
        vehicles = len({vehicle_list[position] for position in positions})
        result[group] = GroupScore(first, average, len(positions), scored, len(positions) - scored, vehicles)
    return result
