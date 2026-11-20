import numpy as np

from weathercam.data.ccpd import from_latin_plate, to_latin_plate
from weathercam.plates.fuzzy import match_plate, plate_distance
from weathercam.retrieval.index import BruteForceIndex
from weathercam.retrieval.ood import knn_distances
from weathercam.service.heads import classify, standardise
from weathercam.service.plate_client import PlateResult

ATTRIBUTE_TOP_K = 3


def weather(head, embeddings):
    return classify(head, embeddings, top_k=len(head.class_ids))


def attributes(heads, embeddings):
    if not heads:
        raise ValueError("heads must not be empty")
    columns = {
        name: classify(
            head, embeddings, top_k=min(ATTRIBUTE_TOP_K, len(head.class_ids))
        )
        for name, head in heads.items()
    }
    rows = len(next(iter(columns.values())))
    return [{name: column[i] for name, column in columns.items()} for i in range(rows)]


def ood_scores(bank, embeddings, mean, std, k=5):
    queries = standardise(embeddings, np.asarray(mean), np.asarray(std))
    return knn_distances(bank, queries, k)


class GalleryIndex:
    def __init__(self, names, embeddings, projector=None):
        vectors = np.asarray(embeddings)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be two-dimensional")
        if len(names) != len(vectors):
            raise ValueError(f"{len(names)} names for {len(vectors)} embeddings")
        if projector is not None:
            vectors = projector.project(vectors)
        self._names = tuple(str(name) for name in names)
        self._projector = projector
        self._index = BruteForceIndex(vectors.shape[1])
        self._index.add(vectors)

    @property
    def projected(self):
        return self._projector is not None

    @property
    def names(self):
        return self._names

    def __len__(self):
        return len(self._names)

    def search(self, embeddings, k):
        if not 1 <= k <= len(self):
            raise ValueError(f"k must be in [1, {len(self)}], got {k}")
        queries = np.asarray(embeddings)
        if self._projector is not None:
            queries = self._projector.project(queries)
        found = self._index.search(queries, k)
        return [
            [(self._names[int(i)], float(s)) for i, s in zip(indices, scores)]
            for indices, scores in zip(found.indices, found.scores)
        ]


def search_text(encoder, gallery, texts, k):
    if gallery.projected:
        raise ValueError("text search needs an unprojected gallery")
    if isinstance(texts, str):
        raise ValueError("texts must be a list of strings")
    items = list(texts)
    if not items:
        raise ValueError("texts must not be empty")
    return gallery.search(encoder.encode_text(items), k)


def latin_text(text):
    try:
        return to_latin_plate(text)
    except ValueError:
        return text


def native_query(query):
    try:
        return from_latin_plate(query.strip().upper())
    except ValueError:
        return query


def read_plates(reader, blobs):
    items = list(blobs)
    results = list(reader.read_plates(items))
    if len(results) != len(items):
        raise ValueError(f"reader returned {len(results)} results for {len(items)} images")
    if not all(isinstance(item, PlateResult) for item in results):
        raise ValueError("reader must return PlateResult objects")
    return results


def find_plates(results, query, max_edits):
    if max_edits < 0:
        raise ValueError(f"max_edits must be >= 0, got {max_edits}")
    query = native_query(query)
    matches = []
    for index, result in enumerate(results):
        if result.reading is None:
            continue
        text = result.reading.text
        if match_plate(text, query, max_edits):
            matches.append((index, plate_distance(text, query)))
    return matches
