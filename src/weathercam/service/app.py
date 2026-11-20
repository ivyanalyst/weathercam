import base64
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from weathercam.service import handlers
from weathercam.service.worker_client import WorkerError

DEFAULT_MAX_IMAGES = 16
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
PAGE_PATH = Path(__file__).parent / "static" / "index.html"
PAGE_POLICY = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'"


@dataclass(frozen=True)
class OodBank:
    bank: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    k: int = 5


class ImagesRequest(BaseModel):
    images: list[str]


class PlatesRequest(BaseModel):
    images: list[str]
    query: str | None = None
    max_edits: int | None = None


class TextSearchRequest(BaseModel):
    texts: list[str]
    k: int


def _unavailable(name):
    return HTTPException(status_code=503, detail=f"{name} is not available")


def _decode(images, max_images, max_image_bytes):
    if not images:
        raise HTTPException(status_code=422, detail="images must not be empty")
    if len(images) > max_images:
        raise HTTPException(status_code=413, detail=f"at most {max_images} images per request")
    blobs = []
    for item in images:
        try:
            blob = base64.b64decode(item, validate=True)
        except ValueError:
            raise HTTPException(status_code=422, detail="images must be base64 strings") from None
        if len(blob) > max_image_bytes:
            raise HTTPException(status_code=413, detail=f"images are limited to {max_image_bytes} bytes")
        blobs.append(blob)
    return blobs


def _prediction(item):
    return {
        "class_id": item.class_id,
        "label": item.label,
        "probability": item.probability,
        "top": [
            {"class_id": class_id, "label": label, "probability": probability}
            for class_id, label, probability in item.top
        ],
    }


def _reading(reading):
    if reading is None:
        return None
    return {
        "text": reading.text,
        "text_latin": handlers.latin_text(reading.text),
        "confidence": reading.confidence,
        "positions": list(reading.positions),
    }


def _plate(index, result):
    return {
        "index": index,
        "width": result.width,
        "height": result.height,
        "box": None if result.box is None else list(result.box),
        "score": result.score,
        "reading": _reading(result.reading),
    }


def create_app(
    encoder=None,
    weather_head=None,
    attribute_heads=None,
    ood=None,
    gallery=None,
    plate_reader=None,
    image_reader=None,
    max_images=DEFAULT_MAX_IMAGES,
    max_image_bytes=DEFAULT_MAX_IMAGE_BYTES,
):
    if gallery is not None and gallery.projected:
        raise ValueError("the text gallery must not be projected")
    if image_reader is not None and gallery is None:
        raise ValueError("image_reader needs a gallery")
    features = {
        "weather": encoder is not None and weather_head is not None,
        "attributes": encoder is not None and bool(attribute_heads),
        "ood": encoder is not None and ood is not None,
        "plates": plate_reader is not None,
        "search_text": encoder is not None and gallery is not None,
    }
    app = FastAPI(title="weathercam")
    page = PAGE_PATH.read_text(encoding="utf-8")
    image_names = frozenset(gallery.names) if image_reader is not None else frozenset()
    image_lock = threading.Lock()

    @app.exception_handler(WorkerError)
    async def worker_failed(request, error):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get("/health")
    def health():
        return {
            "features": features,
            "gallery_size": None if gallery is None else len(gallery),
            "gallery_images": image_reader is not None,
            "max_images": max_images,
            "max_image_bytes": max_image_bytes,
        }

    @app.post("/weather")
    def weather(body: ImagesRequest):
        if not features["weather"]:
            raise _unavailable("weather")
        blobs = _decode(body.images, max_images, max_image_bytes)
        predictions = handlers.weather(weather_head, encoder.encode_images(blobs))
        return {"results": [_prediction(item) for item in predictions]}

    @app.post("/attributes")
    def attributes(body: ImagesRequest):
        if not features["attributes"]:
            raise _unavailable("attributes")
        blobs = _decode(body.images, max_images, max_image_bytes)
        rows = handlers.attributes(attribute_heads, encoder.encode_images(blobs))
        return {"results": [{name: _prediction(item) for name, item in row.items()} for row in rows]}

    @app.post("/ood")
    def ood_endpoint(body: ImagesRequest):
        if not features["ood"]:
            raise _unavailable("ood")
        blobs = _decode(body.images, max_images, max_image_bytes)
        embeddings = encoder.encode_images(blobs)
        scores = handlers.ood_scores(ood.bank, embeddings, ood.mean, ood.std, ood.k)
        return {"scores": [float(score) for score in scores], "k": ood.k}

    @app.post("/plates")
    def plates(body: PlatesRequest):
        if not features["plates"]:
            raise _unavailable("plates")
        if (body.query is None) != (body.max_edits is None):
            raise HTTPException(status_code=422, detail="query and max_edits must be given together")
        if body.max_edits is not None and body.max_edits < 0:
            raise HTTPException(status_code=422, detail="max_edits must be >= 0")
        blobs = _decode(body.images, max_images, max_image_bytes)
        results = handlers.read_plates(plate_reader, blobs)
        matches = None
        if body.query is not None:
            found = handlers.find_plates(results, body.query, body.max_edits)
            matches = [{"index": index, "distance": distance} for index, distance in found]
        return {
            "results": [_plate(index, result) for index, result in enumerate(results)],
            "matches": matches,
        }

    @app.post("/search/text")
    def search_text(body: TextSearchRequest):
        if not features["search_text"]:
            raise _unavailable("search_text")
        if not body.texts:
            raise HTTPException(status_code=422, detail="texts must not be empty")
        if len(body.texts) > max_images:
            raise HTTPException(status_code=413, detail=f"at most {max_images} texts per request")
        if not 1 <= body.k <= len(gallery):
            raise HTTPException(status_code=422, detail=f"k must be in [1, {len(gallery)}]")
        found = handlers.search_text(encoder, gallery, body.texts, body.k)
        return {"results": [[{"name": name, "score": score} for name, score in row] for row in found]}

    @app.get("/", include_in_schema=False)
    def index():
        return HTMLResponse(page, headers={"Content-Security-Policy": PAGE_POLICY})

    @app.get("/gallery/image/{name}", include_in_schema=False)
    def gallery_image(name: str):
        if name not in image_names:
            raise HTTPException(status_code=404, detail="no such image")
        try:
            with image_lock:
                data = image_reader.read_image("test", name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="image file missing")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})

    return app
