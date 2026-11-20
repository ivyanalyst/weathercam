# WeatherCam

Research code for vehicle attributes, plate reading, text-to-image retrieval and data mining on CCTV-style images, with a focus on adverse weather. An independent research and engineering project exploring vehicle attribute recognition, retrieval, and OOD behavior under adverse weather. Built for research and educational use.

The pipeline is inspired by Veri-Car (Muñoz et al., arXiv 2411.06864):

    detect vehicle -> crop -> OOD check -> attributes (colour, type, brand) -> plate detect + read
    -> embed -> text-to-image search -> mining / active learning

Everything below is one seed and one embedding backbone (frozen OpenCLIP ViT-B/16, laion2b_s34b_b88k, with small heads). Test data was not used for tuning: each test set was scored once, with settings fixed beforehand.

## What was not measured

- Text-to-image search was measured on 21 queries only (results in the table below). Weather queries (fog, haze, rain, sand storm) are unscored: the gallery has no weather labels, only similarity ranges exist. The first retrieval version used histograms, and a zero-shot test of camera view with text prompts was a negative result (accuracy 0.603 against a 0.564 majority baseline).
- Plate reading is weak off its training distribution: exact match 0.4470 on the test set against 0.9447 on validation. The service therefore returns a detector score and a per-position confidence, not a "confident" label.
- Plates: trained on Chinese plates only (CCPD). No plates from other
  countries were used or measured, so behaviour on them is unknown. Expect it
  to be weaker there; this is not measured. The service stores readings
  in the Chinese format and shows the province as Latin letters.
- Out-of-distribution: no threshold is set. The service shows the raw kNN
  score only. The high AUROC values (0.996 for VMMR against clean VeRi)
  separate one dataset from another, so they mostly reflect a dataset
  difference (source, camera, framing). They were not shown to detect
  weather, haze or fog. On corrupted VeRi the AUROC ranged from 0.589
  (haze) to 0.949 (JPEG) depending on the corruption. A usable threshold
  would need images from the target camera and its own false-accept and
  false-reject rates. That was not measured.
- The weather head is trained on DAWN only and has four classes.
- The colour and type heads are trained on VeRi. Brand uses VMMRdb makes (43 classes including "unknown").

## Results

| Task | Data | Result |
|---|---|---|
| Colour | VeRi test | accuracy 0.8939, macro-F1 0.7605 |
| Vehicle type | VeRi test | accuracy 0.9048, macro-F1 0.7909 |
| Brand (make) | VMMRdb test, 43 classes | accuracy 0.8043, macro-F1 0.8102 |
| Weather, 4 classes | DAWN test | 138/150 = 0.9200 (Wilson 0.8654-0.9536) |
| Retrieval, plain | VeRi, leave-one-out | P@1 0.9431, mAP@R 0.4046 |
| Retrieval, camera-aware | VeRi, leave-one-out | P@1 0.7489, mAP@R 0.3484 (raw 512-d: 0.5066, 0.1116) |
| OOD, kNN k=5, earlier run, bank overlap not rechecked | DAWN, VMMRdb | AUROC 0.9988, 0.9945 |
| Out-of-distribution, kNN k=5, other source | VMMR vs clean VeRi, 500 each | AUROC 0.996 [0.994, 0.998] |
| Text search, 21 queries | VeRi test, 2000 images | 20 of 21 with AUROC >= 0.85; green 0.522 |
| Plate detector | CCPD test | IoU > 0.7 in 0.9895 of images (top-1 box only) |
| Plate OCR | CCPD test | exact match 0.4470, CER 0.1412 |

Limits that go with the table:

- Brand accuracy compares with 0.025 for always predicting the most frequent class.
- Weather: the test accuracy rests on 150 images, and the validation macro-F1 is 0.7889, so 0.92 should not be quoted alone.
- Retrieval uses a leave-one-out protocol over vehicles, not the VeRi query/gallery split. Its mAP@R is not the VeRi-standard mAP and is not comparable to published VeRi numbers. These rows use a 128-d metric head.
- Plain retrieval counts frames from the same camera as correct;
  camera-aware counts only frames from other cameras. Read the camera-aware
  rows as the cross-camera result.
- Query expansion (k 10, weight 8) raised camera-aware P@1 to 0.7616 and mAP@R to 0.3964, but lowered plain P@1 to 0.8862.
- The CCPD protocol (one box per image, IoU > 0.7, exact match) follows the CCPD README, but the models, subsets and splits differ from its baseline table, so the numbers are not comparable. Thirteen test images whose labelled box is not on the plate are excluded from the true-box OCR numbers only.
- Robustness: holding one weather class out gave an OOD AUROC of about 0.73 for the kNN score against about 0.69 for the baseline. Among generic corruptions only blur caused a clear loss. Synthetic rain, snow and haze (strength 0.6) and blur (0.3) cost more than random feature noise of equal cosine similarity.
- Selecting images for mining by k-center against random sampling gave mixed results.
- Text search: P@1 is one trial per query and scores are not comparable across queries. Rare attributes have few matches (fewest: blue bus, 34). Colour and type are correlated in the gallery, so some queries test little beyond colour. Green is the failure case: the top 10 are ten green buses from two vehicles.

## Design

- Metrics (IoU, NMS, AP/mAP, CER, classification, retrieval, ROC AUC), splits, augmentation, corruptions, the kNN index, OOD scoring, k-center, near-duplicate grouping and the service are own code. Libraries: NumPy, Pillow, PyYAML and psutil; OpenCLIP and Ultralytics YOLO11m for the pretrained models; pytest for tests.
- src/weathercam never imports torch or ultralytics. Code that needs them lives in research/ and runs in separate environments (Kaggle or local virtual environments). Heads and all scoring run with NumPy.
- The service layer (src/weathercam/service) has handlers free of torch and web frameworks. Embedding and plate reading run as two subprocess workers over JSON lines, with hash-checked weights and a memory check.
- scripts/check_service.py reproduces stored validation numbers and checks that the workers agree with stored outputs on a small sample (16 images each). The plate sample was not stratified by province or subset, and uploads of other formats or with EXIF rotation were not tested.

## Layout and tests

- src/weathercam: library code (core metrics, data handling, attributes, retrieval, plates, review, service).
- research/: scripts that need torch (embedding, training, detector runs) and scoring scripts; requirements-embed.txt and requirements-yolo.txt list their dependencies.
- scripts/: data export, inspection, review and check scripts.
- tests/: pytest suite.

The main environment needs only numpy, pyyaml, psutil and pillow:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    pip install pytest
    python3 -m pytest -q

## Data and licences

No datasets, dataset images, plate texts or trained weights are part of this repository. The datasets (VeRi, DAWN, CCPD2019, VMMRdb) must be obtained from their owners under their own terms; the terms as known are in THIRD_PARTY.md. Paths are set in configs/default.yaml and a git-ignored configs/local.yaml.

The code is released under AGPL-3.0-or-later (see LICENSE); the detector dependency Ultralytics is AGPL-3.0. Third-party components and weights are listed in THIRD_PARTY.md, and references are in CITATION.cff.

## Web page

The service serves a single HTML page at GET / (no external scripts, fonts or requests). It shows a panel for each feature that /health reports as enabled: weather, vehicle attributes, out-of-distribution scores (raw, no threshold), number plates (with an optional plate query) and text search. For text search each hit shows a thumbnail and then the weather, attributes and plate reading computed on that image. Thumbnails need `--image-zip` pointing to a VeRi zip that holds every gallery name.

Launch (all features):

    python3 scripts/run_service.py --embed-worker-python /path/to/clip-venv/bin/python --plate-worker-python /path/to/yolo-venv/bin/python --gallery /path/to/gallery_2000.npz --image-zip /path/to/VeRi.zip --port 8014

Known limits, seen in a browser check on small samples:
- The weather head labelled a clear photo of a US sedan as fog (86%). It was never validated on such photos.
- The type head called the rear of a Ford sedan a bus (61%) on a close-up CCPD image.
- Plate readings on small VeRi crops have low scores and confidences (0.05 to 0.13 in the samples seen) and should not be trusted.
- Text-search quality was measured on 21 queries only; weather queries are unscored (see RESULTS.md).
- If a worker process dies, the service keeps running and /health still lists its feature. Requests to it return 502 until the service is restarted.
- Large uploads, EXIF rotation, other image formats and concurrent use were not tested.

## Run it yourself (Docker, CPU only)

### Build

    docker build -t weathercam:test .

Needs internet (installs packages from PyPI). The image is about 4 GB.

### What you need besides the image

- `outputs/` folder: the model files and results the service reads.
  Not stored in git; it is shared separately and mounted read-only.
- `gallery_2000.npz`: the text-search index. Also shared separately.
- Optional: the VeRi dataset zip, only for thumbnails in text search.
  Everything else works without it.

### Start

    chmod +x run.sh
    OUTPUTS=./outputs GALLERY_NPZ=/path/to/gallery_2000.npz ./run.sh
    # with thumbnails:
    OUTPUTS=./outputs GALLERY_NPZ=/path/to/gallery_2000.npz VERI_ZIP=/path/to/VeRi.zip ./run.sh

Open http://127.0.0.1:8014

### First run vs later runs

- The first run needs internet: it downloads the embedding model
  (about 600 MB, 572 MB measured) into `~/.cache/weathercam-hf`
  (change with `HF_DIR=...`).
- Later runs work offline (`run.sh` detects the filled cache).
- If the first download was interrupted and the next start fails,
  delete `~/.cache/weathercam-hf` and run again.

### Known behaviour

- Harmless: the Hugging Face "unauthenticated requests" warning and a
  `favicon.ico` 404. No token is needed.
- A worker that dies is not reported: `/health` still lists the feature
  and requests return 502 until you restart the container.
- Plate search: type the reading as shown, e.g. JS-D9157K (province
  code, hyphen, 6 characters). It is converted back to the stored reading
  before matching. A query without a valid code is used as typed and compared
  with the whole stored reading, so AZ5761 is 1 edit away from a reading
  with a province character; a query 2 characters short needs max edits 2.
- JPEG/PNG only; HEIC is not supported.
- Tests run on the host (`pytest`), not inside the image. The image is
  smoke-tested: startup, `/health`, and all five panels.
- Model quality is limited: the weather, type and brand heads make
  visible mistakes.

Phase 16 measurements (OOD scoring and text search): see RESULTS.md.
