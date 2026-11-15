# Third-party components

Versions and licence fields come from installed package metadata on 2026-11-15. A licence field is the package author's claim, not a legal review.

## Python packages

| Package | Version | Licence field | Environment |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | all |
| PyYAML | 6.0.3 | MIT | main |
| psutil | 7.2.2 | BSD-3-Clause | main |
| pytest | 9.2.0 | MIT | main (tests) |
| Pillow | 13.0.0 | MIT-CMU | main, yolo, clip |
| opencv-python | 5.0.0.93 | Apache 2.0 | yolo |
| torch | 2.15.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | yolo, clip |
| torchvision | 0.29.2+cpu | BSD | clip |
| open_clip_torch | 4.2.1 | MIT | clip |
| ultralytics | 8.4.202 | AGPL-3.0 (classifier: GNU Affero General Public License v3 or later) | yolo |

The main environment has no torch. The yolo and clip environments are used only by scripts under research/.

## Ultralytics

Ultralytics YOLO11m is used for detection under AGPL-3.0, accepted for research. This repository is released under AGPL-3.0; see LICENSE.

## Model weights

| Weights | Source | Licence | Note |
|---|---|---|---|
| OpenCLIP ViT-B-16, laion2b_s34b_b88k | Hugging Face repo laion/CLIP-ViT-B-16-laion2B-s34B-b88K | MIT (model card header) | trained on LAION-2B English |
| YOLO11m starting weights | Ultralytics | not recorded | |
| Plate detector, plate OCR, DAWN detector, heads | trained here from the datasets below | not published | never distributed |

The model card calls any deployed use out of scope and advises against image search without thorough in-domain testing with a fixed set of classes. It also rules out surveillance and face recognition and recommends English only. This project is research only, and text-to-image search quality was not measured.

## Datasets

| Dataset | Role | Terms as known |
|---|---|---|
| VeRi | colour, type, retrieval | non-commercial, do not distribute; not redistributed here |
| DAWN | detection, weather | CC BY-NC-SA 4.0 |
| CCPD2019 | plates | MIT per the dataset README and the ECCV 2018 paper; rights in the underlying photos not verified |
| VMMRdb | brand | repository README states no licence; an MIT notice (copyright 2017 Faezeh Tafazzoli) was found separately, source not recorded; rights in the underlying photos not verified |

No dataset images, plate texts or trained weights are part of this repository.

## Method reference

The pipeline is inspired by Veri-Car (Muñoz et al., arXiv 2411.06864). Full references are in CITATION.cff.
