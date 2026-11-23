# Phase 16 results: OOD scoring and text search

Measurement only. No change to the models, the gallery, the service code
or the page. Seed 42 everywhere. Run on 2026-11-17 and 2026-11-18.
Every number below comes from the files in eval/. Sample sizes and
weaknesses are listed next to the numbers. Nothing here is a claim beyond
what was measured.

## Environment

- Repo HEAD during the OOD run: 45cea19. Analysis and text scripts were
  added in 445ccb5.
- Service: `OUTPUTS=./outputs GALLERY_NPZ=/path/to/gallery_2000.npz ./run.sh`
  at http://127.0.0.1:8014. No VERI_ZIP was mounted.
- Model: ViT-B-16 laion2b_s34b_b88k, dim 512, CPU only, about 0.76 s per
  image.
- OOD score: embedding standardised with the service mean/std, L2
  normalised, then the distance sqrt(2 - 2 cos) to the 5th nearest bank
  vector (k=5). Higher means more OOD.
- OOD bank: the 29,918 fit rows of `veri_inner_split(data, 42)` (VeRi train
  images). The inner split is by vehicle. The 7,828 val rows (119 vehicles)
  are not in the bank.
- Text path: `research/embed_worker.py` (repo copy) tokenizes the text and
  calls `model.encode_text`. No prompt template appears there. The copy
  inside the `weathercam:test` image was not checked, so this is
  "appears".
- Preprocessing: the bank extractor (`research/extract_embeddings.py`) and
  the service worker both use the open_clip transform on `convert("RGB")`
  followed by L2 normalisation. This appears to match. The arguments used to
  build the bank file were not checked.

## Data used

- `/path/to/VeRi.zip` (train and test images, labels)
- `outputs/embeddings/veri_vitb16_laion2b.npz` (bank source)
- `/path/to/gallery_2000.npz` (2000 gallery images, all from the
  VeRi test split)
- `/path/to/ccpd_subset` (subsets base, weather, blur,
  challenge only)
- `/path/to/vmmr_subset`

Not used: the CCPD subsets db, fn, rotate and tilt. No non-vehicle image
set is on disk, so the non-vehicle sanity check was omitted.

## Commands, in order

```bash
python3 scripts/build_sets.py \
  --npz /path/to/gallery_2000.npz \
  --veri-zip /path/to/VeRi.zip \
  --ccpd-root /path/to/ccpd_subset \
  --vmmr-root /path/to/vmmr_subset

nohup python3 scripts/eval_ood.py \
  --veri-zip /path/to/VeRi.zip --boot 1000 \
  > eval/ood_run.log 2>&1 &

python3 scripts/analyze_ood.py
python3 scripts/analyze_extra.py

python3 scripts/build_queries.py \
  --npz /path/to/gallery_2000.npz \
  --veri-zip /path/to/VeRi.zip
python3 scripts/eval_text.py
python3 scripts/diag_text.py
```

Outputs: eval/sets.json, eval/ood_scores.csv, eval/ood_metrics.json,
eval/ood_run.log, eval/text_metrics.json, scripts/queries.json, and the
console outputs eval/analyze_ood_out.txt, eval/analyze_extra_out.txt,
eval/eval_text_out.txt, eval/diag_text_out.txt. The OOD run took about 75
minutes (4491 s for 5,700 images). A short smoke test (--limit 16) was a
plumbing check only and is not quoted.

## Part A: OOD

### Sets and sample sizes

| set | what it is | n | vehicles |
|---|---|---|---|
| veri_id_same | VeRi test images, not in gallery, never in bank | 500 | 169 |
| veri_id_val | VeRi train images held out of the bank (by vehicle) | 500 | 113 |
| ccpd_base, _weather, _blur, _challenge | CCPD subsets | 500 each | n/a |
| vmmr | VMMR images | 500 | n/a |
| veri_{noise,blur,jpeg,dark,fog}_s{1,2}, veri_recode_s1 | corrupted copies of the same 200 test images | 200 each | 119 |

The 200 corruption base images are the last 200 of the veri_id_same
sample. The clean reference for those comparisons is the other 300. The
corrupted sets reuse the same base images, so they are paired and not
independent. Corruptions are deterministic (seeded).

Corruption parameters: noise sigma 15 / 40 (0-255 scale); blur radius 2 / 5
px; jpeg quality 20 / 5; dark brightness x0.4 / x0.15; "fog" is a uniform
haze blend toward light gray (225) at 0.4 / 0.7, so call it haze and not a
weather test. Every non-jpeg corruption is re-saved at JPEG quality 95.
veri_recode_s1 is the control: re-save at quality 95 and nothing else.

CCPD "weather" is a folder name, not a verified weather label. CCPD and
VeRi are both Chinese, so the CCPD sets are a camera, viewpoint and scale
shift and not other-country plates.

### Correction to the first set build

The first veri_id_new set was drawn from VeRi train images. 384 of its 500
images (76.8%) were in the OOD bank, so each was its own nearest neighbour.
It was never reported and has been removed. The cause was my set
construction, which excluded gallery vehicles but never checked the bank.
Fixed in 45cea19: the sets are rebuilt with the service's own inner split,
and build_sets.py asserts that no in-distribution set touches the bank or
the gallery.

### Score distributions

| set | median | p5 | p95 | min | max |
|---|---|---|---|---|---|
| veri_id_same | 0.887 | 0.613 | 1.003 | 0.370 | 1.089 |
| veri_id_val | 0.875 | 0.608 | 1.007 | 0.444 | 1.078 |
| ccpd_base | 1.173 | 1.064 | 1.218 | 0.972 | 1.244 |
| ccpd_weather | 1.180 | 1.098 | 1.219 | 1.001 | 1.246 |
| ccpd_blur | 1.168 | 1.082 | 1.211 | 1.039 | 1.233 |
| ccpd_challenge | 1.164 | 1.067 | 1.210 | 0.959 | 1.242 |
| vmmr | 1.115 | 1.033 | 1.173 | 0.953 | 1.201 |

No clean VeRi image scored 1.10 or more, and none was in the 1.13 to 1.24
band (0 of 1,000). Five of the ten lowest-scoring CCPD/VMMR images are VMMR
(the other five are CCPD challenge or base).

### Other-source sets versus clean VeRi

Positive class is OOD. Operating points (named to avoid the ambiguous
"FPR at 95% TPR"):
- OOD accepted = fraction of the OOD set scoring at or below the 95th
  percentile of the clean reference (threshold keeps 95% of clean images).
- Clean flagged = fraction of the clean reference scoring at or above the
  5th percentile of the OOD set (threshold catches 95% of OOD images).

AUPR chance level is 0.50 here (500 vs 500). AUPR equals AUROC to within
0.001 in every row, so it is not repeated.

| OOD set | vs | AUROC [95% CI] | OOD accepted | clean flagged |
|---|---|---|---|---|
| ccpd_base | id_same | 0.999 [0.998, 1.000] | 0.002 | 0.002 |
| ccpd_base | id_val | 0.999 [0.998, 1.000] | 0.002 | 0.006 |
| ccpd_weather | id_same | 1.000 [0.999, 1.000] | 0.002 | 0.000 |
| ccpd_weather | id_val | 1.000 [0.999, 1.000] | 0.002 | 0.000 |
| ccpd_blur | id_same | 1.000 [1.000, 1.000] | 0.000 | 0.002 |
| ccpd_blur | id_val | 1.000 [0.999, 1.000] | 0.000 | 0.000 |
| ccpd_challenge | id_same | 0.998 [0.996, 0.999] | 0.016 | 0.002 |
| ccpd_challenge | id_val | 0.998 [0.996, 0.999] | 0.016 | 0.004 |
| vmmr | id_same | 0.996 [0.994, 0.998] | 0.014 | 0.008 |
| vmmr | id_val | 0.996 [0.994, 0.998] | 0.014 | 0.016 |

The two clean sets give the same picture (medians differ by 0.012; that
difference was not tested).

### Corrupted VeRi versus clean VeRi

Reference = the 300 clean test images that were not corrupted. AUPR chance
is 0.40 (200 of 500). "Delta" is the median over the 200 images of
(corrupted score - the same image's clean score).

| corruption | AUROC [95% CI] | AUPR | OOD accepted | clean flagged | delta |
|---|---|---|---|---|---|
| noise s1 | 0.758 [0.709, 0.801] | 0.719 | 0.630 | 0.893 | +0.091 |
| noise s2 | 0.844 [0.807, 0.878] | 0.804 | 0.545 | 0.573 | +0.113 |
| blur s1 | 0.826 [0.785, 0.867] | 0.835 | 0.395 | 0.867 | +0.126 |
| blur s2 | 0.833 [0.797, 0.868] | 0.779 | 0.565 | 0.540 | +0.111 |
| jpeg s1 | 0.805 [0.763, 0.844] | 0.796 | 0.495 | 0.883 | +0.109 |
| jpeg s2 | 0.949 [0.928, 0.967] | 0.937 | 0.210 | 0.217 | +0.158 |
| dark s1 | 0.679 [0.628, 0.729] | 0.627 | 0.770 | 0.927 | +0.063 |
| dark s2 | 0.866 [0.827, 0.898] | 0.868 | 0.335 | 0.753 | +0.156 |
| haze s1 | 0.589 [0.535, 0.643] | 0.503 | 0.860 | 0.937 | +0.029 |
| haze s2 | 0.738 [0.694, 0.783] | 0.707 | 0.650 | 0.870 | +0.087 |
| recode (control) | 0.511 [0.461, 0.562] | 0.409 | 0.960 | 0.960 | +0.0002 |

The re-encode control shows no measurable shift (AUROC CI includes 0.5,
delta 0.0002), so the corruption results are not a JPEG-95 re-save
artefact. Every real corruption raises the score of the same image. Mild
haze and mild darkening are detected weakly: at the clean-95% threshold,
86% of haze s1 and 77% of dark s1 images are accepted as in-distribution.

### Rates at fixed thresholds

Clean flagged = score above t. OOD accepted = score at or below t.

| t | id_same flagged | id_val flagged | ccpd_base | ccpd_weather | ccpd_blur | ccpd_challenge | vmmr |
|---|---|---|---|---|---|---|---|
| 1.00 | 0.060 | 0.060 | 0.002 | 0.000 | 0.000 | 0.016 | 0.014 |
| 1.03 | 0.012 | 0.016 | 0.010 | 0.006 | 0.000 | 0.024 | 0.040 |
| 1.05 | 0.004 | 0.008 | 0.030 | 0.018 | 0.006 | 0.042 | 0.080 |
| 1.08 | 0.002 | 0.000 | 0.068 | 0.034 | 0.042 | 0.068 | 0.204 |

VMMR contributes most of the accepted OOD images at every threshold.

### Image geometry (a confound, not measured away)

| set | median w x h | median aspect (w/h) |
|---|---|---|
| veri_id_same | 226 x 204 | 1.13 |
| veri_id_val | 220 x 190 | 1.14 |
| ccpd_* (all four) | 720 x 1160, every image | 0.62 |
| vmmr | 548 x 408 | 1.34 |

CCPD images are full scenes (portrait, all the same size), while VeRi
images are vehicle crops. The near-perfect CCPD separation therefore
reflects scene framing and scale as well as the camera. It was not separated
from content: that would need CCPD vehicle crops, which was not done. VMMR
(vehicle-centred photos, closer aspect) is the harder set and is the more
informative OOD number.

## Part B: text search

Gallery: 2000 VeRi test images. Ground truth is VeRi test_label.xml colour
and type. Gallery counts: black 681, white 434, gray 308, red 191, green
127, yellow 126, blue 55, golden 32, brown 25, orange 21; sedan 887, SUV
434, truck 301, pickup 122, hatchback 139, bus 53, van 47, estate 17. MPV
has 0 images and was dropped. Golden and yellow are kept separate. A bare
"car" query was skipped. Each query was ranked over the full gallery
(k=2000). Smallest query: blue bus with 34 matches, so no query is below
10 matches.

Baseline = fraction of the gallery matching (the random P@k). MRR is the
reciprocal rank of the first correct image. AUROC is over the full ranking.
"veh10" is the number of distinct vehicles among the top 10 (a vehicle can
appear as several near-identical frames).

| query | matches | baseline | P@1 | P@5 | P@10 | MRR | AUROC | veh10 |
|---|---|---|---|---|---|---|---|---|
| white | 434 | 0.217 | 1.0 | 1.0 | 0.90 | 1.000 | 0.916 | 6 |
| black | 681 | 0.341 | 1.0 | 0.8 | 0.90 | 1.000 | 0.916 | 8 |
| gray | 308 | 0.154 | 0.0 | 0.8 | 0.80 | 0.500 | 0.851 | 8 |
| red | 191 | 0.096 | 1.0 | 1.0 | 1.00 | 1.000 | 0.995 | 4 |
| yellow | 126 | 0.063 | 1.0 | 1.0 | 1.00 | 1.000 | 0.956 | 5 |
| green | 127 | 0.064 | 1.0 | 1.0 | 1.00 | 1.000 | 0.522 | 2 |
| blue | 55 | 0.028 | 1.0 | 1.0 | 0.90 | 1.000 | 0.960 | 4 |
| sedan | 887 | 0.444 | 1.0 | 1.0 | 0.90 | 1.000 | 0.877 | 10 |
| SUV | 434 | 0.217 | 1.0 | 1.0 | 1.00 | 1.000 | 0.891 | 2 |
| truck | 301 | 0.150 | 1.0 | 1.0 | 1.00 | 1.000 | 0.985 | 5 |
| pickup | 122 | 0.061 | 1.0 | 1.0 | 1.00 | 1.000 | 0.920 | 6 |
| hatchback | 139 | 0.070 | 0.0 | 0.6 | 0.70 | 0.500 | 0.859 | 7 |
| bus | 53 | 0.026 | 1.0 | 1.0 | 1.00 | 1.000 | 1.000 | 3 |
| van | 47 | 0.024 | 1.0 | 0.8 | 0.80 | 1.000 | 0.981 | 5 |
| white SUV | 106 | 0.053 | 1.0 | 1.0 | 1.00 | 1.000 | 0.982 | 4 |
| black sedan | 432 | 0.216 | 1.0 | 1.0 | 1.00 | 1.000 | 0.960 | 5 |
| red hatchback | 70 | 0.035 | 1.0 | 0.8 | 0.60 | 1.000 | 0.987 | 5 |
| green pickup | 98 | 0.049 | 1.0 | 0.6 | 0.40 | 1.000 | 0.925 | 9 |
| yellow truck | 114 | 0.057 | 1.0 | 1.0 | 1.00 | 1.000 | 0.992 | 4 |
| white truck | 122 | 0.061 | 0.0 | 0.8 | 0.80 | 0.500 | 0.929 | 6 |
| blue bus | 34 | 0.017 | 1.0 | 1.0 | 1.00 | 1.000 | 1.000 | 2 |

Reading the table:
- P@1 is a single trial per query. Gray, hatchback and white truck miss at
  rank 1 by near-ties (top two scores within 0.004), so P@1 is unstable.
  AUROC and P@5/P@10 are the steadier figures.
- Top-10 diversity is low for several queries (green 2 vehicles, SUV 2,
  blue bus 2), so a perfect P@10 there rests on very few independent
  vehicles.
- Green: the top 10 is ten green buses from two vehicles. Green cars
  overall rank at about the gallery average (mean rank 959 of 2000), and 98
  of the 127 green images are pickups, which rank among the worst for this
  query (mean rank 1175; only estate, n=17, ranks lower). This
  fits the chance-level AUROC of 0.522 (consistent with, not proven by, the
  diagnostics in eval/diag_text_out.txt).
- Hatchback: four of the top 10 are one orange hatchback vehicle; the
  misses are an SUV and two sedans. White truck: vans rank above trucks for
  this query (mean rank 269 vs 305). Whether these misses are label
  ambiguity or model error was not determined, because the images were not
  inspected.
- Colour and type are strongly correlated in this gallery (114 of 126 yellow
  images are trucks; 98 of 127 green images are pickups), so the combined
  queries "yellow truck" and "green pickup" test little beyond the colour
  alone and cannot separate colour from type.
- Similarity scores are not comparable across queries. Mean score of
  correct hits is 0.15 to 0.19 for colour queries and 0.23 to 0.26 for type
  queries, 0.25 to 0.27 for six of the seven combined queries (green pickup
  0.182), and the mean score of misses for type queries (up to
  0.217) is higher than that of correct hits for colour queries. No text
  threshold is proposed.

### Weather queries (unscored)

There are no weather labels in the gallery, so these cannot be scored. The
similarity ranges over the 2000 gallery images are shown only for
reference.

| text | top 1 | rank 10 | median | min |
|---|---|---|---|---|
| fog | 0.219 | 0.210 | 0.179 | 0.136 |
| haze | 0.224 | 0.206 | 0.173 | 0.124 |
| rain | 0.216 | 0.206 | 0.168 | 0.127 |
| sand storm | 0.201 | 0.178 | 0.105 | 0.035 |

## Part C: weather on vehicle crops (exploratory, not a measurement)

One quick check, run once. The script is not in the repo. DAWN
validation split (split_v1), 161 images, the saved weather head,
ground-truth boxes (car, bus, truck, motorcycle). Boxes with a short
side under 48 px were dropped (330 boxes); every image kept at least
one box.

Whole image: 127 of 161 correct. Largest crop: 90 (51 right to wrong,
14 wrong to right). Mean over all crops: 91 (48, 12). Rain fell from 20
of 31 to 3 and 1 of 31. Classes have 31 to 53 images.

The head was trained on whole images. No head was trained on crops, no
detector boxes were used, and the validation split also chose the head's
settings. The image encoder centre-crops each input to a square. The
service sends the whole image to the weather route.

## Threshold proposal (not applied; nothing was added to the page)

Proposal only: if an OOD flag is ever added for "other-source vehicle
images", a threshold in the range 1.03 to 1.05 is the candidate the numbers
support.
- At 1.03: 1.4% of the 1,000 clean VeRi images flagged (14 images);
  VMMR accepted 4.0%; CCPD accepted 0% to 2.4%.
- At 1.05: 0.6% of clean flagged (6 images); VMMR accepted 8.0%; CCPD
  accepted 0.6% to 4.2%.
Limits:
- It was chosen from the same scores it is evaluated on (in-sample).
- CCPD cannot confirm it because of the framing confound above, so VMMR is
  the only informative OOD set.
- The clean sets cover 282 vehicles from one camera network.
- It does not support catching haze or other degradations (haze s1 and dark
  s1 are mostly accepted).
- Nothing was measured on weather-camera frames. Full-scene frames like
  CCPD may well score above the threshold; this is a hypothesis and was not
  tested.

## Caveats

- Bootstrap CIs resample images. The clean sets contain correlated images
  from 169 and 113 vehicles, so the CIs are too narrow. The effective sample
  size is closer to the vehicle count.
- CCPD vs VMMR vs VeRi differ in resolution and framing as well as content
  (see the geometry table). Not separated.
- The corrupted sets are paired (same 200 images, 119 vehicles).
- The text queries use a 2000-image gallery from one camera network, with
  small counts for rare attributes. Single-query P@1 is one trial.
- Label noise in VeRi colour and type was not measured.
- Absolute distances assume the offline bank embeddings and the live
  service preprocess images identically; this appears to hold but was not
  verified end to end.
- The earlier hand-probed scores of 1.13 to 1.24 could not be traced (no
  record of the images). In this run, no clean VeRi image scored in that
  band. The 1.056 probe could still be a clean image in the upper tail (5
  of 1,000 clean images scored 1.04 to 1.07). The earlier explanation that a
  gallery image matches itself was wrong: the bank is not the gallery.
- No bug was found by these measurements.
