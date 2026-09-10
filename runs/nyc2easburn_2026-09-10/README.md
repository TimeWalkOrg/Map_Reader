# nyc2easburn_2026-09-10 — NYC 1776 fire-map vectors → Easburn 1776 Philadelphia pass

**Status: machine-candidate run. Nothing here is accepted geometry. No PostGIS or QGIS-project writes.**
Requested by Sunil (Slack #qgis thread `1789070222.107089`): "run a pass through on the Easburn map using this data".

## What was trained
Two small U-Nets (base 32, depth 5, 7.76 M params, 2-channel input) on tiles cut from the NYC fire map
(`1776_nyc_map_fire_map_upscaled.tif`, sha256 `7aae33bd…3ba08a`) resampled to a common **0.5 m/px EPSG:3857** grid,
with a shared ink preprocessing (background-normalised darkness + adaptive-threshold binary ink) so that the Easburn
plate (`tw_1776_philadelphia_map_easburn_plan_v2_cog.tif`, sha256 `7f2d5509…ae99c`) can be fed to the same models.
Block-level spatial split from `training/sunil_nyc_1776_fire_map_2026-09-10/proposed_split_*.csv` (val = NE band);
train tiles never contain val pixels. Trained on pc5090 (RTX 5090, torch 2.9.1+cu128) — `ckpt/*_train_log.json`.

* **roads** — street-*corridor* mask from `1776_nyc_vector_road_graph`, but only the 530 `roads.cga` edges that have a
  measured white corridor on the plate (`qa_roads.csv`: n≥3, width≥2 m); the 773 graph edges with no drawn corridor
  (planned/undrawn streets, hatching) and piers/greenspace are *ignored*, not taught. Best val IoU **0.42**.
  Two earlier formulations are kept for the record: v1 thin 2 m centerline (val IoU 0.11), v2 corridor on all edges (0.20).
* **buildings** — weak labels: 2,353 building polygons (off-plate / blank-plate / modern-proxy-suspect removed) +
  51 landmarks (10 not-on-ink removed), eroded 3 m, boundary ignore band; negatives *only* inside the 10 exhaustive
  blocks and the road corridors. Best val IoU vs weak labels **0.52** (precision 0.57 / recall 0.86) — it learned
  "built-up block vs street", which is all the weak labels can teach.

## What came out on Easburn
| file | rows | what it is |
|---|---|---|
| `easburn_road_candidates.gpkg` | see manifest | centerlines = pruned skeleton of the predicted corridor (prob ≥ 0.4), `confidence` = mean prob |
| `easburn_builtup_unet_candidates.gpkg` | see manifest | raw NYC-trained building head, thresholded — **does not transfer** (0/51 parcels), kept as evidence |
| `easburn_building_candidates.gpkg` | see manifest | hybrid instances: plate ink (close/open morphology) gated by the NYC-trained road mask and the drawn-city polygon, refined per blob with the existing Easburn Geo-SAM (SAM-B) embeddings; `method` column records `ink_blob` vs `ink_blob+geosam_refine` |

All layers: EPSG:3857, valid geometry, `origin='machine_candidate_nyc_trained'`, `confidence`, `oversize_flag`.

## Metrics (`metrics.json`)
* Easburn hybrid instances vs Sunil's 51 manual parcels (held-out, one-to-one Hungarian matching, IoU ≥ 0.5):
  recall 21/51, median matched IoU 0.85, median boundary distance 0.6 m; 11 parcels hit by party-wall merges,
  8 oversize; precision inside the parcel envelope ~0.32 (*indicative only* — the parcels are positives-only).
* Raw NYC-trained building head on Easburn: 0/51.
* Easburn road centerlines vs PostGIS `1776_philadelphia_road_graph` (254 edges, mostly OSM-derived; omits alleys):
  see `metrics.json` — completeness ~0.18/0.27/0.36 within 3/5/8 m, median offset 3.8 m for candidate points that
  are on a mapped street; the model is strong on the built eastern half and patchy on the blank western grid.
* NYC val roads: see `nyc_val_roads_*` in `metrics.json` (drawn-corridor edges only is the fair comparison).

## Honest limits
* The NYC building polygons are not traced from the plate (QA: IoU vs drawn cells 0.21), and the two plates use
  different conventions (NYC: white tick-divided lot cells; Easburn: hatched/solid dark blocks). A footprint model
  trained on NYC weak labels therefore cannot learn Easburn footprints — the instance candidates come from Easburn's
  own ink + Geo-SAM, with the NYC model contributing only the street/built-up gate.
* Roads are the part that genuinely transferred.
* Human review incomplete: candidates need QGIS verification (skill step 5) before anything moves toward PostGIS.

## Reproduce
```
env/bin/python prep_tiles.py                       # tiles + Easburn input (16 s)
python train.py --task roads --data nyc_tiles.npz --out ckpt   # on GPU (≈8 min); same for --task buildings
env/bin/python infer_vectorize.py --which easburn  # + --which nyc
geo-sam/env/bin/python instance_candidates.py
env/bin/python score.py && env/bin/python evidence_pngs.py && env/bin/python build_manifest.py
```
Large intermediates (`scratch/`, `ckpt/*.pt` 31 MB each) — checkpoints are committed; scratch is not.
