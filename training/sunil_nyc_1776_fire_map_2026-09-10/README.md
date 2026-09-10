# NYC 1776 fire-map vectors — registration + QA (2026-09-10)

Sunil (Slack #map_reader, 2026-09-10) asked that his three NYC 1776 PostGIS layers, which he
describes as vectors/parcels digitized over `1776_nyc_map_fire_map_upscaled`, be used as
Map_Reader training data. This folder registers them read-only and records how well they
actually sit on that plate. **No training run was started. No DB table or QGIS project was
modified.** Authoritative selection: [../active_dataset.json](../active_dataset.json).

## Source snapshot

PostGIS `timewalk` schema, read at **2026-09-10 20:01:13 UTC** (`export/snapshot_utc.txt`).
`prepare.py` turns the `psql \copy` CSVs (all columns, hex-EWKB geometry) into lossless
GeoPackages; geometry round-trip is byte-exact and every attribute cell matches
(`validation.json`). The GPKG FID column is `gpkg_fid` so the road graph's own `fid` survives.

| layer | rows | geometry | valid | dup geoms | notes |
|---|---|---|---|---|---|
| `1776_nyc_parcels_buildings` | 2,733 | MultiPolygon 3857 | 2,733 | 0 | 2 pairs overlap > 1 m² (5.1 m² total); median 119 m² (UTM18N) |
| `1776_nyc_parcels_landmarks` | 62 | MultiPolygon 3857 | 61 | 0 | id 54 *Fish Market* self-intersects |
| `1776_nyc_vector_road_graph` | 1,590 | MultiLineString **Z** 3857 | 1,590 | 0 | 49.2 km; CityEngine attributes |

(The task brief said ~1,506 road rows; the table holds 1,590.)

Provenance columns say nothing about tracing: buildings' `source` describes attribute origins
(`period_defaults_v1`, `1786_NYC_Directory`, `great_fire_spatial_analysis`, …), house numbers are
generated (`address_generated` = true for 2,658), and the road graph carries Esri CityEngine
rule attributes (`roads.cga`, `piers.cga`, lane/sidewalk widths, `randomseed`). The landmarks
layer *is* tied to the plate: ids 1–52 and names follow the plate's numbered key.

## Raster

`/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif` — the datasource of
QGIS layer `1776_nyc_map_fire_map_upscaled` in both the local `TimeWalk_Maps.qgs` and the DB copy
(`timewalk.qgis_projects`). sha256 `7aae33bd7350b077d095a04e26a080b789f93929a6a56aa10098735a983ba08a`
(= Git LFS oid), 194,692,587 B, 8331 × 7788 × 3 uint8, EPSG:3857, 0.2893 m/px projected
(≈ 0.219 m/px on the ground at 40.71° N), uncompressed, no overviews. Black-on-white binarised
engraving; off-plate area is pure black. Lots are drawn as tick-divided frontage cells (median
drawn cell ≈ 38 ground m²), public buildings as solid blobs, the burnt district is overprinted
with diagonal hatching.

## QA results (`qa_alignment.py`, `qa_cells.py`, `qa_landmarks_blob.py`)

**Buildings — not pixel-traced from this plate.**
- Boundary-within-3-px-of-ink: median 0.48, identical to a 4 m shifted baseline (gain median
  **0.00**, p10 −0.12). The plate is too dense for distance-to-ink alone, hence the cell test:
- IoU with the best drawn lot cell: median **0.21**; 133 / 2,507 on-plate polygons ≥ 0.5, 5 ≥ 0.7.
  Polygons are ~3× the drawn cells and span 0 (839), 1 (902), 2 (465), 3+ (301) cells.
- 226 polygons are off the plate (NE corner), 140 sit on blank plate with no footprint ink.
- OSM proxy check (Overpass 2026-09-10, 3,498 footprints): OSM IoU median 0.03; 18 polygons have
  IoU ≥ 0.5 and 15 of those also have poor ink support → `modern_proxy_suspect`
  (ids in `manifest.json`). They are not modern copies as a set; the flagged few need review.
- Class counts: loose_placement 2,115 · traced_single_cell 133 · traced_merged_cells 119 ·
  blank_plate 140 · off_plate 226. Evidence: `evidence/buildings_contact_sheet.jpg`,
  `evidence/buildings_cell_match_contact_sheet.jpg`, `evidence/buildings_cell_iou_hist.png`.

**Landmarks — plate-derived, approximate.** 24 / 61 on-plate polygons sit on solid ink blobs
(ink-centroid offset median 1.2 m ground, p90 2.7 m), 27 partial, 10 not on any blob
(ids 17, 25, 26, 38, 39, 44, 54, 60, 63, 72). Rectangles, not traced outlines.
Evidence: `evidence/landmarks_contact_sheet.jpg`.

**Road graph — follows the corridors.** On real white corridors (624 edges, width ≥ 2 m)
|centerline offset| median **1.1 m** ground, p75 1.9 m, p90 2.9 m; 139 edges > 2 m. 408 measured
edges run through ink (water/burnt-district hatching, piers) and cannot be scored; 558 are too
short or off-plate. Evidence: `evidence/roads_offset_hist.png`.

**Coverage.** Blocks = 261 faces polygonised from the road graph. Exhaustive (≥ 90 % of ≥ 3
drawn cells overlapped): **10 blocks, 77,294 ground m², 196 building polygons** (18 of them
traced-cell matches). Partial 101, sparse 74, no drawn cells 39, off-plate 37. Overall 2,126 /
4,101 drawn cells (52 %) are overlapped by some polygon. Evidence: `evidence/block_coverage_map.png`,
`block_coverage.gpkg/.csv`. Caveat: where the road graph does not close a block several blocks
merge into one face, and hatched water/burnt areas inflate the drawn-cell count in sparse blocks.

## Training roles (proposed — see `../active_dataset.json`)

- Buildings + landmarks → **polygon task, weak labels only.** Usable as instance/location
  priors and for block-level evaluation; do not treat their edges as the plate's footprint
  boundaries. Negative pixels may be sampled only inside the 10 exhaustive blocks.
- Road graph → **separate line-extraction task** with a ≥ 2 m tolerance buffer; never mixed
  with the polygon task.
- Block-level split (`propose_split.py`): val = contiguous NE band, 41 blocks / 481 buildings /
  10 landmarks; train = 183 blocks / 2,025 buildings / 51 landmarks; 227 buildings excluded
  off-plate. Per-feature assignments in `proposed_split_*.csv`.
- Easburn 51 (Philadelphia) remains the primary manual Easburn set and the held-out cross-map
  validation set.

## Files

`*.gpkg` layers · `export/` raw CSVs, schema, OSM snapshot, great-fire polygon · `validation.json`
· `qa_alignment.json`, `qa_cells.json`, `split_summary.json`, per-feature `qa_*.csv` ·
`block_coverage.gpkg` · `manifest.json` (everything above, with hashes) · `evidence/` PNGs.
Scripts run with `Map_Reader/env/bin/python` in the order prepare → qa_alignment → qa_cells →
qa_landmarks_blob → propose_split → build_manifest.
