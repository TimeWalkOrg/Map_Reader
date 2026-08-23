# Pilot results — SAM vs hand-traced (1762 Clarkson & Biddle)

| id | building | IoU | precision | recall | predict (s) |
|---|---|---|---|---|---|
| 1 | Pennsylvania State House / Independence Hall | 0.220 | 0.349 | 0.374 | 0.08 |
| 2 | Christ Church | 0.442 | 0.721 | 0.532 | 0.07 |
| 13 | High Street Market shambles | 0.081 | 0.127 | 0.181 | 0.06 |
| 19 | Old Gaol and Work House (Old Stone Prison) | 0.281 | 0.302 | 0.808 | 0.06 |

Mean IoU: **0.256**

## Timing (Mac mini, x86-64 CPU, SAM ViT-B)
- Model load: 0.4 s (after one-time 375 MB checkpoint download)
- Image encode (2298×1148 px crop, one-time per window): ~5 s
- Per-building predict: **0.06–0.09 s** — effectively interactive
- Total pilot wall clock: ~5 s for 4 buildings after setup

## Honest read
Mean IoU 0.256 (box+point prompts; box-only was 0.186). **Not usable for
unattended auto-tracing.** Why the low scores:

1. **Ground-truth mismatch by design.** Per the geometry policy, surviving
   buildings (Independence Hall, Christ Church) are traced from *modern OSM
   footprints*, not period ink — SAM traces the ink, so they disagree even
   when SAM is "right" about the map.
2. **SAM grabs whole ink blocks.** On dense hatched blocks (Old Gaol, market
   shambles) a single prompt returns the entire connected hatch region or a
   street-length strip, not one building.
3. **Thin geometry punishes IoU.** The market shambles is a ~10 px-wide strip;
   a few-pixel lateral offset halves the IoU (0.081 despite SAM finding a
   visually similar strip).
4. **Georeferencing residual** (~6 m median ≈ 15 px) shifts everything slightly.

## What this means for the workflow
- The **speed** result is the win: one-time encode + sub-0.1 s prompts makes
  the QGIS Geo-SAM human-in-the-loop flow viable — click, get polygon, fix,
  accept. Even a wrong-ish polygon is a better starting point than a blank
  canvas.
- Vanilla SAM is a *tracing accelerant*, not a tracer. Expect to add/remove
  points per building and hand-fix corners.
- Upgrade paths: ViT-H checkpoint (better masks, slower encode), MapSAM-style
  fine-tuning on our own hand-traced pairs, negative points on neighboring
  buildings to split hatch blocks.

## Alignment verification (2026-08-23)

Report: "pretty bad misalignment" between `candidates_threshold.gpkg` and the
map ink in QGIS. Investigated with phase correlation (rasterized polygons vs
binarized ink, 5 well-separated 1024 px windows across the city core).

**Finding: there is no misalignment against the source raster.** Against
`tw_1762_philadelphia_map_clarkson_biddle_cog.tif` (the COG the polygons were
extracted from), measured offsets are dx,dy <= 0.02 px = **under 1 cm** in
every window, vs both the morphed mask and the raw ink. The
`src.window_transform(win)` -> `rasterio.features.shapes` pipeline is
pixel-exact. Proof crops (green outlines over the raster):

- `overlay_registration_2nd_market.png` (2nd & Market)
- `overlay_registration_4th_chestnut.png` (4th & Chestnut / Willings Alley)
- `overlay_registration_front_pine.png` (Front & Pine wharves)

**Likely cause of the report:** overlaying these 1762-derived candidates on a
*different* Philadelphia COG. Each historical map was independently
GCP-warped, so they disagree with each other by a few meters, *non-uniformly*
(varies in sign and magnitude across the sheet — it is a relative warp, not a
constant translation, so no global shift can fix it). See
`overlay_candidates_on_easburn1776v2.png`: the same polygons sit visibly a few
meters off the Easburn 1776 ink — exactly the reported symptom. Reproduce
measurements with `pilot/measure_offset.py` and `pilot/verify_alignment.py`.

Practical notes for QGIS:
- Compare `candidates_threshold.gpkg` only against
  `tw_1762_philadelphia_map_clarkson_biddle_cog.tif` (local file must match
  the NYC_Maps Gitea LFS copy).
- Load **both** layers: `candidates` (10-2000 m2 shapes) *and*
  `oversize_blocks` — contiguous hatched row-blocks larger than 2000 m2 live
  in the second layer, so blocks can look "missing" if only `candidates` is
  loaded. Note `candidates` still includes street-name letterforms and other
  ink false-positives; it is a candidate layer, not a finished footprint
  layer.
