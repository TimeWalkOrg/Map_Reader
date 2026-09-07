# Philadelphia training-label selection — 2026-09-07

Sunil requested that his manually drawn `1776_philadelphia_building_parcels`
replace his Geo-SAM output as the training-label source. The authoritative
selection is [active_dataset.json](active_dataset.json). There is no existing
fine-tuning entry point in this repository to reconfigure; future training
must consume this selection and exclude Geo-SAM and machine candidates.
No model was trained or retrained by this change.

## Primary map

Sunil explicitly confirmed **Easburn** as the main map on 2026-09-07.
Use `tw_1776_philadelphia_map_easburn_plan_v2_cog.tif` with these labels.
Clarkson & Biddle is a legacy comparison source only; its pre-encoded
features, chip grids, and extraction extents must not be reused for Easburn.
Existing 1762 scripts are not an Easburn production pipeline.

## Verified source

- NYC_Maps main, commit `29539f115351119f882525cf7397b4296ceb68de` (Gitea, not GitHub).
- Source commit: “Philadelphia easburn plan parcel layer. 50 parcels for testing.”
- Actual count: **51**; EPSG:3857; all valid polygons, no empty or duplicate geometries.
- All original `id` values are null; preserve them and add 1-based `source_row`.
- Preserve the original geometry. Parcels 41 and 42 overlap by approximately
  **0.65 m²** in UTM 18N; flag for review rather than silently trimming them.
- Easburn v2 is referenced by the source QGIS project. Do not pair these
  polygons with Clarkson & Biddle Geo-SAM features. Verify the raster/grid
  before generating training masks.
- These are positive examples, **not exhaustive annotated tiles**. Unlabeled
  areas must not automatically become negative training pixels. Spatially
  split training/validation and keep the independent benchmark separate.

## Import package

[sunil_parcels_2026-09-07](sunil_parcels_2026-09-07/) contains the original
shapefile components, lossless GeoPackage, validation report, and transaction
SQL for `timewalk."1776_philadelphia_building_parcels"`.

**Database status: NOT IMPORTED.** A direct connection probe returned “no
password supplied”; no supported authenticated connection was available.
The protected credential request received no answer. No database or QGIS
project changes were made.

`prepare.py` regenerates the package locally using GeoPandas/Shapely. The SQL
creates a new table, preserves source metadata, verifies geometry/count,
and refuses a conflicting existing dataset instead of overwriting it.
An exact replay is a no-op. It enables RLS without adding public access;
QGIS's existing postgres login can access the table. Apply the SQL only via
an authenticated connection, then independently verify the stored geometry
against the packaged source before reporting completion.
