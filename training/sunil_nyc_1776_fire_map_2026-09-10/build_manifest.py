"""Assemble manifest.json from validation.json, qa_alignment.json, qa_cells.json, split_summary.json."""
import hashlib, json
from pathlib import Path
BASE = Path(__file__).resolve().parent
val = json.loads((BASE / "validation.json").read_text()); qa = json.loads((BASE / "qa_alignment.json").read_text())
cells = json.loads((BASE / "qa_cells.json").read_text()); split = json.loads((BASE / "split_summary.json").read_text())
RASTER = Path("/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif")
sha = hashlib.sha256(RASTER.read_bytes()).hexdigest()
m = {
  "dataset": "sunil_nyc_1776_fire_map_2026-09-10",
  "created_utc": "2026-09-10",
  "requested_by": "Sunil Prabhu (Slack C08BNQJQB1P msg 1789070222.107089, 2026-09-10)",
  "requester_claim": "vectors/parcels digitized over raster 1776_nyc_map_fire_map_upscaled; use as training data",
  "database": {"host": "db.hvlhsfqzjpnvmmqlighy.supabase.co", "schema": "timewalk", "access": "read-only SELECT; no tables modified",
               "snapshot_utc": val["snapshot_utc"], "export_method": "psql \\copy CSV (all columns, geom as hex EWKB, NULL as \\N) -> prepare.py -> GPKG"},
  "raster": {"qgis_layer_name": "1776_nyc_map_fire_map_upscaled", "qgis_layer_id": "1776_nyc_map_fire_map_upscaled_8a2698ef_ce50_4085_9d99_89750cc437fc",
             "qgis_datasource": "localized:raster/1776/1776_nyc_map_fire_map_upscaled.tif",
             "qgis_project_evidence": "same datasource in /Users/gabriel/NYC_Maps/TimeWalk_Maps.qgs and in timewalk.qgis_projects.TimeWalk_Maps (DB copy, last_modified 2026-09-01)",
             "file": str(RASTER), "repo": "Gitea TimeWalk/NYC_Maps (Git LFS)", "sha256": sha, "lfs_oid_matches": True,
             "size_bytes": RASTER.stat().st_size, "width": 8331, "height": 7788, "bands": 3, "dtype": "uint8", "crs": "EPSG:3857",
             "pixel_size_3857_m": qa["pixel_size_3857_m"], "ground_m_per_px_at_40_71N": qa["ground_m_per_px"],
             "origin_3857": [-8239760.364740431, 4970603.385257988], "bounds_3857": [-8239760.364740431, 4968350.326047873, -8237350.21627106, 4970603.385257988],
             "compression": "none", "tiled": False, "overviews": False,
             "plate_notes": "binarised black-on-white engraving (upscaled); off-plate region is pure black; lots drawn as tick-divided frontage cells, public buildings as solid blobs; burnt district overprinted with diagonal hatching"},
  "layers": {},
  "provenance_from_attributes": {
    "1776_nyc_parcels_buildings": "No geometry-provenance column. `source` describes ATTRIBUTE provenance only (period_defaults_v1/v2_stokes, 1786_NYC_Directory, great_fire_spatial_analysis, landmarks_table, Stokes_Iconography_IV...). `confidence_notes` = generated house numbers / inferred materials; `address_generated` true for 2658/2733. `timewalk_notes` mentions 'Updated by Claude via Supabase MCP'. Nothing states the footprints were traced from the fire-map plate.",
    "1776_nyc_parcels_landmarks": "`source` = literature (Stokes, Wikipedia, HMDB...) for 7 rows, empty for 55. Names and ids 1-52 follow the plate's own numbered key (1 Fire commenced ... 52 Hanover Square), which ties this layer to the fire-map plate.",
    "1776_nyc_vector_road_graph": "Esri CityEngine street-graph attributes (rulefile roads.cga/piers.cga, startrule, strtwdth, sdwlkwdthl/r, lanewidth, shape__id, randomseed, shpcrtn); 3-D MultiLineString Z. Exported from a CityEngine scene, not digitised in QGIS; no raster reference."},
  "qa": {"alignment": qa, "cells_and_coverage": cells, "proposed_split": split},
  "verdict": {
    "buildings": "NOT pixel-traced from this plate. Boundary-ink agreement is no better than a 4 m shifted baseline (gain median 0.00); IoU with the best drawn lot cell median 0.21, only 133/2507 on-plate polygons reach IoU>=0.5; polygons (median 118 ground m2) are ~3x the drawn cells (median 38 m2) and typically straddle 1-2+ cells; 140 sit on blank plate, 226 lie off the plate entirely. Placement is correct at block/frontage level (a few metres), so they are usable as WEAK location labels / instance priors, not as segmentation masks for the plate's footprint ink.",
    "landmarks": "Derived from the plate key (numbering matches). 24/61 sit on solid ink blobs (centroid offset median 1.2 m ground), 27 partial, 10 not on any blob (ids 17,25,26,38,39,44,54,60,63,72). Shapes are approximate rectangles, not traced outlines. id 54 (Fish Market) invalid (self-intersection).",
    "roads": "Centerlines follow the plate corridors: on real corridors (624 edges, width>=2 m) |offset| median 1.1 m ground, p90 2.9 m, 139 edges >2 m; 408 measured edges run through ink (water hatching, burnt-district hatching, piers) and cannot be scored; 558 edges too short/off-plate to measure. Suitable as a line-extraction set with a tolerance buffer, kept separate from the polygon task.",
    "coverage": "Only 10 of 261 road-graph block faces are exhaustively covered (>=90% of drawn cells overlapped) - 77,294 ground m2 holding 196 building polygons (18 of them traced-cell matches); 101 partial, 74 sparse, 39 no drawn cells, 37 off-plate. Overall 2126/4101 drawn cells (52%) are overlapped by any polygon. Negatives may only be sampled inside the 10 exhaustive blocks; elsewhere unlabeled ink is NOT confirmed negative."},
  "modern_proxy_check": {"osm_snapshot": "Overpass 2026-09-10T20:04:51Z, 3498 buildings, export/osm_buildings_2026-09-10.gpkg", "buildings_osm_iou_ge_0_5": qa["buildings"]["osm_iou_ge_0_5"], "buildings_flagged_modern_proxy_suspect": qa["buildings"]["flagged_modern_proxy_ids"], "landmarks_flagged": qa["landmarks"]["flagged_modern_proxy_ids"], "note": "OSM IoU median 0.03 - the polygons are not copies of modern footprints either; the 15 flagged are small OSM-coincident rectangles with poor ink support, review individually."},
  "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.glob("*")) if p.is_file() and p.name != "manifest.json"},
}
for name, v in val["layers"].items():
    m["layers"][name] = {k: v[k] for k in ("source_table", "primary_key", "row_count", "distinct_pk", "srid", "geometry_types", "has_z", "valid", "invalid_ids", "invalid_reasons", "empty", "exact_duplicate_geometries", "bounds_wgs84", "gpkg_sha256", "gpkg_geometry_roundtrip_exact", "gpkg_attribute_mismatch_cells")}
    m["layers"][name]["gpkg"] = f"{name}.gpkg"; m["layers"][name]["columns"] = v["columns"]
    for k in ("area_utm18n_m2", "overlapping_pairs_over_1_m2", "overlap_total_m2", "total_length_utm18n_m"):
        if k in v: m["layers"][name][k] = v[k]
(BASE / "manifest.json").write_text(json.dumps(m, indent=2))
print("manifest written", len(json.dumps(m)))
