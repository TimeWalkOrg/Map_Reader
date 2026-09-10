#!/usr/bin/env python3
"""Pre-encode the 1776 Easburn "Plan of the City of Philadelphia" v2 COG for
the QGIS Geo-SAM plugin (v2.0).

Sibling of encode_1762.py (Clarkson & Biddle). Replicates the plugin's
"Geo-SAM Image Encoder" processing algorithm 1:1 so the output feature cache
(manifest.parquet + <layer>/features/chip_*.pt) loads in the plugin's
Pre-encoded mode. Runs standalone (no QGIS) on the geosam PyPI library.

Easburn has its OWN extent and grid — nothing from the C&B package is reused.
Chip filter: chips (1024 px, stride 512, in the raster's native pixel grid)
that intersect DRAWN_CITY_PX, a hand-digitized polygon over the drawn city
(Northern Liberties → city core → Southwark incl. the wharf line and the
Fort), excluding the Delaware, Windmill Island, the Delaware Bay inset at
bottom-left, the blank margins and the black (alpha=0) collar of the rotated
sheet. Digitized 2026-09-10 on the /8 overview of the v2 COG
(scratch/ov8_top.png, ov8_bot.png); coordinates below are FULL-RES pixels.

Usage: env/bin/python encode_easburn.py [--limit N] [--stride 512]
                                        [--dry-run] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, box, mapping

from geosam import BoundingBox, RasterDataset, build_model_adapter
from geosam.runtime import (
    chip_extent_rectangles_for_source,
    create_model_spec_from_checkpoint,
)

HERE = Path(__file__).parent
CHECKPOINT = HERE / "models" / "sam_b.pt"
MODEL_ID = "sam_b"
# Same file as NYC_Maps/maps/... (LFS) — byte-identical, sha256 recorded in
# encode_easburn_run.json next to the output.
COG = (
    HERE.parent.parent
    / "philly_georef_work"
    / "tw_1776_philadelphia_map_easburn_plan_v2_cog.tif"
)
# Output layer folder name = sanitized raster layer name, exactly what the
# plugin would produce for a layer named after the file stem.
LAYER_NAME = "tw_1776_philadelphia_map_easburn_plan_v2_cog"
BANDS = [1, 2, 3]  # RGB of the RGBA COG
CHIP_SIZE = 1024

# Drawn-city polygon, full-resolution pixel (col, row) coordinates on the
# 8864 x 14614 v2 COG. Clockwise from the NW corner of the sheet.
DRAWN_CITY_PX: list[tuple[int, int]] = [
    (1080, 320),    # NW: sheet's top-left, north of Sassafras/Vine grid
    (5600, 160),    # NE: Northern Liberties wharves at the sheet's top edge
    (5600, 1120),   # wharf line (with ~300 px river buffer) going south …
    (5360, 2400),
    (5200, 3600),
    (5080, 4480),
    (4920, 5440),   # Old Ferry Slip / Crooked Billet
    (4720, 6320),   # The Dock
    (4600, 7040),
    (4800, 7680),   # Southwark wharves (Almond St)
    (4800, 9200),
    (4720, 10400),  # Catherine / Christian St wharves
    (4600, 11440),
    (4480, 12160),  # the Fort, south tip of Southwark
    (4160, 12160),  # sheet's bottom edge
    (2640, 11840),  # inset's bottom-right corner region
    (3160, 7560),   # inset's top-right corner (just south of Cedar St)
    (0, 7680),      # inset's top edge meets the sheet's west edge
    (0, 3840),      # west edge of the sheet
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drawn_city_polygon(src: rasterio.DatasetReader) -> Polygon:
    return Polygon([src.transform * (c, r) for c, r in DRAWN_CITY_PX])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="encode at most N chips (0 = all)")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", type=Path, default=HERE / "features_easburn_sam_b")
    ap.add_argument("--dry-run", action="store_true", help="only write the chip-grid GeoJSON")
    args = ap.parse_args()

    if not COG.exists():
        raise SystemExit(f"source COG not found: {COG}")

    with rasterio.open(COG) as src:
        city_poly = drawn_city_polygon(src)
        crs_text = src.crs.to_string()
        raster_shape = (src.width, src.height)

    rectangles = chip_extent_rectangles_for_source(
        str(COG),
        bands=BANDS,
        crs=crs_text,
        chip_size=CHIP_SIZE,
        stride=args.stride,
    )
    # rectangle order = (left, bottom, right, top)
    keep = [r for r in rectangles if box(r[0], r[1], r[2], r[3]).intersects(city_poly)]
    print(f"chips total={len(rectangles)} intersecting drawn-city poly={len(keep)}")

    out_dir = args.out / LAYER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = gpd.GeoDataFrame(
        {"chip_id": [f"chip_{i:06d}" for i in range(len(keep))]},
        geometry=[box(*r) for r in keep],
        crs=crs_text,
    )
    grid.to_file(args.out / "chip_grid.geojson", driver="GeoJSON")
    gpd.GeoDataFrame({"name": ["drawn_city"]}, geometry=[city_poly], crs=crs_text).to_file(
        args.out / "drawn_city.geojson", driver="GeoJSON"
    )
    if args.dry_run:
        return

    model_spec = create_model_spec_from_checkpoint(
        CHECKPOINT, model_id=MODEL_ID, device=None  # CPU
    )
    adapter = build_model_adapter(model_spec)
    if not model_spec.resolved_supports_feature_reuse:
        raise SystemExit("model does not support feature reuse")

    dataset = RasterDataset(str(COG), indexes=BANDS, crs=crs_text)
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    todo = keep[: args.limit] if args.limit else keep
    t0 = time.time()
    try:
        for index, rectangle in enumerate(todo):
            chip_bounds = BoundingBox(
                rectangle[0], rectangle[1], rectangle[2], rectangle[3],
                crs=dataset.crs,
            )
            sample = dataset[chip_bounds]
            model_image = sample.to_model_image(value_range=None)
            encoded = adapter.encode_image(model_image)
            chip_id = f"chip_{index:06d}"
            feature_path = features_dir / f"{chip_id}.pt"
            encoded.save(feature_path)
            rows.append(
                {
                    "feature_path": str(feature_path),
                    "chip_id": chip_id,
                    "source_path": sample.source_path,
                    "checkpoint_path": encoded.checkpoint_path,
                    "model_type": encoded.model_type,
                    "transform": json.dumps(list(sample.transform)[:6]),
                    "shape": json.dumps(list(sample.shape)),
                    "crs": sample.crs.to_string(),
                    "dst_shape": json.dumps(list(encoded.dst_shape)),
                    "chip_center_x": sample.bbox.center[0],
                    "chip_center_y": sample.bbox.center[1],
                    "geometry": sample.bbox.to_geometry(),
                }
            )
            del encoded, model_image, sample
            done = index + 1
            el = time.time() - t0
            print(
                f"[{done}/{len(todo)}] {chip_id}  {el/done:.1f}s/chip  "
                f"eta {el/done*(len(todo)-done)/60:.0f} min",
                flush=True,
            )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    elapsed = time.time() - t0
    manifest = gpd.GeoDataFrame(rows, geometry="geometry", crs=dataset.crs)
    manifest_path = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path)
    print(f"wrote {manifest_path} with {len(rows)} chips in {elapsed:.0f}s")

    run_info = {
        "source": str(COG),
        "source_sha256": sha256_of(COG),
        "source_size_px": raster_shape,
        "crs": crs_text,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_of(CHECKPOINT),
        "model_id": MODEL_ID,
        "chip_size": CHIP_SIZE,
        "stride": args.stride,
        "chips_total_grid": len(rectangles),
        "chips_encoded": len(rows),
        "encode_seconds": round(elapsed, 1),
        "seconds_per_chip": round(elapsed / max(len(rows), 1), 2),
        "drawn_city_px": DRAWN_CITY_PX,
        "drawn_city_geom": mapping(city_poly),
        "host": platform.node(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "encode_easburn_run.json").write_text(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
