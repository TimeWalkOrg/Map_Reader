#!/usr/bin/env python3
"""Pre-encode the 1762 Clarkson & Biddle COG for the QGIS Geo-SAM plugin (v2.0).

Replicates the plugin's "Geo-SAM Image Encoder" processing algorithm
(sam_processing_algorithm.py) 1:1 so the output feature cache
(manifest.parquet + <layer>/features/chip_*.pt) loads in the plugin's
Pre-encoded mode. Runs standalone (no QGIS) on the geosam PyPI library.

Extra vs the plugin: chips that do not intersect the hand-digitized
city-core polygon (batch_common.CORE_POLY — includes Southwark and the
Northern Liberties, excludes legend/cartouche/insets/river) are skipped,
so we don't burn CPU encoding blank paper.

Usage: env/bin/python encode_1762.py [--limit N] [--stride 512]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, box

from geosam import BoundingBox, RasterDataset, build_model_adapter
from geosam.runtime import (
    chip_extent_rectangles_for_source,
    create_model_spec_from_checkpoint,
)

HERE = Path(__file__).parent
COG = Path(
    "/Users/gabriel/.openclaw/workspace-timewalker/philly_georef_work/cb_v2/"
    "tw_1762_philadelphia_map_clarkson_biddle_v2_cog.tif"
)
CHECKPOINT = HERE / "models" / "sam_b.pt"
MODEL_ID = "sam_b"
# Output layer folder name = sanitized raster layer name, exactly what the
# plugin would produce for a layer named after the file stem.
LAYER_NAME = "tw_1762_philadelphia_map_clarkson_biddle_v2_cog"
BANDS = [1, 2, 3]  # RGB of the RGBA COG
CHIP_SIZE = 1024

# City-core polygon from Map_Reader/pilot/batch_common.py (digitized on the
# /8 overview of the v1 COG; v2 COG is pinned to the same grid).
sys.path.insert(0, str(HERE.parent / "pilot"))
import batch_common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="encode at most N chips (0 = all)")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", type=Path, default=HERE / "features_1762_sam_b")
    args = ap.parse_args()

    with rasterio.open(COG) as src:
        core_poly: Polygon = batch_common.load_core_poly(src)
        crs_text = src.crs.to_string()

    rectangles = chip_extent_rectangles_for_source(
        str(COG),
        bands=BANDS,
        crs=crs_text,
        chip_size=CHIP_SIZE,
        stride=args.stride,
    )
    # rectangle order = (left, bottom, right, top)
    keep = [r for r in rectangles if box(r[0], r[1], r[2], r[3]).intersects(core_poly)]
    print(f"chips total={len(rectangles)} intersecting core poly={len(keep)}")

    model_spec = create_model_spec_from_checkpoint(
        CHECKPOINT, model_id=MODEL_ID, device=None  # CPU
    )
    adapter = build_model_adapter(model_spec)
    if not model_spec.resolved_supports_feature_reuse:
        raise SystemExit("model does not support feature reuse")

    dataset = RasterDataset(str(COG), indexes=BANDS, crs=crs_text)

    out_dir = args.out / LAYER_NAME
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

    manifest = gpd.GeoDataFrame(rows, geometry="geometry", crs=dataset.crs)
    manifest_path = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path)
    print(f"wrote {manifest_path} with {len(rows)} chips in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
