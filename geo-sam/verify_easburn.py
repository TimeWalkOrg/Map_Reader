#!/usr/bin/env python3
"""End-to-end check of the Easburn 1776 Geo-SAM feature cache.

For each of Sunil's 51 manual parcels (Map_Reader/training/
sunil_parcels_2026-09-07), issue ONE positive click at the parcel's
representative point against the pre-encoded features (exactly what the
plugin does in Pre-encoded mode), polygonize the mask, keep the piece under
the click, and compare it with the hand-drawn parcel (IoU, area ratio).
A second pass adds the parcel's bounding box as a box prompt.

This is an oracle-assisted plumbing + quality check (prompts derive from the
labels); it is NOT an independent detection benchmark.

Usage: env/bin/python verify_easburn.py [--features DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from geosam import BoundingBox, FeatureQueryEngine, Points, PromptSet
from geosam.runtime import create_model_spec_from_checkpoint

HERE = Path(__file__).parent
LABELS = (
    HERE.parent / "training" / "sunil_parcels_2026-09-07"
    / "1776_philadelphia_building_parcels.gpkg"
)


def mask_to_polygon_under(result, click: Point):
    mask = result.mask_array[0].astype(np.uint8)
    polys = [
        shape(g)
        for g, v in rasterio.features.shapes(mask, mask=mask.astype(bool), transform=result.mask_transform)
        if v == 1
    ]
    if not polys:
        return None, 0
    hit = [p for p in polys if p.covers(click)]
    return (unary_union(hit) if hit else max(polys, key=lambda p: p.area)), len(polys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=HERE / "features_easburn_sam_b" / "tw_1776_philadelphia_map_easburn_plan_v2_cog")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "reports" / "geosam-easburn-verify-2026-09-10")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = create_model_spec_from_checkpoint(HERE / "models" / "sam_b.pt", model_id="sam_b", device="cpu")
    engine = FeatureQueryEngine(args.features, spec)
    crs = engine.manifest.crs
    labels = gpd.read_file(LABELS).to_crs(crs).reset_index(drop=True)
    print(f"manifest: {len(engine.manifest)} chips, model {engine.manifest['model_type'].iloc[0]}; {len(labels)} parcels")

    rows, geoms = [], []
    t_all = time.time()
    try:
        for i, parcel in labels.geometry.items():
            click = parcel.representative_point()
            rec = {"parcel_idx": i, "source_row": i + 1, "manual_area_m2": round(parcel.area, 1)}
            for mode in ("point", "point+box"):
                pts = Points([[click.x, click.y]], labels=[1], crs=crs)
                if mode == "point":
                    q = pts
                else:
                    b = parcel.bounds
                    q = PromptSet(points=pts, bbox=BoundingBox(b[0], b[1], b[2], b[3], crs=crs))
                t0 = time.time()
                try:
                    res = engine.query(q)
                except ValueError as e:  # no covering chip
                    rec[f"{mode}_error"] = str(e)
                    continue
                dt = time.time() - t0
                poly, npieces = mask_to_polygon_under(res, click)
                if poly is None or poly.is_empty:
                    rec[f"{mode}_iou"] = 0.0
                    continue
                inter = poly.intersection(parcel).area
                union = poly.union(parcel).area
                rec.update({
                    f"{mode}_chip": res.chip_id,
                    f"{mode}_score": round(float(res.scores.max()), 3),
                    f"{mode}_query_s": round(dt, 3),
                    f"{mode}_sam_area_m2": round(poly.area, 1),
                    f"{mode}_area_ratio": round(poly.area / parcel.area, 3),
                    f"{mode}_iou": round(inter / union if union else 0.0, 3),
                    f"{mode}_pieces": npieces,
                })
                geoms.append({"source_row": i + 1, "mode": mode, "iou": rec[f"{mode}_iou"],
                              "score": rec[f"{mode}_score"], "geometry": poly})
            rows.append(rec)
            print(f"row {i+1:2d}  manual {rec['manual_area_m2']:7.1f} m²  "
                  f"point IoU {rec.get('point_iou', float('nan')):.3f} (x{rec.get('point_area_ratio', float('nan')):.2f})  "
                  f"point+box IoU {rec.get('point+box_iou', float('nan')):.3f}  chip {rec.get('point_chip')}", flush=True)
    finally:
        engine.close()

    df = gpd.pd.DataFrame(rows)
    summary = {
        "features": str(args.features),
        "chips_in_manifest": int(len(engine.manifest)),
        "model_type": str(engine.manifest["model_type"].iloc[0]),
        "parcels": int(len(labels)),
        "parcels_covered_by_a_chip": int(df["point_iou"].notna().sum()) if "point_iou" in df else 0,
        "total_seconds": round(time.time() - t_all, 1),
        "prompt_provenance": "representative point (+bbox) of each manual parcel; oracle-assisted, not a detection benchmark",
    }
    for mode in ("point", "point+box"):
        col = f"{mode}_iou"
        if col in df:
            s = df[col].dropna()
            summary[mode] = {
                "n": int(len(s)),
                "iou_mean": round(float(s.mean()), 3),
                "iou_median": round(float(s.median()), 3),
                "iou_ge_0.5": int((s >= 0.5).sum()),
                "iou_ge_0.7": int((s >= 0.7).sum()),
                "iou_ge_0.9": int((s >= 0.9).sum()),
                "area_ratio_median": round(float(df[f"{mode}_area_ratio"].dropna().median()), 3),
                "query_s_mean": round(float(df[f"{mode}_query_s"].dropna().mean()), 3),
            }
    (args.out / "verify_result.json").write_text(json.dumps({"summary": summary, "per_parcel": rows}, indent=2))
    gpd.GeoDataFrame(geoms, geometry="geometry", crs=crs).to_file(args.out / "sam_polygons.gpkg", driver="GPKG")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
