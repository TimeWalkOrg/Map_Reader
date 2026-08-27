#!/usr/bin/env python3
"""Post-fix cleanup for v5 vision-added polygons: some additions clamp
around lettering flourishes / street-line tails attached to the real hatch
core (spiky outlines with many vertices). Re-run the proven v4
hatch-core-trim on every ADDED polygon and re-regularize when it fires.
Evidence-gated: only trims when the removed part is itself non-hatched
(v4.hatch_core_trim semantics); if the trim would change the polygon
beyond recognition (IoU < 0.30), the polygon is flagged needs_review
instead of altered.

Run: env/bin/python pilot/postfix_v5_trim.py
"""
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc
import extract_candidates_v4 as v4
from apply_fixes_v5 import region_from, rast, largest_poly

HERE = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(HERE, "results", "candidates_v5.gpkg")


def main():
    src = rasterio.open(bc.COG)
    import pyogrio
    have = {l[0] for l in pyogrio.list_layers(GPKG)}
    layers = {ln: gpd.read_file(GPKG, layer=ln) for ln, _ in
              [(l, None) for l in have]}
    cands = layers["candidates"]
    theta = v4.grid_theta_from(
        [{"area": g.area, "geom": g} for g in cands.geometry])

    n_trim = n_flag = 0
    review_rows = []
    for i, row in cands.iterrows():
        if row.get("v5_action") != "added":
            continue
        g = row.geometry
        reg = region_from(src, g.bounds, pad_m=6.0)
        if reg is None:
            continue
        pm = (rast(g, reg) > 0) & (reg["mask"] > 0)
        pm = pm.astype(np.uint8)
        if not pm.any():
            continue
        trimmed_mask, fired = v4.hatch_core_trim(pm, reg["rhatch"])
        if not fired:
            continue
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        p = cv2.morphologyEx(trimmed_mask, cv2.MORPH_CLOSE, k3)
        p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3)
        gs = [shape(gg) for gg, _ in rfeatures.shapes(
            p, mask=p.astype(bool), transform=reg["tr"])]
        if not gs:
            continue
        core_geom = max(gs, key=lambda q: q.area)
        if core_geom.area < bc.MIN_AREA:
            continue
        geo, method = v4.regularize(core_geom, theta)
        geo, method = v4.clamp_to_ink(geo, core_geom, method)
        if not geo.is_valid or geo.is_empty or geo.area < bc.MIN_AREA:
            continue
        iou = geo.intersection(g).area / max(geo.union(g).area, 1e-9)
        if iou < 0.30:
            review_rows.append({
                "category": "shape_mismatch",
                "note": "v5 addition trims to a very different core; verify",
                "pass": 4, "confidence": "medium", "geometry": g})
            n_flag += 1
            continue
        cands.at[i, "geometry"] = geo
        cands.at[i, "area_m2_3857"] = round(geo.area, 1)
        cands.at[i, "reg_method"] = "v5vision+trim+" + method
        cands.at[i, "v5_action"] = "added+trim"
        n_trim += 1

    layers["candidates"] = cands
    if review_rows:
        extra = gpd.GeoDataFrame(review_rows, crs=cands.crs)
        if "needs_review" in layers:
            layers["needs_review"] = gpd.GeoDataFrame(
                np.nan, index=[], columns=[]) if False else \
                gpd.GeoDataFrame(
                    __import__("pandas").concat(
                        [layers["needs_review"], extra], ignore_index=True),
                    crs=cands.crs)
        else:
            layers["needs_review"] = extra

    os.remove(GPKG)
    order = ["candidates", "oversize_blocks", "needs_review", "core_v4"]
    for ln in order:
        if ln in layers and len(layers[ln]):
            layers[ln].to_file(GPKG, driver="GPKG", layer=ln)
    print(f"trimmed {n_trim} added polygons; flagged {n_flag} for review")


if __name__ == "__main__":
    main()
