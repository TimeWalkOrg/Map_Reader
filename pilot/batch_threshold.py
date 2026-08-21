#!/usr/bin/env python3
"""No-ML baseline candidate generator (NYPL map-vectorizer style):
ink extraction by grayscale threshold + morphological cleanup, then
polygonization, over the city-core extent of the 1762 Clarkson & Biddle
COG. Output: pilot/results/candidates_threshold.gpkg

Run:  env/bin/python pilot/batch_threshold.py
"""
import os
import sys
import time

import cv2
import numpy as np
import rasterio
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape, Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

STRIDE = 2048
MARGIN = 128
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "candidates_threshold.gpkg")

CLOSE_K = 3   # px: solidify hatched fills
OPEN_K = 5    # px: remove street lines / text strokes (5 px ~= 2 m)


def main():
    t0 = time.time()
    src = rasterio.open(bc.COG)
    core = bc.load_core_poly(src)
    tiles = list(bc.iter_tiles(src, core, STRIDE, MARGIN))
    print(f"{len(tiles)} tiles (stride {STRIDE}px, margin {MARGIN}px)", flush=True)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_K, CLOSE_K))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))

    records = []
    oversize = []   # contiguous rows/blocks > MAX_AREA: kept in a side layer
    for i, (tid, win, own) in enumerate(tiles):
        t1 = time.time()
        data = src.read(window=win)
        gray = bc.gray_from_tile(data)
        ink = bc.ink_mask(gray)
        frac = float(ink.mean())
        if frac < bc.MIN_INK_FRACTION:
            print(f"[{i+1}/{len(tiles)}] {tid} skip (ink {frac:.4f})", flush=True)
            continue
        m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        transform = src.window_transform(win)
        kept = 0
        for g, v in rfeatures.shapes(m, mask=m.astype(bool), transform=transform):
            if v != 1:
                continue
            geom = shape(g)
            rp = geom.representative_point()
            if not own.contains(rp):     # tile-ownership dedupe
                continue
            if not core.intersects(geom):
                continue
            if not bc.passes_filters(geom):
                if bc.MAX_AREA < geom.area < 50000:
                    oversize.append({
                        "tile": tid,
                        "area_m2_3857": round(geom.area, 1),
                        "geometry": geom.simplify(0.4, preserve_topology=True),
                    })
                continue
            geom = geom.simplify(0.4, preserve_topology=True)
            records.append({
                "tile": tid,
                "area_m2_3857": round(geom.area, 1),
                "geometry": geom,
            })
            kept += 1
        print(f"[{i+1}/{len(tiles)}] {tid} ink={frac:.3f} kept={kept} "
              f"({time.time()-t1:.1f}s)", flush=True)

    gdf = gpd.GeoDataFrame(records, crs=src.crs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gdf.to_file(OUT, driver="GPKG", layer="candidates")
    if oversize:
        gpd.GeoDataFrame(oversize, crs=src.crs).to_file(
            OUT, driver="GPKG", layer="oversize_blocks")
    print(f"wrote {OUT}: {len(gdf)} candidates + {len(oversize)} oversize "
          f"blocks, total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
