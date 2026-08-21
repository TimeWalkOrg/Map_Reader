#!/usr/bin/env python3
"""Run SAM (via segment-geospatial) on the cropped 1762 map window using
box prompts derived from the ground-truth landmark bboxes (padded 15%),
plus a foreground point at the building centroid — simulating the real
workflow of a user dragging a rough box and clicking the building.
(Box-only scored mean IoU 0.186; box+point 0.256 — point kept.)

Produces one georeferenced polygon per landmark in sam_output.gpkg.
"""
import json
import time
import numpy as np
import rasterio
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from samgeo import SamGeo

CROP = "crop_1762_core.tif"
GT = "ground_truth.geojson"
OUT_GPKG = "results/sam_output.gpkg"
PAD_FRAC = 0.15  # box prompt padding around ground-truth bbox

t0 = time.time()
sam = SamGeo(model_type="vit_b", automatic=False, sam_kwargs=None)
print(f"model load: {time.time()-t0:.1f}s")

sam.set_image(CROP)
print(f"set_image (encoder): {time.time()-t0:.1f}s total")

with rasterio.open(CROP) as src:
    transform = src.transform
    crs = src.crs
    inv = ~transform

with open(GT) as f:
    fc = json.load(f)

records = []
for feat in fc["features"]:
    fid = feat["properties"]["id"]
    name = feat["properties"]["name"]
    geom = shape(feat["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    px = (maxx - minx) * PAD_FRAC
    py = (maxy - miny) * PAD_FRAC
    # geographic box -> pixel box (col/row), y axis flips
    c0, r0 = inv * (minx - px, maxy + py)
    c1, r1 = inv * (maxx + px, miny - py)
    box = [c0, r0, c1, r1]
    cx, cy = inv * (geom.centroid.x, geom.centroid.y)

    t1 = time.time()
    masks, scores, _ = sam.predictor.predict(
        box=np.array(box),
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    dt = time.time() - t1
    mask = masks[0].astype(np.uint8)

    polys = [
        shape(g)
        for g, v in rfeatures.shapes(mask, mask=mask.astype(bool), transform=transform)
        if v == 1
    ]
    merged = unary_union(polys) if polys else None
    print(f"id={fid} {name}: score={scores[0]:.3f} predict={dt:.2f}s "
          f"pixels={int(mask.sum())}")
    records.append({"id": fid, "name": name, "sam_score": float(scores[0]),
                    "predict_s": round(dt, 2), "geometry": merged})

gdf = gpd.GeoDataFrame(records, crs=crs)
gdf.to_file(OUT_GPKG, driver="GPKG", layer="sam_footprints")
print(f"wrote {OUT_GPKG}; total wall clock {time.time()-t0:.1f}s")
