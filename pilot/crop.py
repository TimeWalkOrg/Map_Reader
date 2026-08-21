#!/usr/bin/env python3
"""Crop a working window from the 1762 Clarkson & Biddle COG around the
pilot landmark parcels (union bbox of ground_truth.geojson + padding).

Uses rasterio (bundled GDAL) because the Homebrew gdal_translate on this
Mac is broken (abseil dylib mismatch). Equivalent gdal command recorded
in the README.
"""
import json
import rasterio
from rasterio.windows import from_bounds

COG = "/Users/gabriel/NYC_Maps/maps/tw_1762_philadelphia_map_clarkson_biddle_cog.tif"
GT = "ground_truth.geojson"
OUT = "crop_1762_core.tif"
PAD = 80.0  # metres (EPSG:3857)

with open(GT) as f:
    fc = json.load(f)

xs, ys = [], []
for feat in fc["features"]:
    for poly in feat["geometry"]["coordinates"]:
        for ring in poly:
            for x, y in ring:
                xs.append(x)
                ys.append(y)

bounds = (min(xs) - PAD, min(ys) - PAD, max(xs) + PAD, max(ys) + PAD)
print("crop bounds (EPSG:3857):", bounds)

with rasterio.open(COG) as src:
    win = from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets().round_lengths()
    data = src.read(window=win)
    transform = src.window_transform(win)
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        width=win.width,
        height=win.height,
        transform=transform,
        compress="deflate",
        tiled=False,
    )
    # drop COG-specific creation options that plain GTiff writer rejects
    for k in ("blockxsize", "blockysize"):
        profile.pop(k, None)
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(data)

print(f"wrote {OUT}: {int(win.width)}x{int(win.height)} px, {data.shape[0]} bands")
