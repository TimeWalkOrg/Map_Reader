#!/usr/bin/env python3
"""Measure registration offset between candidates_threshold.gpkg polygons
and the source COG ink, via phase correlation in several sample windows."""
import sys, os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features as rfeatures
import geopandas as gpd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

GPKG = (sys.argv[1] if len(sys.argv) > 1 else
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "candidates_threshold.gpkg"))

src = rasterio.open(bc.COG)
gdf = gpd.read_file(GPKG, layer="candidates")
sidx = gdf.sindex

# Sample window centers (col, row) at full res, spread over city core.
# Core spans roughly cols 1500-5700, rows 300-11500 (from preview poly *8).
centers = [
    (2400, 2400),   # NW core
    (4800, 1600),   # NE / Northern Liberties
    (3200, 5600),   # center
    (4800, 6400),   # E center near Dock
    (2400, 8000),   # SW
    (4400, 9600),   # S Southwark
]
W = 1024

print(f"{'window':>14} {'npoly':>6} {'dx_px':>7} {'dy_px':>7} {'dx_m':>7} {'dy_m':>7} {'resp':>6}")
results = []
for (cc, cr) in centers:
    c0, r0 = cc - W // 2, cr - W // 2
    win = Window(c0, r0, W, W)
    data = src.read(window=win)
    gray = bc.gray_from_tile(data)
    ink = bc.ink_mask(gray)
    # same morphology as generator so shapes match what was polygonized
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, ck)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ok)

    wt = src.window_transform(win)
    # window bounds in world coords
    x0, y1 = wt * (0, 0)
    x1, y0 = wt * (W, W)
    from shapely.geometry import box as sbox
    cand = gdf.iloc[list(sidx.intersection((x0, y0, x1, y1)))]
    if len(cand) < 5:
        print(f"({cc},{cr}) skip: only {len(cand)} polys")
        continue
    poly_r = rfeatures.rasterize(
        ((g, 1) for g in cand.geometry), out_shape=(W, W),
        transform=wt, fill=0, dtype="uint8")

    a = m.astype(np.float32)
    b = poly_r.astype(np.float32)
    # window to reduce edge effects
    hann = cv2.createHanningWindow((W, W), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * hann, b * hann)
    # phaseCorrelate returns shift of b relative to a in (x=col, y=row)
    dx_m = dx * 0.4
    dy_m = -dy * 0.4   # row shift -> world y is negative row
    print(f"({cc:5d},{cr:5d}) {len(cand):6d} {dx:7.2f} {dy:7.2f} {dx_m:7.2f} {dy_m:7.2f} {resp:6.3f}")
    results.append((dx, dy, resp))

if results:
    arr = np.array(results)
    print("\nmedian dx_px=%.2f dy_px=%.2f  -> dx_m=%.2f dy_m=%.2f" % (
        np.median(arr[:, 0]), np.median(arr[:, 1]),
        np.median(arr[:, 0]) * 0.4, -np.median(arr[:, 1]) * 0.4))
