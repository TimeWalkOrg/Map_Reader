#!/usr/bin/env python3
"""1) Phase-correlate candidates vs RAW ink of the source Clarkson COG.
2) Phase-correlate the Clarkson ink vs other Philly COGs (Easburn/Reed/
   Montresor/Hills) in the same world windows -> inter-map georef offsets.
3) Render high-zoom overlay PNGs (polygon outlines over raster)."""
import sys, os
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio import features as rfeatures
import geopandas as gpd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GPKG = os.path.join(RES, "candidates_threshold.gpkg")
MAPS = "/Users/gabriel/NYC_Maps/maps"
OTHERS = {
    "easburn1776": f"{MAPS}/tw_1776_philadelphia_map_easburn_plan_cog.tif",
    "easburn1776v2": f"{MAPS}/tw_1776_philadelphia_map_easburn_plan_v2_cog.tif",
    "reed1774": f"{MAPS}/tw_1774_philadelphia_map_reed_city_liberties_cog.tif",
    "montresor1777v2": f"{MAPS}/tw_1777_philadelphia_map_montresor_survey_v2_cog.tif",
    "hills1796": f"{MAPS}/tw_1796_philadelphia_map_hills_plan_cog.tif",
}

src = rasterio.open(bc.COG)
gdf = gpd.read_file(GPKG, layer="candidates")
sidx = gdf.sindex
W = 1024
centers = [(2400, 2400), (4800, 1600), (3200, 5600), (4800, 6400), (4400, 9600)]

def ink_of(ds, x0, y0, x1, y1, shape):
    win = from_bounds(x0, y0, x1, y1, ds.transform)
    data = ds.read(window=win, out_shape=(ds.count,)+shape, boundless=True, fill_value=255)
    g = (0.299*data[0] + 0.587*data[1] + 0.114*data[2]).astype(np.uint8)
    if data.shape[0] >= 4:
        g = np.where(data[3] > 0, g, 255).astype(np.uint8)
    return (g < bc.INK_THRESHOLD).astype(np.float32)

print("== candidates vs RAW Clarkson ink ==")
hann = cv2.createHanningWindow((W, W), cv2.CV_32F)
for (cc, cr) in centers:
    win = Window(cc-W//2, cr-W//2, W, W)
    wt = src.window_transform(win)
    x0, y1 = wt * (0, 0); x1, y0 = wt * (W, W)
    raw = ink_of(src, x0, y0, x1, y1, (W, W))
    cand = gdf.iloc[list(sidx.intersection((x0, y0, x1, y1)))]
    pr = rfeatures.rasterize(((g,1) for g in cand.geometry), out_shape=(W,W),
                             transform=wt, fill=0, dtype="uint8").astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(raw*hann, pr*hann)
    print(f"  ({cc},{cr}) dx={dx*0.4:+.2f}m dy={-dy*0.4:+.2f}m resp={resp:.3f}")

print("\n== Clarkson ink vs other COGs (same world window; + = other map shifted E/N) ==")
for name, path in OTHERS.items():
    if not os.path.exists(path):
        continue
    ds = rasterio.open(path)
    offs = []
    for (cc, cr) in centers[:4]:
        win = Window(cc-W//2, cr-W//2, W, W)
        wt = src.window_transform(win)
        x0, y1 = wt * (0, 0); x1, y0 = wt * (W, W)
        a = ink_of(src, x0, y0, x1, y1, (W, W))
        b = ink_of(ds, x0, y0, x1, y1, (W, W))
        if b.mean() < 0.005:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(a*hann, b*hann)
        if resp > 0.05:
            offs.append((dx*0.4, -dy*0.4, resp))
    if offs:
        arr = np.array(offs)
        print(f"  {name}: median dx={np.median(arr[:,0]):+.2f}m dy={np.median(arr[:,1]):+.2f}m over {len(offs)} windows "
              f"(per-window: " + ", ".join(f"({r[0]:+.1f},{r[1]:+.1f})" for r in offs) + ")")
    else:
        print(f"  {name}: no correlatable ink in test windows")

print("\n== overlay PNGs ==")
crops = [("overlay_registration_2nd_market", 4700, 3550), ("overlay_registration_4th_chestnut", 3500, 5400),
         ("overlay_registration_front_pine", 4900, 7300)]
Z = 700
for name, cc, cr in crops:
    win = Window(cc-Z//2, cr-Z//2, Z, Z)
    data = src.read(window=win)
    img = np.dstack([data[0], data[1], data[2]])
    wt = src.window_transform(win)
    x0, y1 = wt * (0, 0); x1, y0 = wt * (Z, Z)
    cand = gdf.iloc[list(sidx.intersection((x0, y0, x1, y1)))]
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    canvas = cv2.resize(canvas, (Z*2, Z*2), interpolation=cv2.INTER_LANCZOS4)
    inv = ~wt
    for g in cand.geometry:
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for p in polys:
            for ring in [p.exterior] + list(p.interiors):
                pts = np.array([inv * (x, y) for x, y in ring.coords])
                pts = np.round(pts * 2).astype(np.int32)
                cv2.polylines(canvas, [pts], True, (0, 200, 0), 2, cv2.LINE_AA)
    out = os.path.join(RES, f"{name}.png")
    cv2.imwrite(out, canvas)
    print(f"  {out}")
