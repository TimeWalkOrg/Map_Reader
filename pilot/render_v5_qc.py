#!/usr/bin/env python3
"""Render before/after QC crops (v4 LEFT | v5 RIGHT) for the v5 vision-audit
pass, centered on the exact locations of Sunil's 2026-08-27 QC screenshots
(recovered by multi-scale template matching against the COG, scores
0.80-0.91) plus optional extra audit locations.

Green = candidates, orange = oversize_blocks, red dashed = needs_review.

Run:  env/bin/python pilot/render_v5_qc.py
"""
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box as sbox
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, full-res px center (col, row), half-size px)
# first five = Sunil's 2026-08-27 screenshots, template-matched to the sheet
AREAS = [
    ("sunil_a49f_gap_polygons", (3634, 5167), 300),
    ("sunil_b14c_bar_over_street", (3358, 5115), 300),
    ("sunil_bdaa_missing_hatch", (3316, 5205), 260),
    # dd05: raw template match was fooled by blank paper; re-located by
    # matching the skewed orange oversize block itself (block idx 25, v4)
    ("sunil_dd05_skewed_oversize", (4379, 5038), 300),
    ("sunil_fb9f_gap_span", (3213, 5051), 300),
    # densest recall cluster from the audit (7 buildings added, T1306)
    ("recall_society_hill_south", (3470, 6962), 300),
]


def main():
    src = rasterio.open(bc.COG)
    import pyogrio
    layers = {}
    for ver in ("v4", "v5"):
        gpkg = os.path.join(HERE, "results", f"candidates_{ver}.gpkg")
        have = {l[0] for l in pyogrio.list_layers(gpkg)}
        layers[ver] = {
            ln: gpd.read_file(gpkg, layer=ln)
            for ln in ("candidates", "oversize_blocks", "needs_review")
            if ln in have}

    for name, (ccol, crow), half in AREAS:
        r0, c0 = crow - half, ccol - half
        win = Window(c0, r0, 2 * half, 2 * half)
        g = bc.gray_from_tile(src.read(window=win))
        tr = src.window_transform(win)
        inv = ~tr
        x0, y0 = tr * (0, 2 * half)
        x1, y1 = tr * (2 * half, 0)
        bbox = sbox(x0, y0, x1, y1)

        panes = []
        for ver in ("v4", "v5"):
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

            def draw(geom, color, thick, dashed=False):
                polys = (geom.geoms if geom.geom_type == "MultiPolygon"
                         else [geom])
                for p in polys:
                    if p.geom_type != "Polygon":
                        continue
                    for ring in [p.exterior] + list(p.interiors):
                        pts = np.array([inv * (x, y)
                                        for x, y in ring.coords])
                        ipts = np.round(pts).astype(np.int32)
                        if dashed:
                            for k in range(0, len(ipts) - 1):
                                if k % 2 == 0:
                                    cv2.line(vis, tuple(ipts[k]),
                                             tuple(ipts[k + 1]), color,
                                             thick, cv2.LINE_AA)
                        else:
                            cv2.polylines(vis, [ipts], True, color, thick,
                                          cv2.LINE_AA)

            L = layers[ver]
            for lname, color, thick, dash in (
                    ("oversize_blocks", (0, 140, 255), 2, False),
                    ("candidates", (0, 200, 0), 2, False),
                    ("needs_review", (0, 0, 255), 1, True)):
                gdf = L.get(lname)
                if gdf is None:
                    continue
                for geom in gdf.geometry:
                    if geom is not None and geom.intersects(bbox):
                        draw(geom, color, thick, dashed=dash)
            cv2.putText(vis, ver, (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
            panes.append(vis)

        out = np.hstack([panes[0],
                         np.full((2 * half, 10, 3), 255, np.uint8),
                         panes[1]])
        path = os.path.join(HERE, "results", f"qc_v5_{name}.png")
        cv2.imwrite(path, out)
        print(path)


if __name__ == "__main__":
    main()
