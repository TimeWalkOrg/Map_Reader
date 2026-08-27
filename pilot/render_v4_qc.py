#!/usr/bin/env python3
"""Render high-zoom QC comparisons (v3 LEFT | v4 RIGHT) of the three v4
fixes from Sunil's 2026-08-27 screenshots: recall (hatched buildings v3
missed), dark-divider block splitting, and ink-clamped parcels that no
longer spill onto streets / street-name lettering.

Green = candidates, orange = oversize_blocks.

Run:  env/bin/python pilot/render_v4_qc.py
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
AREAS = [
    # hatched buildings v3 missed entirely, now captured (recall fix)
    ("recall_society_hill", (5350, 2480), 400),
    # merged block split into parcels along its dark divider lines
    ("split_dividers", (4609, 3926), 380),
    # v3 parcel overshot onto street + lettering; v4 clamped to the ink
    ("clamp_overshoot", (5132, 3588), 320),
    # v3 parcel spilled over street lettering near the wharves
    ("clamp_lettering", (4597, 9531), 320),
    # curved Dock street row: recall + no fragmentation regression
    ("dock_curved_row", (4327, 5650), 340),
]


def main():
    src = rasterio.open(bc.COG)
    layers = {}
    for ver in ("v3", "v4"):
        gpkg = os.path.join(HERE, "results", f"candidates_{ver}.gpkg")
        layers[ver] = (gpd.read_file(gpkg, layer="candidates"),
                       gpd.read_file(gpkg, layer="oversize_blocks"))

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
        for ver in ("v3", "v4"):
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            cands, blocks = layers[ver]

            def draw(geom, color, thick):
                polys = (geom.geoms if geom.geom_type == "MultiPolygon"
                         else [geom])
                for p in polys:
                    for ring in [p.exterior] + list(p.interiors):
                        pts = np.array([inv * (x, y) for x, y in ring.coords])
                        cv2.polylines(vis, [np.round(pts).astype(np.int32)],
                                      True, color, thick, cv2.LINE_AA)

            for geom in blocks.geometry:
                if geom.intersects(bbox):
                    draw(geom, (0, 140, 255), 2)   # orange: oversize
            for geom in cands.geometry:
                if geom.intersects(bbox):
                    draw(geom, (0, 200, 0), 2)     # green: candidates
            cv2.putText(vis, ver, (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
            panes.append(vis)

        out = np.hstack([panes[0],
                         np.full((2 * half, 10, 3), 255, np.uint8),
                         panes[1]])
        path = os.path.join(HERE, "results", f"qc_v4_{name}.png")
        cv2.imwrite(path, out)
        print(path)


if __name__ == "__main__":
    main()
