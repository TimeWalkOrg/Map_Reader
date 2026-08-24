#!/usr/bin/env python3
"""Render high-zoom QC overlays of candidates_v3.gpkg on the source COG.

Areas chosen to prove the four v3 fixes: straight edges, no text, no trees,
no ships/river blobs.

Run:  env/bin/python pilot/render_v3_qc.py
"""
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(HERE, "results", "candidates_v3.gpkg")

# (name, preview-px center (x8 overview), half-size in full-res px)
AREAS = [
    ("waterfront_oldferryslip", (627, 448), 450),
    ("text_street_label",       (555, 505), 300),
    ("trees_publick_square",    (350, 300), 400),
    ("dense_blocks",            (430, 620), 400),
    ("north_edge_gardens",      (240, 645), 450),
]


def main():
    src = rasterio.open(bc.COG)
    cands = gpd.read_file(GPKG, layer="candidates")
    blocks = gpd.read_file(GPKG, layer="oversize_blocks")
    core = gpd.read_file(GPKG, layer="core_v3").geometry[0]

    for name, (pcx, pcy), half in AREAS:
        ccol, crow = pcx * 8, pcy * 8
        r0, r1 = max(0, crow - half), min(src.height, crow + half)
        c0, c1 = max(0, ccol - half), min(src.width, ccol + half)
        win = ((r0, r1), (c0, c1))
        d = src.read(window=win)
        g = (0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]).astype(np.uint8)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        tr = src.window_transform(win)
        inv = ~tr

        def draw(geom, color, thick):
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in polys:
                rings = [p.exterior] + list(p.interiors)
                for ring in rings:
                    pts = np.array([inv * (x, y) for x, y in ring.coords])
                    cv2.polylines(vis, [np.round(pts).astype(np.int32)],
                                  True, color, thick, cv2.LINE_AA)

        x0, y0 = tr * (0, r1 - r0)
        x1, y1 = tr * (c1 - c0, 0)
        from shapely.geometry import box as sbox
        bbox = sbox(x0, y0, x1, y1)
        for geom in blocks.geometry:
            if geom.intersects(bbox):
                draw(geom, (0, 140, 255), 2)   # orange: oversize blocks
        for geom in cands.geometry:
            if geom.intersects(bbox):
                draw(geom, (0, 200, 0), 2)     # green: candidates
        if core.intersects(bbox) and not core.contains(bbox):
            draw(core.intersection(bbox.buffer(50)).intersection(bbox),
                 (0, 0, 255), 3)               # red: core_v3 boundary

        out = os.path.join(HERE, "results", f"qc_v3_{name}.png")
        cv2.imwrite(out, vis)
        print(out)


if __name__ == "__main__":
    main()
