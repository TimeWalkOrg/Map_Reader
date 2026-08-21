#!/usr/bin/env python3
"""Shared plumbing for the overnight candidate-vector batch runs over the
full 1762 Clarkson & Biddle COG (EPSG:3857).

Provides:
- CORE_POLY: city-core extent polygon (map coords). Digitized from an /8
  overview of the COG; excludes the REFERENCES legend (top-left), the two
  Schuylkill-Delaware inset maps (bottom-left), the cartouche
  (bottom-right), and most of the Delaware River/ships.
- Tile iteration with core-stride ownership dedupe (each output polygon is
  kept by exactly one tile: the one whose stride cell contains its
  representative point; the read window extends `margin` px beyond the
  cell so boundary buildings are never cut).
- Ink mask + blank-tile skip.
- Area/shape filters per Ted's spec: keep 10..2000 m^2 in EPSG:3857 terms,
  drop extreme slivers.
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon, box

COG = "/Users/gabriel/NYC_Maps/maps/tw_1762_philadelphia_map_clarkson_biddle_cog.tif"

INK_THRESHOLD = 165      # gray < this = ink (paper peak is 200-224)
MIN_INK_FRACTION = 0.005  # skip tiles with less ink than this (blank paper)
MIN_AREA = 10.0          # m^2, EPSG:3857 terms
MAX_AREA = 2000.0        # m^2, EPSG:3857 terms
MIN_WIDTH = 2.0          # m, min side of minimum rotated rectangle (sliver)
MAX_ASPECT = 12.0        # max length/width of min rotated rectangle

# City-core polygon, digitized on the /8 overview (preview px), clockwise
# from 8th & Vine. Converted to map coords via the COG transform at import.
_PREVIEW_PX = [
    (240, 222),   # NW: 8th & Vine
    (455, 210),   # Vine at legend bottom-right corner
    (470, 45),    # legend right edge, top of built area
    (650, 155),   # NE corner: shore at north sheet edge
    (705, 240),   # Northern Liberties wharves bulge
    (685, 320),
    (668, 390),
    (650, 440),   # Austins Ferry
    (620, 555),
    (610, 610),   # Crooked Billet
    (600, 700),
    (590, 770),   # The Dock
    (600, 830),
    (590, 950),   # shore south of Cedar
    (590, 1030),
    (585, 1130),
    (580, 1230),
    (575, 1330),
    (572, 1440),  # S tip: the Fort (cartouche stays east)
    (480, 1425),
    (428, 1315),  # Wicaco hamlet
    (445, 1150),  # W edge Southwark
    (450, 1000),
    (480, 902),   # 2nd & Cedar (inset maps lie SW of here)
    (215, 848),   # SW: 8th & Cedar
    (190, 600),   # W edge of grid
    (205, 390),
]
_PREVIEW_SCALE = 8


def load_core_poly(src):
    pts = []
    for cx, cy in _PREVIEW_PX:
        col = cx * _PREVIEW_SCALE + _PREVIEW_SCALE / 2
        row = cy * _PREVIEW_SCALE + _PREVIEW_SCALE / 2
        x, y = src.transform * (col, row)
        pts.append((x, y))
    return Polygon(pts)


def gray_from_tile(data):
    """data: (bands, h, w) uint8 -> grayscale uint8. Band 4 (alpha) marks
    the rotated sheet; outside-alpha pixels forced to white (no ink)."""
    g = (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]).astype(np.uint8)
    if data.shape[0] >= 4:
        g = np.where(data[3] > 0, g, 255).astype(np.uint8)
    return g


def ink_mask(gray, threshold=INK_THRESHOLD):
    return (gray < threshold).astype(np.uint8)


def iter_tiles(src, core_poly, stride_px, margin_px):
    """Yield (tile_id, read_window, own_box) for stride cells intersecting
    the core polygon. own_box is the stride cell in map coords; a polygon
    belongs to the tile whose own_box contains its representative point."""
    ncols = int(np.ceil(src.width / stride_px))
    nrows = int(np.ceil(src.height / stride_px))
    for tr in range(nrows):
        for tc in range(ncols):
            c0 = tc * stride_px
            r0 = tr * stride_px
            c1 = min(c0 + stride_px, src.width)
            r1 = min(r0 + stride_px, src.height)
            x0, y0 = src.transform * (c0, r1)  # lower-left
            x1, y1 = src.transform * (c1, r0)  # upper-right
            own = box(x0, y0, x1, y1)
            if not own.intersects(core_poly):
                continue
            rc0 = max(0, c0 - margin_px)
            rr0 = max(0, r0 - margin_px)
            rc1 = min(src.width, c1 + margin_px)
            rr1 = min(src.height, r1 + margin_px)
            win = Window(rc0, rr0, rc1 - rc0, rr1 - rr0)
            yield f"r{tr:02d}c{tc:02d}", win, own


def passes_filters(geom):
    """Area + sliver filters in EPSG:3857 terms."""
    a = geom.area
    if a < MIN_AREA or a > MAX_AREA:
        return False
    mrr = geom.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    s1 = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
    s2 = ((xs[2] - xs[1]) ** 2 + (ys[2] - ys[1]) ** 2) ** 0.5
    w, l = min(s1, s2), max(s1, s2)
    if w < MIN_WIDTH:
        return False
    if w > 0 and l / w > MAX_ASPECT:
        return False
    return True


def count_tiles(src, core_poly, stride_px, margin_px):
    return sum(1 for _ in iter_tiles(src, core_poly, stride_px, margin_px))
