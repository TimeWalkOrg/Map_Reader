#!/usr/bin/env python3
"""Candidate building-footprint extraction v4 for the 1762 Clarkson & Biddle
COG. Addresses tester (Sunil) feedback on candidates_v3.gpkg (2026-08-27,
QC screenshots): (a) a few hatched buildings still not parceled at all,
(b) merged blocks that should be split where a dark ink line divides them,
(c) a couple of parcels overshooting the building ink onto streets and
street-name lettering.

Changes over v3:

1. RECALL - the hatch band was measured only on part of the sheet; the
   full-sheet gradient-orientation histogram peaks at 155..170 with mass up
   to ~174, so v3's 140..165 band clipped the top of the real hatch peak and
   missed finer/steeper-hatched buildings. v4 widens the band to 138..174
   (italic script strokes sit at 125..135; grid linework at ~9.5/~99.5 -
   both still excluded) and lowers the tensor energy floor 4000 -> 2500 so
   faint fine hatching still registers (paper noise energy is ~180).
   Additionally, small hatched houses of 45..80 m2 are now kept when the
   hatch evidence is strong (hatch_frac >= 0.40); v3 dropped everything
   under 80 m2 unconditionally.

2. DARK-DIVIDER BLOCK SPLITTING - components larger than 350 m2 are cut
   along internal grid-aligned dark linework before any other processing.
   Divider pixels: structure-tensor gradient orientation within +-8 deg of
   the street-grid directions (9.5 / 99.5 deg; hatch sits at 138..174 so the
   two signatures do not mix), coherence > 0.5, energy > 4000, gray < 135,
   then filtered by an oriented line-opening (15 px line kernels along the
   two grid directions) so isolated speckle on hatch strokes cannot cut.
   The component is cut where dividers cross it; resulting pieces >= 55 m2
   become separate parcels and the divider pixels themselves are assigned
   to the nearest piece (label dilation), so pieces tile the block exactly.
   Pieces still over 2000 m2 fall through to the v3 neck-split.

3. INK CLAMP (no more parcels on street lettering) - two mechanisms:
   a. hatch-core trim: if an accepted piece carries a non-hatched appendage
      (attached street line, lettering, wharf tail) amounting to 4..45% of
      its pixels, the piece is trimmed to its (closed) hatched core.
   b. post-regularization clamp: a regularized polygon that overhangs the
      source ink outline by more than 7% of its area (beyond a 1.2 m
      tolerance) is intersected back with the ink outline (0.5 m mitre
      buffer) - reg_method gains a '+clamp' suffix.

Output: pilot/results/candidates_v4.gpkg (EPSG:3857)
  layer candidates      - regularized building footprints
  layer oversize_blocks - hatched row-blocks/blocks 2000..50000 m2
  layer core_v4         - the tightened city-core polygon used for clipping

Run:  env/bin/python pilot/extract_candidates_v4.py   (~1 min)
"""
import os
import sys
import time

import cv2
import numpy as np
import rasterio
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape, Polygon, box
from shapely import affinity
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "candidates_v4.gpkg")

STRIDE = 2048
MARGIN = 192

# --- hatch signature (tensor gradient-direction band, deg mod 180) ---------
# full-sheet histogram: hatch gradient peak 155..170, mass to ~174; italic
# script 125..135; grid linework ~9.5 and ~99.5. v3's 140..165 clipped the
# peak and missed fine/steep hatch (Sunil recall issue #1).
HATCH_LO = 138.0
HATCH_HI = 174.0
COH_MIN = 0.45           # structure-tensor coherence minimum
ENERGY_MIN = 2500.0      # tensor energy minimum (paper noise ~180)
TENSOR_WIN = 9           # px, structure-tensor integration window

# --- grid-aligned dark divider lines (block splitting) ----------------------
GRID_GRAD = (9.5, 99.5)  # gradient directions of the street grid, deg mod 180
DIV_BAND = 8.0           # +- deg around each grid gradient direction
DIV_COH = 0.50
DIV_ENERGY = 4000.0
DIV_GRAY = 135           # divider strokes are heavy dark ink
DIV_LINE_LEN = 15        # px, oriented line-opening kernel length
# relaxed second-pass divider (pieces still over 2000 m2 only - these land
# in oversize_blocks for hand review anyway, so a slightly braver cut is
# worth it to shrink that layer):
DIVR_BAND = 10.0
DIVR_COH = 0.45
DIVR_ENERGY = 2500.0
DIVR_GRAY = 150
DIVR_LINE_LEN = 11
SPLIT_MIN_AREA = 350.0   # m2: only try to split components above this
SPLIT_PIECE_MIN = 55.0   # m2: min size for a split-off piece
DIV_CUT_DILATE = 3       # px, widen divider before cutting
SPLIT_WIDTH_MIN = 2.5    # m: a split producing a piece thinner than this
                         # cut a wall lengthwise (diagonal/curved building),
                         # not a parcel divider -> reject the whole split

# --- mask morphology (px at 0.402 m/px) -------------------------------------
CLOSE_INK = 3            # solidify hatched fills
OPEN_FINAL = 5           # remove street lines / text strokes / rigging
OPEN_SPLIT = 9           # split >2000 m2 chains at necks narrower than this
DILATE_RESTORE = 5       # grow split pieces back inside the parent mask

# --- per-component acceptance ------------------------------------------------
HATCH_FRAC_MIN = 0.25    # hatch px / FULL ink-component px
SOLIDITY_MIN = 0.70      # area / convex-hull area, applied below ...
SOLIDITY_AREA = 150.0    # ... this area (m2): kills stringy text/tree scraps
SOLIDITY_MID = 0.45      # looser solidity bar for mid-size shapes below ...
SOLIDITY_MID_AREA = 400.0  # ... this area: kills curved creek-bank shading
                           # arcs and script-lettering blobs (Dock street)
CLIP_KEEP_FRAC = 0.35    # drop polygon if clipping to core removes more
EMBED_AREA = 60.0        # m2: small components embedded in surrounding
EMBED_MAX = 0.30         # markings (tree-canopy speckle) are dropped
EMBED_GRAY = 190         # ring test counts any marking darker than this

# --- hatch-core trim (lettering / street-line appendages) --------------------
TRIM_CLOSE = 9           # px ellipse: close hatch strokes into a solid core
TRIM_MIN_FRAC = 0.06     # trim only if it removes at least this fraction
TRIM_MAX_FRAC = 0.45     # ... and at most this (solid-fill houses untouched)
TRIM_HATCH_MAX = 0.20    # removed region must itself be non-hatched (a real
                         # appendage: lettering / street line, not building)

# --- regularization ----------------------------------------------------------
SNAP_TOL = 10.0          # deg: snap own angle to grid theta if within this
RECT_FIT_MIN = 0.70      # area/MRR-area above which shape is "a rectangle"
RECT_AREA_MAX = 150.0    # m2: small shapes always become rectangles
HOLE_AREA_MIN = 150.0    # m2: preserve courtyard holes bigger than this

# --- post-regularization ink clamp -------------------------------------------
CLAMP_TOL = 1.2          # m: allowed overhang beyond the ink outline
CLAMP_TRIGGER = 0.07     # clamp if overhang exceeds this fraction of area
CLAMP_BUF = 0.5          # m: mitre buffer around ink outline when clamping

OVERSIZE_MAX = 50000.0   # m2
KEEP_AREA_MIN = 80.0     # m2 - smallest unconditionally kept footprint
TINY_MIN = 45.0          # m2: 45..80 m2 kept only with strong hatch...
TINY_HATCH_MIN = 0.40    # ... (recall issue #1: small hatched houses)


# ----------------------------------------------------------------------------
# texture masks
# ----------------------------------------------------------------------------
def tensor_fields(gray_f32):
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1)
    w = (TENSOR_WIN, TENSOR_WIN)
    jxx = cv2.boxFilter(gx * gx, -1, w)
    jyy = cv2.boxFilter(gy * gy, -1, w)
    jxy = cv2.boxFilter(gx * gy, -1, w)
    ori = 0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy)) % 180.0
    energy = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (energy + 1e-6)
    return ori, energy, coh


def hatch_mask_from(ori, energy, coh):
    return ((energy > ENERGY_MIN) & (coh > COH_MIN)
            & (ori >= HATCH_LO) & (ori <= HATCH_HI)).astype(np.uint8)


def _ang_dist(a, center):
    return np.abs((a - center + 90.0) % 180.0 - 90.0)


def _line_kernel(angle_deg, length):
    """Binary line kernel of given length/orientation (image coords)."""
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2.0
    dx = np.cos(np.radians(angle_deg))
    dy = np.sin(np.radians(angle_deg))
    p0 = (int(round(c - dx * c)), int(round(c - dy * c)))
    p1 = (int(round(c + dx * c)), int(round(c + dy * c)))
    cv2.line(k, p0, p1, 1, 1)
    return k

# gradient direction g -> the line itself runs at g+90
_DIV_KERNELS = [_line_kernel(g + 90.0, DIV_LINE_LEN) for g in GRID_GRAD]
_DIVR_KERNELS = [_line_kernel(g + 90.0, DIVR_LINE_LEN) for g in GRID_GRAD]


def _divider(gray, ori, energy, coh, band_w, coh_min, en_min, gray_max,
             kernels):
    band = np.zeros(ori.shape, bool)
    for g in GRID_GRAD:
        band |= _ang_dist(ori, g) <= band_w
    div = ((energy > en_min) & (coh > coh_min) & band
           & (gray < gray_max)).astype(np.uint8)
    opened = np.zeros_like(div)
    for k in kernels:
        opened |= cv2.morphologyEx(div, cv2.MORPH_OPEN, k)
    return opened


def divider_mask_from(gray, ori, energy, coh):
    """Grid-aligned heavy dark linework = candidate block dividers.
    Oriented line-opening kills isolated speckle on hatch strokes."""
    return _divider(gray, ori, energy, coh, DIV_BAND, DIV_COH, DIV_ENERGY,
                    DIV_GRAY, _DIV_KERNELS)


def divider_mask_relaxed(gray, ori, energy, coh):
    return _divider(gray, ori, energy, coh, DIVR_BAND, DIVR_COH, DIVR_ENERGY,
                    DIVR_GRAY, _DIVR_KERNELS)


# ----------------------------------------------------------------------------
# divider split: cut a component along internal dark lines
# ----------------------------------------------------------------------------
def divider_split(comp, div_roi, px2):
    """Cut comp (uint8) where dilated divider lines cross it. Returns a list
    of piece masks that exactly tile comp (divider px assigned to the
    nearest piece by label dilation), or [comp] if no usable split."""
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DIV_CUT_DILATE, DIV_CUT_DILATE))
    cut = comp & (cv2.dilate(div_roi, k) == 0)
    ns, sl, sstats, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
    # sliver guard: a cut piece thinner than SPLIT_WIDTH_MIN means the
    # "divider" was a building's own long wall (diagonal / curved row) cut
    # lengthwise - such pieces are not seeds; they get absorbed into the
    # nearest solid piece by the label dilation below. Only solid pieces
    # (area and width) count as split seeds.
    px_m = np.sqrt(px2)
    big = []
    for s in range(1, ns):
        if sstats[s, 4] * px2 < SPLIT_PIECE_MIN:
            continue
        ys, xs = np.where(sl == s)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, (rw, rh), _) = cv2.minAreaRect(pts)
        if min(rw, rh) * px_m < SPLIT_WIDTH_MIN:
            continue
        big.append(s)
    if len(big) < 2 or len(big) > 40:
        return [comp]
    lab = np.zeros(comp.shape, np.float32)
    for i, s in enumerate(big, start=1):
        lab[sl == s] = i
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for _ in range(64):
        unclaimed = (comp > 0) & (lab == 0)
        if not unclaimed.any():
            break
        grown = cv2.dilate(lab, k3)
        lab[unclaimed & (grown > 0)] = grown[unclaimed & (grown > 0)]
    return [((lab == i) & (comp > 0)).astype(np.uint8)
            for i in range(1, len(big) + 1)]


# ----------------------------------------------------------------------------
# hatch-core trim: drop non-hatched appendages (lettering, street lines)
# ----------------------------------------------------------------------------
def hatch_core_trim(p, roi_hatch):
    """If p carries a NON-HATCHED appendage of 6..45% of its pixels
    (attached street line, lettering, wharf tail), trim to the largest
    connected piece of its closed hatch core. The removed region must
    itself be non-hatched, so hatched building mass is never cut."""
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TRIM_CLOSE, TRIM_CLOSE))
    core = cv2.morphologyEx(
        cv2.dilate((roi_hatch > 0).astype(np.uint8), k3),
        cv2.MORPH_CLOSE, kc) & p
    p_px = int(p.sum())
    if p_px == 0 or not core.any():
        return p, False
    ns, sl, sstats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    if ns < 2:
        return p, False
    s = 1 + int(np.argmax(sstats[1:, 4]))
    largest = (sl == s).astype(np.uint8)
    removed_mask = (p > 0) & (largest == 0)
    n_removed = int(removed_mask.sum())
    removed = n_removed / p_px
    if not (TRIM_MIN_FRAC <= removed <= TRIM_MAX_FRAC):
        return p, False
    rem_hatch = float(roi_hatch[removed_mask].mean()) if n_removed else 0.0
    if rem_hatch > TRIM_HATCH_MAX:
        return p, False               # removed part is hatched -> building
    return largest, True


# ----------------------------------------------------------------------------
# tightened city core from the accumulated /8 hatch canvas (unchanged v3)
# ----------------------------------------------------------------------------
def build_core_v4(src, core_old, hatch_canvas):
    scale = 8
    oh, ow = hatch_canvas.shape
    tr_ov = src.transform * rasterio.Affine.scale(scale)

    dens = cv2.boxFilter(hatch_canvas, -1, (11, 11))
    urban = (dens > 0.08).astype(np.uint8)
    urban = cv2.morphologyEx(
        urban, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    core_mask = rfeatures.rasterize(
        [(core_old, 1)], out_shape=(oh, ow), transform=tr_ov, dtype=np.uint8)
    anchor = cv2.erode(core_mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (17, 17)))

    nlab, lab = cv2.connectedComponents(urban, connectivity=8)
    anchored_ids = np.unique(lab[(anchor > 0) & (lab > 0)])
    city = np.isin(lab, anchored_ids) & (lab > 0)

    core_rows = np.where(core_mask.any(axis=1))[0]
    r0, r1 = core_rows.min(), core_rows.max()
    oc_env = np.full(oh, -1, np.int32)
    for r in core_rows:
        oc_env[r] = np.where(core_mask[r])[0].max()

    env_raw = np.full(oh, np.nan, np.float64)
    for r in np.where(city.any(axis=1))[0]:
        env_raw[r] = np.where(city[r])[0].max()
    rr = np.arange(r0, r1 + 1)
    valid = ~np.isnan(env_raw[r0:r1 + 1])
    if valid.sum() < 10:
        return core_old
    interp = np.interp(rr, rr[valid], env_raw[r0:r1 + 1][valid])
    env_med = np.empty_like(interp)
    for i in range(len(interp)):
        lo, hi = max(0, i - 8), min(len(interp), i + 9)
        env_med[i] = np.median(interp[lo:hi])

    east_pts = []
    for i, r in enumerate(rr):
        c = min(env_med[i] + 4.0, float(oc_env[r]) + 0.5)
        x, y = tr_ov * (c + 1.0, r + 0.5)
        east_pts.append((x, y))
    x_far = src.bounds.left - 1000.0
    west_region = Polygon(
        [(x_far, east_pts[0][1])] + east_pts + [(x_far, east_pts[-1][1])])
    west_region = west_region.buffer(0)
    core = core_old.intersection(west_region)
    if core.geom_type == "MultiPolygon":
        core = max(core.geoms, key=lambda p: p.area)
    return core.simplify(2.0, preserve_topology=True)


# ----------------------------------------------------------------------------
# regularization (unchanged v3)
# ----------------------------------------------------------------------------
def _mrr_long_angle(geom):
    mrr = geom.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return 0.0, 1.0
    xs, ys = mrr.exterior.coords.xy
    e1 = (xs[1] - xs[0], ys[1] - ys[0])
    e2 = (xs[2] - xs[1], ys[2] - ys[1])
    l1 = np.hypot(*e1)
    l2 = np.hypot(*e2)
    ex, ey = (e1 if l1 >= l2 else e2)
    ang = np.degrees(np.arctan2(ey, ex)) % 90.0
    fit = geom.area / mrr.area if mrr.area > 0 else 0.0
    return ang, fit


def _snap_theta(own, grid):
    d = (own - grid + 45.0) % 90.0 - 45.0
    return grid if abs(d) <= SNAP_TOL else own


def _ortho_ring(rot, tol):
    ring = rot.simplify(tol, preserve_topology=True).exterior
    pts = list(ring.coords)[:-1]
    n = len(pts)
    if n < 4:
        return None
    cls = []
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cls.append('H' if abs(x1 - x0) >= abs(y1 - y0) else 'V')
    runs = []
    for i in range(n):
        if runs and runs[-1][0] == cls[i]:
            runs[-1][1].append(i)
        else:
            runs.append((cls[i], [i]))
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], runs[-1][1] + runs[0][1])
        runs.pop()
    if len(runs) < 4:
        return None
    coords = []
    for c, idxs in runs:
        tw, ts = 0.0, 0.0
        for i in idxs:
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            wgt = np.hypot(x1 - x0, y1 - y0)
            mid = (y0 + y1) / 2.0 if c == 'H' else (x0 + x1) / 2.0
            tw += wgt
            ts += wgt * mid
        coords.append((c, ts / tw if tw > 0 else 0.0))
    verts = []
    m = len(coords)
    for i in range(m):
        c0, v0 = coords[i]
        c1, v1 = coords[(i + 1) % m]
        if c0 == c1:
            return None
        verts.append((v0 if c0 == 'V' else v1, v0 if c0 == 'H' else v1))
    poly = Polygon(verts)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    if not (0.5 * rot.area < poly.area < 2.0 * rot.area):
        return None
    return poly


def regularize(geom, grid_theta):
    own, fit = _mrr_long_angle(geom)
    theta = _snap_theta(own, grid_theta)
    cen = geom.centroid
    rot = affinity.rotate(geom, -theta, origin=cen)

    if fit >= RECT_FIT_MIN or geom.area <= RECT_AREA_MAX:
        x0, y0, x1, y1 = rot.bounds
        rect = box(x0, y0, x1, y1)
        if rect.area > 0:
            s = np.sqrt(geom.area / rect.area)
            s = min(1.0, max(0.75, s))
            rect = affinity.scale(rect, xfact=s, yfact=s, origin='center')
        return affinity.rotate(rect, theta, origin=cen), 'rect'

    tol = 0.75 if geom.area < 2000.0 else 2.0
    ortho = _ortho_ring(rot, tol)
    if ortho is None:
        x0, y0, x1, y1 = rot.bounds
        return affinity.rotate(box(x0, y0, x1, y1), theta, origin=cen), 'bbox'
    for hole in rot.interiors:
        hp = Polygon(hole)
        if hp.area < HOLE_AREA_MIN:
            continue
        oh = _ortho_ring(hp, tol)
        if oh is not None:
            carved = ortho.difference(oh)
            if carved.geom_type == "Polygon" and not carved.is_empty:
                ortho = carved
    return affinity.rotate(ortho, theta, origin=cen), 'ortho'


def clamp_to_ink(reg, clipped, method):
    """Post-regularization ink clamp: if reg overhangs the source ink
    outline by more than CLAMP_TRIGGER of its area (beyond CLAMP_TOL),
    intersect it back with the ink outline (issue #3: parcels spilling
    onto streets / street-name lettering)."""
    try:
        over = reg.difference(clipped.buffer(CLAMP_TOL)).area
    except Exception:
        return reg, method
    if reg.area <= 0 or over / reg.area <= CLAMP_TRIGGER:
        return reg, method
    clamped = reg.intersection(
        clipped.buffer(CLAMP_BUF, join_style=2).simplify(0.3))
    if clamped.geom_type == "MultiPolygon":
        clamped = max(clamped.geoms, key=lambda p: p.area)
    clamped = clamped.simplify(0.8, preserve_topology=True)
    if (clamped.geom_type == "Polygon"
            and len(clamped.exterior.coords) > 40):
        clamped = clamped.simplify(1.5, preserve_topology=True)
    if (clamped.is_empty or clamped.geom_type != "Polygon"
            or not clamped.is_valid or clamped.area < bc.MIN_AREA):
        return reg, method
    return clamped, method + '+clamp'


def grid_theta_from(records):
    z = 0j
    for r in records:
        if not 30.0 < r['area'] < 1000.0:
            continue
        ang, fit = _mrr_long_angle(r['geom'])
        if fit <= 0.6:
            continue
        z += r['area'] * np.exp(1j * np.radians(ang * 4.0))
    if abs(z) < 1e-9:
        return 10.0
    return float(np.degrees(np.angle(z)) / 4.0) % 90.0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    src = rasterio.open(bc.COG)
    core_old = bc.load_core_poly(src)
    tiles = list(bc.iter_tiles(src, core_old, STRIDE, MARGIN))
    print(f"{len(tiles)} tiles (stride {STRIDE}px, margin {MARGIN}px)",
          flush=True)

    k_close_ink = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_INK, CLOSE_INK))
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_FINAL, OPEN_FINAL))

    canvas = np.zeros((src.height // 8 + 1, src.width // 8 + 1), np.float32)
    raw = []
    n_rej = {'hatch': 0, 'embedded': 0, 'toobig': 0}
    n_split_pieces = 0
    n_trimmed = 0
    for i, (tid, win, own) in enumerate(tiles):
        t1 = time.time()
        data = src.read(window=win)
        gray = bc.gray_from_tile(data)
        ink = bc.ink_mask(gray)
        frac = float(ink.mean())
        if frac < bc.MIN_INK_FRACTION:
            print(f"[{i+1}/{len(tiles)}] {tid} skip (ink {frac:.4f})",
                  flush=True)
            continue
        ori, energy, coh = tensor_fields(gray.astype(np.float32))
        hatch = hatch_mask_from(ori, energy, coh)
        div = divider_mask_from(gray, ori, energy, coh)
        divr = divider_mask_relaxed(gray, ori, energy, coh)
        del ori, energy, coh

        # accumulate /8 hatch canvas for the shoreline envelope
        r_off, c_off = int(win.row_off), int(win.col_off)
        small = cv2.resize(hatch.astype(np.float32),
                           (hatch.shape[1] // 8, hatch.shape[0] // 8),
                           interpolation=cv2.INTER_AREA)
        rr, cc = r_off // 8, c_off // 8
        h8, w8 = small.shape
        h8 = min(h8, canvas.shape[0] - rr)
        w8 = min(w8, canvas.shape[1] - cc)
        np.maximum(canvas[rr:rr + h8, cc:cc + w8], small[:h8, :w8],
                   out=canvas[rr:rr + h8, cc:cc + w8])

        m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close_ink)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

        nlab, lab, cstats, _ = cv2.connectedComponentsWithStats(
            m, connectivity=4)
        flat = lab.ravel()
        area_px = np.bincount(flat, minlength=nlab)

        transform = src.window_transform(win)
        k_split = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (OPEN_SPLIT, OPEN_SPLIT))
        k_restore = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_RESTORE, DILATE_RESTORE))
        px2 = abs(src.transform.a * src.transform.e)
        kept = 0
        for v in range(1, nlab):
            if area_px[v] * px2 < bc.MIN_AREA:
                continue
            x, y, w, h = cstats[v, :4]
            pad = 4
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(m.shape[1], x + w + pad)
            y1 = min(m.shape[0], y + h + pad)
            comp = (lab[y0:y1, x0:x1] == v).astype(np.uint8)
            roi_t = transform * rasterio.Affine.translation(x0, y0)
            roi_hatch = hatch[y0:y1, x0:x1]
            roi_ink = ink[y0:y1, x0:x1]
            roi_gray = gray[y0:y1, x0:x1]
            roi_div = div[y0:y1, x0:x1]
            roi_divr = divr[y0:y1, x0:x1]

            # 1) cut merged blocks along internal dark divider lines
            if area_px[v] * px2 > SPLIT_MIN_AREA:
                pieces0 = divider_split(comp, roi_div, px2)
            else:
                pieces0 = [comp]
            was_split = len(pieces0) > 1
            if was_split:
                n_split_pieces += len(pieces0)

            # 2) pieces still over 2000 m2: braver relaxed divider cut
            pieces1 = []
            for p in pieces0:
                if int(p.sum()) * px2 > bc.MAX_AREA:
                    subs = divider_split(p, roi_divr, px2)
                    if len(subs) > 1:
                        n_split_pieces += len(subs)
                        was_split = True
                    pieces1.extend(subs)
                else:
                    pieces1.append(p)

            # 3) remaining giant chains -> v3 neck split
            pieces = []
            for p in pieces1:
                if int(p.sum()) * px2 > bc.MAX_AREA:
                    opened = cv2.morphologyEx(p, cv2.MORPH_OPEN, k_split)
                    ns, sl = cv2.connectedComponents(opened, connectivity=4)
                    subs = []
                    for s in range(1, ns):
                        q = ((sl == s).astype(np.uint8))
                        q = cv2.dilate(q, k_restore) & p
                        subs.append(q)
                    pieces.extend(subs if subs else [p])
                else:
                    pieces.append(p)

            for p in pieces:
                p_px = int(p.sum())
                a_est = p_px * px2
                if a_est < bc.MIN_AREA:
                    continue
                h_px = int(roi_hatch[p > 0].sum())
                hfrac = h_px / max(1, p_px)
                if hfrac < HATCH_FRAC_MIN:    # full-component hatch test
                    n_rej['hatch'] += 1
                    continue
                if a_est < EMBED_AREA:
                    ring = ((cv2.dilate(p, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (9, 9))) > 0)
                        & (cv2.dilate(p, cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (3, 3))) == 0))
                    if ring.any() and float(
                            (roi_gray[ring] < EMBED_GRAY).mean()) > EMBED_MAX:
                        n_rej['embedded'] += 1  # canopy/letter fragment
                        continue
                # 4) trim non-hatched appendages (lettering, street lines)
                p, trimmed = hatch_core_trim(p, roi_hatch)
                if trimmed:
                    n_trimmed += 1
                    # smooth the trim scar so regularization sees a clean
                    # outline instead of a ragged hatch-core edge
                    k3s = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, k3s)
                    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3s)
                    nsm, slm, ssm, _ = cv2.connectedComponentsWithStats(
                        p, connectivity=4)
                    if nsm > 2:
                        p = (slm == 1 + int(np.argmax(ssm[1:, 4]))).astype(
                            np.uint8)
                    p_px = int(p.sum())
                    a_est = p_px * px2
                    if a_est < bc.MIN_AREA:
                        continue
                for g, val in rfeatures.shapes(
                        p, mask=p.astype(bool), transform=roi_t):
                    geom = shape(g)
                    a = geom.area
                    if a < bc.MIN_AREA:
                        continue
                    if a > OVERSIZE_MAX:
                        n_rej['toobig'] += 1
                        continue
                    dup = False
                    if not own.contains(geom.representative_point()):
                        truncated = (x <= 0 or y <= 0 or x + w >= m.shape[1]
                                     or y + h >= m.shape[0])
                        if truncated and geom.intersects(own):
                            dup = True
                        else:
                            continue
                    raw.append({
                        'tile': tid, 'geom': geom, 'area': a,
                        'hatch_frac': round(hfrac, 3),
                        'ink_frac': round(
                            float(roi_ink[p > 0].mean()), 3),
                        'dup': dup,
                        'split': was_split,
                        'trimmed': trimmed,
                    })
                    kept += 1
        print(f"[{i+1}/{len(tiles)}] {tid} ink={frac:.3f} kept={kept} "
              f"({time.time()-t1:.1f}s)", flush=True)

    print("building tightened core from hatch canvas...", flush=True)
    core = build_core_v4(src, core_old, canvas)
    print(f"core_v4: area {core.area/1e6:.2f} km2 "
          f"(old {core_old.area/1e6:.2f} km2)", flush=True)
    theta = grid_theta_from(raw)
    print(f"grid theta: {theta:.2f} deg (mod 90)", flush=True)
    print(f"divider-split pieces: {n_split_pieces}, "
          f"hatch-core trims: {n_trimmed}", flush=True)

    # resolve duplicates (unchanged v3)
    n_rej['dup'] = 0
    geoms_all = [r['geom'] for r in raw]
    tree = STRtree(geoms_all)
    resolved = []
    for idx, r in enumerate(raw):
        if r['dup'] or r['area'] > 1000.0:
            shadowed = False
            for j in tree.query(r['geom']):
                j = int(j)
                if j == idx:
                    continue
                o = raw[j]
                inter = r['geom'].intersection(o['geom']).area
                if inter / min(r['area'], o['area']) > 0.6 and (
                        o['area'] > r['area']
                        or (o['area'] == r['area'] and j < idx)):
                    shadowed = True
                    break
            if shadowed:
                n_rej['dup'] += 1
                continue
        resolved.append(r)
    raw = resolved

    # pass 2: clip to the wharf-line core, shape filters, regularize, clamp
    n_rej.update({'river': 0, 'sliver': 0, 'solidity': 0, 'tiny': 0})
    n_clamped = 0
    cands, blocks = [], []
    for r in raw:
        geom = r['geom']
        clipped = geom.intersection(core)
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda p: p.area)
        if (clipped.is_empty or clipped.geom_type != "Polygon"
                or clipped.area < bc.MIN_AREA
                or clipped.area / r['area'] < CLIP_KEEP_FRAC):
            n_rej['river'] += 1               # ship / river blob
            continue
        a = clipped.area
        if a < KEEP_AREA_MIN:
            # v3 dropped everything under 80 m2; v4 keeps 45..80 m2 when
            # hatch evidence is strong (small hatched houses, recall #1)
            if not (a >= TINY_MIN and r['hatch_frac'] >= TINY_HATCH_MIN):
                n_rej['tiny'] += 1
                continue
        oversize = a > bc.MAX_AREA
        if not oversize and not bc.passes_filters(clipped):
            n_rej['sliver'] += 1
            continue
        solidity = a / max(clipped.convex_hull.area, 1e-9)
        if a < SOLIDITY_AREA and solidity < SOLIDITY_MIN:
            n_rej['solidity'] += 1
            continue
        if a < SOLIDITY_MID_AREA and solidity < SOLIDITY_MID:
            n_rej['solidity'] += 1      # curved shading arc / script blob
            continue
        if not oversize and solidity < 0.50:
            mrr_fit = a / max(clipped.minimum_rotated_rectangle.area, 1e-9)
            if mrr_fit < 0.35:
                n_rej['solidity'] += 1  # crescent along curved street /
                continue                # script-word blob (Dock street)
        reg, method = regularize(clipped, theta)
        reg, method = clamp_to_ink(reg, clipped, method)
        if method.endswith('+clamp'):
            n_clamped += 1
        rec = {
            'tile': r['tile'],
            'area_m2_3857': round(reg.area, 1),
            'src_area_m2': round(a, 1),
            'hatch_frac': r['hatch_frac'],
            'ink_frac': r['ink_frac'],
            'reg_method': method,
            'div_split': bool(r['split']),
            'ink_trimmed': bool(r['trimmed']),
            'geometry': reg,
        }
        (blocks if oversize else cands).append(rec)

    # safety net vs v3 (recall must never regress): any v3 candidate whose
    # footprint is <30% covered by v4 output is carried over verbatim,
    # flagged reg_method='v3carry'. These are a handful of borderline
    # curved-row / low-hatch cases where v4's stricter piece handling
    # dropped what v3 kept.
    v3_path = os.path.join(os.path.dirname(OUT), "candidates_v3.gpkg")
    n_carry = 0
    if os.path.exists(v3_path):
        prev = gpd.read_file(v3_path, layer="candidates")
        out_geoms = ([r['geometry'] for r in cands]
                     + [r['geometry'] for r in blocks])
        otree = STRtree(out_geoms)
        for _, row in prev.iterrows():
            g = row.geometry
            cov = sum(out_geoms[int(j)].intersection(g).area
                      for j in otree.query(g))
            if cov / g.area >= 0.30:
                continue
            cands.append({
                'tile': row['tile'],
                'area_m2_3857': row['area_m2_3857'],
                'src_area_m2': row['src_area_m2'],
                'hatch_frac': row['hatch_frac'],
                'ink_frac': row['ink_frac'],
                'reg_method': 'v3carry',
                'div_split': False,
                'ink_trimmed': False,
                'geometry': g,
            })
            n_carry += 1
    print(f"v3 carry-over candidates: {n_carry}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    gpd.GeoDataFrame(cands, crs=src.crs).to_file(
        OUT, driver="GPKG", layer="candidates")
    if blocks:
        gpd.GeoDataFrame(blocks, crs=src.crs).to_file(
            OUT, driver="GPKG", layer="oversize_blocks")
    gpd.GeoDataFrame({'geometry': [core]}, crs=src.crs).to_file(
        OUT, driver="GPKG", layer="core_v4")
    print(f"wrote {OUT}: {len(cands)} candidates + {len(blocks)} oversize "
          f"blocks; clamped {n_clamped}; rejected {n_rej}; "
          f"total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
