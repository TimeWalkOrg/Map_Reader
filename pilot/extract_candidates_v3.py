#!/usr/bin/env python3
"""Candidate building-footprint extraction v3 for the 1762 Clarkson & Biddle
COG. Addresses tester (Sunil) feedback on candidates_threshold.gpkg (v1):

1. STRAIGHT-LINED POLYGONS - every footprint is regularized: quasi-
   rectangular/small shapes become oriented rectangles; larger shapes become
   rectilinear (orthogonal) polygons. Orientations snap to the dominant
   street-grid angle (computed globally; the grid sits ~9.5 deg off the
   EPSG:3857 axes). Courtyard holes of big blocks are preserved.
2. NO WORDS / TREES - candidates must carry the map's building signature:
   diagonal hatch fill. Hatch is detected per-pixel with a structure tensor;
   the accepted orientation band (gradient direction 140..165 deg mod 180,
   measured at building rows across the whole sheet) excludes italic script
   strokes (125..135) and tree/garden speckle (~5..10 / ~95).
3. HATCH AS THE PRIMARY POSITIVE TEST - the test unit is the FULL ink
   component (whole letter, whole tree canopy, whole building): a component
   is kept only if >=25% of its pixels carry hatch texture. Whole-object
   statistics are what separate a building (0.4..0.9 in-band) from a tree
   canopy or letter cluster (~0..0.15 in-band) - fragment-level tests could
   not (small diagonal serif/canopy arcs locally mimic hatch).
4. NO SHIPS / RIVER BLOBS - the city-core boundary is tightened along the
   Delaware using a shoreline envelope computed from the hatch mask
   (river lettering has no hatch and cannot inflate it; hatched ships are
   not anchored to the city fabric and cannot extend it). Every polygon is
   CLIPPED to the tightened core (the wharf line); polygons that lose >65%
   of their area or fall below the minimum area are dropped. A ship joined
   to a wharf by rigging loses its river half at the clip and survives only
   as the genuine wharf structure.

Output: pilot/results/candidates_v3.gpkg (EPSG:3857)
  layer candidates      - regularized building footprints (10..2000 m2)
  layer oversize_blocks - hatched row-blocks/blocks 2000..50000 m2, regularized
  layer core_v3         - the tightened city-core polygon used for clipping

Run:  env/bin/python pilot/extract_candidates_v3.py   (~10 s)
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
                   "results", "candidates_v3.gpkg")

STRIDE = 2048
MARGIN = 192

# --- hatch signature (tensor gradient-direction band, deg mod 180) ---------
# measured on building rows across the sheet: 140..165; italic script strokes
# sit at 125..135, trees/garden speckle at ~5..10 and ~95.
HATCH_LO = 140.0
HATCH_HI = 165.0
COH_MIN = 0.45           # structure-tensor coherence minimum
ENERGY_MIN = 4000.0      # tensor energy minimum (paper noise ~180, ink ~20k+)
TENSOR_WIN = 9           # px, structure-tensor integration window

# --- mask morphology (px at 0.402 m/px) -------------------------------------
CLOSE_INK = 3            # solidify hatched fills
OPEN_FINAL = 5           # remove street lines / text strokes / rigging
OPEN_SPLIT = 9           # split >2000 m2 chains at necks narrower than this
DILATE_RESTORE = 5       # grow split pieces back inside the parent mask

# --- per-component acceptance ------------------------------------------------
HATCH_FRAC_MIN = 0.25    # hatch px / FULL ink-component px
SOLIDITY_MIN = 0.70      # area / convex-hull area, applied below ...
SOLIDITY_AREA = 150.0    # ... this area (m2): kills stringy text/tree scraps
CLIP_KEEP_FRAC = 0.35    # drop polygon if clipping to core removes more
EMBED_AREA = 60.0        # m2: small components embedded in surrounding
EMBED_MAX = 0.30         # markings (tree-canopy speckle) are dropped
EMBED_GRAY = 190         # ring test counts any marking darker than this

# --- regularization ----------------------------------------------------------
SNAP_TOL = 10.0          # deg: snap own angle to grid theta if within this
RECT_FIT_MIN = 0.70      # area/MRR-area above which shape is "a rectangle"
RECT_AREA_MAX = 150.0    # m2: small shapes always become rectangles
HOLE_AREA_MIN = 150.0    # m2: preserve courtyard holes bigger than this

OVERSIZE_MAX = 50000.0   # m2
KEEP_AREA_MIN = 80.0     # m2 — smallest credible building footprint here


# ----------------------------------------------------------------------------
# hatch texture
# ----------------------------------------------------------------------------
def hatch_mask(gray_f32):
    """Per-pixel hatch detector: structure tensor orientation/coherence."""
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1)
    w = (TENSOR_WIN, TENSOR_WIN)
    jxx = cv2.boxFilter(gx * gx, -1, w)
    jyy = cv2.boxFilter(gy * gy, -1, w)
    jxy = cv2.boxFilter(gx * gy, -1, w)
    ori = 0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy)) % 180.0
    energy = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (energy + 1e-6)
    return ((energy > ENERGY_MIN) & (coh > COH_MIN)
            & (ori >= HATCH_LO) & (ori <= HATCH_HI)).astype(np.uint8)


# ----------------------------------------------------------------------------
# tightened city core from the accumulated /8 hatch canvas
# ----------------------------------------------------------------------------
def build_core_v3(src, core_old, hatch_canvas):
    """East (Delaware) edge = envelope of hatch density anchored to the city
    fabric. Hatched ships are isolated in the river and never anchor; river
    lettering has no hatch at all. The edge is clamped to never exceed the
    hand-digitized v1 core, +3 px margin so wharf-edge buildings stay whole."""
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
# regularization
# ----------------------------------------------------------------------------
def _mrr_long_angle(geom):
    """Orientation (deg, mod 90) of the long edge of the min rotated rect."""
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
    """Rectilinear fit of an axis-aligned-ish polygon (already rotated into
    the grid frame). Returns a hole-free Polygon or None."""
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
    runs = []  # (class, [edge indices]), with wraparound merge
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
    """Return (regularized polygon, method): oriented rectangles for small or
    compact shapes, rectilinear polygons (with preserved courtyard holes) for
    larger complexes, all snapped to the dominant grid orientation when
    within SNAP_TOL of it."""
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


def grid_theta_from(records):
    """Area-weighted circular mean (mod 90) of MRR angles of clean shapes."""
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
        hatch = hatch_mask(gray.astype(np.float32))

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
        hatch_px = np.bincount(flat[hatch.ravel() > 0], minlength=nlab)
        ink_px = np.bincount(flat[ink.ravel() > 0], minlength=nlab)

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

            # big chains (street-line / crease bridges) -> split at necks
            if area_px[v] * px2 > bc.MAX_AREA:
                opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, k_split)
                ns, sl = cv2.connectedComponents(opened, connectivity=4)
                pieces = []
                for s in range(1, ns):
                    p = ((sl == s).astype(np.uint8))
                    p = cv2.dilate(p, k_restore) & comp
                    pieces.append(p)
                if not pieces:
                    pieces = [comp]
            else:
                pieces = [comp]

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
                        # window-truncated pieces can land outside every
                        # tile's ownership cell; emit flagged, dedupe later
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
                    })
                    kept += 1
        print(f"[{i+1}/{len(tiles)}] {tid} ink={frac:.3f} kept={kept} "
              f"({time.time()-t1:.1f}s)", flush=True)

    print("building tightened core from hatch canvas...", flush=True)
    core = build_core_v3(src, core_old, canvas)
    print(f"core_v3: area {core.area/1e6:.2f} km2 "
          f"(old {core_old.area/1e6:.2f} km2)", flush=True)
    theta = grid_theta_from(raw)
    print(f"grid theta: {theta:.2f} deg (mod 90)", flush=True)

    # resolve duplicates: window-truncated variants of the same ink can be
    # emitted by several tiles (flagged, or big pieces whose truncated
    # centroid still lands in the emitting tile's cell). Keep the largest.
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

    # pass 2: clip to the wharf-line core, shape filters, regularize
    n_rej.update({'river': 0, 'sliver': 0, 'solidity': 0, 'tiny': 0})
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
            # QC contact sheets: every kept component under ~80 m2 was a
            # letter fragment, tree canopy bit, or street-line sliver;
            # real detached buildings on this sheet start around 80 m2.
            n_rej['tiny'] += 1
            continue
        oversize = a > bc.MAX_AREA
        if not oversize and not bc.passes_filters(clipped):
            n_rej['sliver'] += 1
            continue
        if (a < SOLIDITY_AREA
                and a / max(clipped.convex_hull.area, 1e-9) < SOLIDITY_MIN):
            n_rej['solidity'] += 1
            continue
        reg, method = regularize(clipped, theta)
        rec = {
            'tile': r['tile'],
            'area_m2_3857': round(reg.area, 1),
            'src_area_m2': round(a, 1),
            'hatch_frac': r['hatch_frac'],
            'ink_frac': r['ink_frac'],
            'reg_method': method,
            'geometry': reg,
        }
        (blocks if oversize else cands).append(rec)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    gpd.GeoDataFrame(cands, crs=src.crs).to_file(
        OUT, driver="GPKG", layer="candidates")
    if blocks:
        gpd.GeoDataFrame(blocks, crs=src.crs).to_file(
            OUT, driver="GPKG", layer="oversize_blocks")
    gpd.GeoDataFrame({'geometry': [core]}, crs=src.crs).to_file(
        OUT, driver="GPKG", layer="core_v3")
    print(f"wrote {OUT}: {len(cands)} candidates + {len(blocks)} oversize "
          f"blocks; rejected {n_rej}; total {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
