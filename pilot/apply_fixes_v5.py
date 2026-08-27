#!/usr/bin/env python3
"""Verify + apply vision-audit findings to the v4 candidate vectors (v5).

Findings from the vision judge (audit_vision.py pairs -> image-model JSONL)
are PROPOSALS. Every finding is re-verified against raster evidence from the
COG before anything changes; the decision policy is:

  false_positive : polygon under the bbox has (relaxed) hatch fraction
                   < FP_DELETE and is not a solid dark fill -> DELETE.
                   FP_DELETE..FP_REVIEW -> needs_review. Else dismissed.
  missing_feature: local re-extract inside the bbox with relaxed hatch
                   thresholds; a component with real hatch (or solid-fill)
                   evidence not already covered -> ADD (regularized+clamped,
                   reg_method='v5vision'). Weak evidence -> needs_review.
  shape_mismatch : measured against local ink: overhang beyond ink >20% of
                   area -> re-clamp to ink; missing >25% of its hatch core
                   -> extend/re-derive. Verified improvement only, else
                   needs_review / dismissed.
  missing_split  : relaxed dark-divider split of the polygon mask; >=2 solid
                   pieces -> SPLIT. Else needs_review.

Ambiguity NEVER auto-resolves: it lands in the `needs_review` layer with a
note. Findings contradicted by raster evidence are logged as dismissed.

Usage:
  env/bin/python pilot/apply_fixes_v5.py --pass 1 \
      --in pilot/results/candidates_v4.gpkg \
      --findings 'pilot/results/audit_v5/findings_pass1_*.jsonl' \
      --out pilot/results/candidates_v5.gpkg

Later passes use --in candidates_v5.gpkg (in-place iteration) and append to
the same changelog. Emits audit_v5/changed_tiles_passN.txt for re-audit.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape, box as sbox, Polygon
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc
import extract_candidates_v4 as v4

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "results", "audit_v5")

# evidence thresholds (fractions of polygon/component pixels)
FP_DELETE = 0.06      # relaxed-hatch frac below this -> junk, delete
FP_REVIEW = 0.15      # ..below this -> needs_review
SOLID_DARK = 0.50     # dark-fill fraction marking solid-fill buildings
MF_ACCEPT = 0.22      # hatch frac to auto-add a missed building
MF_REVIEW = 0.10      # ..to flag for review
MF_MIN_AREA = 15.0    # m2
MF_MAX_COVER = 0.25   # max fraction already covered by existing vectors
SM_OVERHANG = 0.20    # overhang frac beyond ink triggering re-clamp
SM_MISSED = 0.25      # missed hatch-core frac triggering extend
SPLIT_PIECE = 35.0    # m2 minimum piece from a vision-directed split
BBOX_MIN_PX = 8       # ignore degenerate bboxes

# relaxed hatch band (v4 band is 138..174 / energy 2500 / coh .45)
RH_LO, RH_HI, RH_EN, RH_COH = 132.0, 178.0, 1500.0, 0.40


def region_from(src, bounds, pad_m=10.0):
    """Read a COG window around map-coord bounds; return dict of arrays."""
    x0, y0, x1, y1 = bounds
    x0 -= pad_m; y0 -= pad_m; x1 += pad_m; y1 += pad_m
    inv = ~src.transform
    c0, r0 = inv * (x0, y1)
    c1, r1 = inv * (x1, y0)
    c0 = max(0, int(np.floor(c0))); r0 = max(0, int(np.floor(r0)))
    c1 = min(src.width, int(np.ceil(c1))); r1 = min(src.height, int(np.ceil(r1)))
    if c1 - c0 < 4 or r1 - r0 < 4:
        return None
    win = Window(c0, r0, c1 - c0, r1 - r0)
    gray = bc.gray_from_tile(src.read(window=win))
    tr = src.window_transform(win)
    ori, energy, coh = v4.tensor_fields(gray.astype(np.float32))
    hatch = v4.hatch_mask_from(ori, energy, coh)
    rhatch = ((energy > RH_EN) & (coh > RH_COH)
              & (ori >= RH_LO) & (ori <= RH_HI)).astype(np.uint8)
    divr = v4.divider_mask_relaxed(gray, ori, energy, coh)
    ink = bc.ink_mask(gray)
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kc),
                            cv2.MORPH_OPEN, ko)
    return {"gray": gray, "ink": ink, "mask": mask, "hatch": hatch,
            "rhatch": rhatch, "divr": divr, "tr": tr,
            "px2": abs(src.transform.a * src.transform.e)}


def rast(geom, reg):
    """Rasterize a geometry onto the region grid -> uint8 mask."""
    return rfeatures.rasterize([(geom, 1)], out_shape=reg["gray"].shape,
                               transform=reg["tr"], dtype=np.uint8)


def poly_fracs(geom, reg):
    m = rast(geom, reg) > 0
    n = int(m.sum())
    if n == 0:
        return 0.0, 0.0, 0.0
    return (float(reg["rhatch"][m].mean()),
            float((reg["gray"][m] < 120).mean()),
            float(reg["ink"][m].mean()))


def ink_rect_fit(geom, reg):
    """Rectangularity of the ink under a polygon: largest ink component's
    area / its min-area-rect area. Solid-fill BUILDINGS are compact
    rectangles (fit >= ~0.7); script-letter knots / ornaments are curvy
    (fit < 0.7). Distinguishes the two dark-fill cases."""
    m = (rast(geom, reg) > 0) & (reg["mask"] > 0)
    if not m.any():
        return 0.0
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=4)
    if n < 2:
        return 0.0
    s = 1 + int(np.argmax(st[1:, 4]))
    ys, xs = np.where(lab == s)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, (rw, rh_), _) = cv2.minAreaRect(pts)
    if rw * rh_ <= 0:
        return 0.0
    return float(st[s, 4]) / float(rw * rh_)


def ink_polys(reg, min_area=5.0):
    out = []
    for g, val in rfeatures.shapes(reg["mask"], mask=reg["mask"].astype(bool),
                                   transform=reg["tr"]):
        p = shape(g)
        if p.area >= min_area:
            out.append(p)
    return out


def largest_poly(g):
    if g.geom_type == "MultiPolygon":
        return max(g.geoms, key=lambda p: p.area)
    return g if g.geom_type == "Polygon" else None


class Working:
    """Mutable candidate set with stable ids + an audit trail."""

    def __init__(self, gpkg):
        import pyogrio
        have = {l[0] for l in pyogrio.list_layers(gpkg)}
        rows = []
        for layer in ("candidates", "oversize_blocks"):
            if layer not in have:
                continue
            g = gpd.read_file(gpkg, layer=layer)
            for _, r in g.iterrows():
                d = {k: r[k] for k in g.columns if k != "geometry"}
                d["geometry"] = r.geometry
                d.setdefault("v5_action", "kept")
                d.setdefault("v5_note", "")
                rows.append(d)
        self.rows = rows
        self.alive = [True] * len(rows)
        self.touched = set()
        if "core_v4" in have:
            self.core = gpd.read_file(gpkg, layer="core_v4").geometry.iloc[0]
        else:
            self.core = None
        self.review = []
        if "needs_review" in have:
            g = gpd.read_file(gpkg, layer="needs_review")
            for _, r in g.iterrows():
                self.review.append({k: r[k] for k in g.columns})

    def tree(self):
        idx = [i for i, a in enumerate(self.alive) if a]
        return STRtree([self.rows[i]["geometry"] for i in idx]), idx

    def find_under(self, bbox_geom, min_inter_frac=0.4):
        """Rows substantially inside the bbox (inter/poly >= frac)."""
        tree, idx = self.tree()
        out = []
        for j in tree.query(bbox_geom):
            i = idx[int(j)]
            g = self.rows[i]["geometry"]
            inter = g.intersection(bbox_geom).area
            if g.area > 0 and inter / g.area >= min_inter_frac:
                out.append(i)
        return out

    def best_overlap(self, bbox_geom):
        tree, idx = self.tree()
        best, best_i = 0.0, None
        for j in tree.query(bbox_geom):
            i = idx[int(j)]
            g = self.rows[i]["geometry"]
            inter = g.intersection(bbox_geom).area
            sc = inter / min(g.area, bbox_geom.area) if g.area > 0 else 0
        # noqa
            if sc > best:
                best, best_i = sc, i
        return best_i, best

    def coverage(self, geom):
        tree, idx = self.tree()
        cov = 0.0
        for j in tree.query(geom):
            i = idx[int(j)]
            cov += self.rows[i]["geometry"].intersection(geom).area
        return cov / geom.area if geom.area > 0 else 0.0

    def add(self, geom, tile, hfrac, ifrac, method, note):
        self.rows.append({
            "tile": tile, "area_m2_3857": round(geom.area, 1),
            "src_area_m2": round(geom.area, 1),
            "hatch_frac": round(hfrac, 3), "ink_frac": round(ifrac, 3),
            "reg_method": method, "div_split": False, "ink_trimmed": False,
            "v5_action": "added", "v5_note": note, "geometry": geom})
        self.alive.append(True)
        self.touched.add(len(self.rows) - 1)
        return len(self.rows) - 1

    def flag(self, geom, category, note, pass_no, confidence):
        self.review.append({"category": category, "note": note,
                            "pass": pass_no, "confidence": confidence,
                            "geometry": geom})


def load_findings(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            for f in rec.get("findings", []):
                bb = f.get("bbox_px")
                if (not bb or len(bb) != 4
                        or f.get("category") not in (
                            "missing_feature", "false_positive",
                            "shape_mismatch", "missing_split")):
                    continue
                x0, y0, x1, y1 = [float(v) for v in bb]
                if x1 - x0 < BBOX_MIN_PX or y1 - y0 < BBOX_MIN_PX:
                    continue
                out.append({"tile": rec["tile"], "category": f["category"],
                            "bbox_px": [x0, y0, x1, y1],
                            "confidence": f.get("confidence", "low"),
                            "note": f.get("note", "")})
    return out


def to_map_bbox(f, tiles):
    t = tiles[f["tile"]]
    a, b, c, d, e, ff = t["transform"]
    x0, y0, x1, y1 = f["bbox_px"]
    X0 = c + a * x0; X1 = c + a * x1
    Y0 = ff + e * y0; Y1 = ff + e * y1
    return sbox(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1))


def dedupe(findings):
    """Merge same-category findings whose map bboxes overlap (IoU>0.35)."""
    kept = []
    for f in sorted(findings, key=lambda f: -f["geom"].area):
        merged = False
        for k in kept:
            if k["category"] != f["category"]:
                continue
            inter = k["geom"].intersection(f["geom"]).area
            union = k["geom"].union(f["geom"]).area
            if union > 0 and inter / union > 0.35:
                merged = True
                break
        if not merged:
            kept.append(f)
    return kept


CONF_ORD = {"high": 2, "medium": 1, "low": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--findings", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = rasterio.open(bc.COG)
    manifest = json.load(open(os.path.join(AUDIT, "tiles_manifest.json")))
    tiles = {t["id"]: t for t in manifest["tiles"]}

    work = Working(args.inp)
    theta = v4.grid_theta_from(
        [{"area": r["geometry"].area, "geom": r["geometry"]}
         for r in work.rows])
    print(f"grid theta {theta:.2f}; loaded {len(work.rows)} rows")

    raw_f = load_findings(args.findings)
    for f in raw_f:
        if f["tile"] not in tiles:
            f["geom"] = None
            continue
        f["geom"] = to_map_bbox(f, tiles)
    raw_f = [f for f in raw_f if f["geom"] is not None]
    findings = dedupe(raw_f)
    print(f"{len(raw_f)} findings -> {len(findings)} after dedupe")

    log = []
    changed_tiles = set()

    def touch(geom):
        for tid, t in tiles.items():
            if sbox(*t["bounds"]).intersects(geom):
                changed_tiles.add(tid)

    order = {"false_positive": 0, "missing_split": 1, "shape_mismatch": 2,
             "missing_feature": 3}
    findings.sort(key=lambda f: order[f["category"]])

    for f in findings:
        cat, bbox = f["category"], f["geom"]
        entry = {"tile": f["tile"], "category": cat,
                 "confidence": f["confidence"], "note": f["note"],
                 "bbox": list(bbox.bounds), "pass": args.pass_no}
        reg = region_from(src, bbox.bounds)
        if reg is None:
            entry["disposition"] = "dismissed"; entry["reason"] = "degenerate"
            log.append(entry); continue

        if cat == "false_positive":
            hits = work.find_under(bbox, 0.4)
            if not hits:
                entry["disposition"] = "dismissed"
                entry["reason"] = "no polygon mostly inside bbox"
                log.append(entry); continue
            dispositions = []
            for i in hits:
                g = work.rows[i]["geometry"]
                rh, dark, ink = poly_fracs(g, reg)
                if dark >= SOLID_DARK and ink_rect_fit(g, reg) >= 0.70:
                    dispositions.append(("dismissed",
                                         f"solid rect fill ({dark:.2f})"))
                elif dark >= SOLID_DARK:
                    # dark but NOT rectangular: script-letter knot/ornament
                    if args.pass_no > 1 and i in work.touched:
                        work.flag(g, cat, f["note"], args.pass_no,
                                  f["confidence"])
                        dispositions.append(("needs_review", "touched"))
                    else:
                        work.alive[i] = False
                        work.rows[i]["v5_action"] = "deleted"
                        work.rows[i]["v5_note"] = (f["note"]
                                                   + " [dark non-rect]")
                        touch(g)
                        dispositions.append(
                            ("fixed_delete", f"dark {dark:.2f} non-rect"))
                elif rh < FP_DELETE:
                    if args.pass_no > 1 and i in work.touched:
                        work.flag(g, cat, "re-flagged after earlier fix: "
                                  + f["note"], args.pass_no, f["confidence"])
                        work.alive[i] = False
                        touch(g)
                        dispositions.append(("needs_review",
                                             f"touched earlier; rh={rh:.2f}"))
                    else:
                        work.alive[i] = False
                        work.rows[i]["v5_action"] = "deleted"
                        work.rows[i]["v5_note"] = f["note"]
                        touch(g)
                        dispositions.append(("fixed_delete", f"rh={rh:.2f}"))
                elif rh < FP_REVIEW:
                    work.flag(g, cat, f["note"] + f" (rh={rh:.2f})",
                              args.pass_no, f["confidence"])
                    dispositions.append(("needs_review", f"rh={rh:.2f}"))
                else:
                    dispositions.append(("dismissed",
                                         f"real hatch rh={rh:.2f}"))
            best = max(dispositions,
                       key=lambda d: {"fixed_delete": 2, "needs_review": 1,
                                      "dismissed": 0}[d[0]])
            entry["disposition"], entry["reason"] = best
            entry["n_polys"] = len(hits)

        elif cat == "missing_feature":
            pad_bbox = bbox.buffer(4.0)
            nlab, lab, cst, _ = cv2.connectedComponentsWithStats(
                reg["mask"], connectivity=4)
            bbm = rast(pad_bbox, reg) > 0
            best = None
            for v in range(1, nlab):
                comp = lab == v
                a = float(comp.sum()) * reg["px2"]
                if a < MF_MIN_AREA:
                    continue
                inb = float((comp & bbm).sum()) / float(comp.sum())
                if inb < 0.35:
                    continue
                rh = float(reg["rhatch"][comp].mean())
                dark = float((reg["gray"][comp] < 120).mean())
                sc = max(rh, dark * 0.6)
                if best is None or sc > best[0]:
                    best = (sc, rh, dark, comp, a)
            if best is None:
                entry["disposition"] = "dismissed"
                entry["reason"] = "no ink component in bbox"
                log.append(entry); continue
            sc, rh, dark, comp, a = best
            geoms = [shape(g) for g, _ in rfeatures.shapes(
                comp.astype(np.uint8), mask=comp, transform=reg["tr"])]
            comp_geom = max(geoms, key=lambda p: p.area)
            if work.core is not None:
                comp_geom = largest_poly(comp_geom.intersection(work.core)) \
                    or comp_geom
            cov = work.coverage(comp_geom)
            if cov > MF_MAX_COVER:
                entry["disposition"] = "dismissed"
                entry["reason"] = f"already covered {cov:.0%}"
                log.append(entry); continue
            if rh >= MF_ACCEPT or (dark >= SOLID_DARK and a <= 1500.0):
                geo, method = v4.regularize(comp_geom, theta)
                geo, method = v4.clamp_to_ink(geo, comp_geom, method)
                if (geo.is_valid and not geo.is_empty
                        and geo.area >= MF_MIN_AREA):
                    i = work.add(geo, f["tile"], rh,
                                 float(reg["ink"][comp].mean()),
                                 "v5vision+" + method, f["note"])
                    touch(geo)
                    entry["disposition"] = "fixed_add"
                    entry["reason"] = (f"hatch={rh:.2f} dark={dark:.2f} "
                                       f"area={geo.area:.0f}m2")
                else:
                    entry["disposition"] = "dismissed"
                    entry["reason"] = "regularization failed"
            elif rh >= MF_REVIEW:
                work.flag(comp_geom, cat, f["note"] + f" (rh={rh:.2f})",
                          args.pass_no, f["confidence"])
                entry["disposition"] = "needs_review"
                entry["reason"] = f"weak hatch {rh:.2f}"
            else:
                entry["disposition"] = "dismissed"
                entry["reason"] = f"no hatch evidence (rh={rh:.2f})"

        elif cat == "shape_mismatch":
            i, sc = work.best_overlap(bbox)
            if i is None or sc < 0.25:
                entry["disposition"] = "dismissed"
                entry["reason"] = "no polygon overlapping bbox"
                log.append(entry); continue
            g = work.rows[i]["geometry"]
            reg2 = region_from(src, g.union(bbox).bounds)
            inks = ink_polys(reg2)
            if not inks:
                entry["disposition"] = "dismissed"
                entry["reason"] = "no ink under polygon"
                log.append(entry); continue
            from shapely.ops import unary_union
            ink_u = unary_union(inks)
            over = g.difference(ink_u.buffer(1.2)).area / max(g.area, 1e-9)
            # hatch core the polygon SHOULD cover (within bbox+poly region)
            hc = cv2.morphologyEx(
                reg2["rhatch"], cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            hgs = [shape(gg) for gg, _ in rfeatures.shapes(
                hc, mask=hc.astype(bool), transform=reg2["tr"])]
            hgs = [h for h in hgs if h.area >= 20.0
                   and h.intersection(bbox.buffer(3)).area / h.area > 0.5]
            missed = 0.0
            hc_u = None
            if hgs:
                hc_u = unary_union(hgs)
                missed = (hc_u.difference(g.buffer(0.5)).area
                          / max(hc_u.area, 1e-9))
            if i in work.touched and args.pass_no > 1:
                work.flag(g, cat, "re-flagged after earlier fix: "
                          + f["note"], args.pass_no, f["confidence"])
                entry["disposition"] = "needs_review"
                entry["reason"] = "already touched this pass cycle"
            elif over > SM_OVERHANG:
                new = g.intersection(ink_u.buffer(0.5, join_style=2))
                new = largest_poly(new)
                if new is not None and new.area >= bc.MIN_AREA:
                    new = new.simplify(0.5, preserve_topology=True)
                    work.rows[i]["geometry"] = new
                    work.rows[i]["area_m2_3857"] = round(new.area, 1)
                    work.rows[i]["v5_action"] = "reshaped"
                    work.rows[i]["v5_note"] = f["note"]
                    work.rows[i]["reg_method"] = str(
                        work.rows[i].get("reg_method", "")) + "+v5clamp"
                    work.touched.add(i)
                    touch(g); touch(new)
                    entry["disposition"] = "fixed_reshape"
                    entry["reason"] = f"overhang {over:.0%} -> clamped"
                else:
                    work.flag(g, cat, f["note"], args.pass_no,
                              f["confidence"])
                    entry["disposition"] = "needs_review"
                    entry["reason"] = "clamp emptied polygon"
            elif missed > SM_MISSED and hc_u is not None:
                cand = largest_poly(unary_union(
                    [g.intersection(ink_u.buffer(0.5))] + hgs).buffer(0))
                if cand is not None and cand.area >= bc.MIN_AREA:
                    geo, method = v4.regularize(cand, theta)
                    geo, method = v4.clamp_to_ink(geo, cand, method)
                    iou = (geo.intersection(g).area
                           / max(geo.union(g).area, 1e-9))
                    if geo.is_valid and not geo.is_empty and iou < 0.98:
                        work.rows[i]["geometry"] = geo
                        work.rows[i]["area_m2_3857"] = round(geo.area, 1)
                        work.rows[i]["v5_action"] = "reshaped"
                        work.rows[i]["v5_note"] = f["note"]
                        work.rows[i]["reg_method"] = "v5reshape+" + method
                        work.touched.add(i)
                        touch(g); touch(geo)
                        entry["disposition"] = "fixed_reshape"
                        entry["reason"] = f"missed hatch {missed:.0%} -> extended"
                    else:
                        entry["disposition"] = "dismissed"
                        entry["reason"] = "extend produced no change"
                else:
                    work.flag(g, cat, f["note"], args.pass_no,
                              f["confidence"])
                    entry["disposition"] = "needs_review"
                    entry["reason"] = "extend failed"
            elif over > 0.12 or missed > 0.15:
                work.flag(g, cat, f["note"]
                          + f" (over={over:.0%} missed={missed:.0%})",
                          args.pass_no, f["confidence"])
                entry["disposition"] = "needs_review"
                entry["reason"] = f"borderline over={over:.0%} missed={missed:.0%}"
            else:
                entry["disposition"] = "dismissed"
                entry["reason"] = (f"shape checks out (over={over:.0%}, "
                                   f"missed={missed:.0%})")

        elif cat == "missing_split":
            i, sc = work.best_overlap(bbox)
            if i is None or sc < 0.25:
                entry["disposition"] = "dismissed"
                entry["reason"] = "no polygon overlapping bbox"
                log.append(entry); continue
            g = work.rows[i]["geometry"]
            reg2 = region_from(src, g.bounds)
            pm = rast(g, reg2)
            pieces = v4.divider_split(pm, reg2["divr"] & pm, reg2["px2"])
            good = []
            for p in pieces:
                a = float(p.sum()) * reg2["px2"]
                if a < SPLIT_PIECE:
                    continue
                gs = [shape(gg) for gg, _ in rfeatures.shapes(
                    p, mask=p.astype(bool), transform=reg2["tr"])]
                if gs:
                    good.append(max(gs, key=lambda q: q.area))
            if len(good) >= 2:
                if i in work.touched and args.pass_no > 1:
                    work.flag(g, cat, "re-flagged after earlier fix: "
                              + f["note"], args.pass_no, f["confidence"])
                    entry["disposition"] = "needs_review"
                    entry["reason"] = "already touched"
                    log.append(entry); continue
                work.alive[i] = False
                work.rows[i]["v5_action"] = "split_parent"
                for q in good:
                    geo, method = v4.regularize(q, theta)
                    geo, method = v4.clamp_to_ink(geo, q, method)
                    if geo.is_valid and not geo.is_empty \
                            and geo.area >= bc.MIN_AREA:
                        rh, dark, ink = poly_fracs(geo, reg2)
                        j = work.add(geo, f["tile"], rh, ink,
                                     "v5split+" + method, f["note"])
                        work.rows[j]["div_split"] = True
                touch(g)
                entry["disposition"] = "fixed_split"
                entry["reason"] = f"{len(good)} pieces"
            else:
                work.flag(g, cat, f["note"], args.pass_no, f["confidence"])
                entry["disposition"] = "needs_review"
                entry["reason"] = "no confident divider found"
        log.append(entry)

    # ---- write outputs ----
    cands, blocks = [], []
    for i, r in enumerate(work.rows):
        if not work.alive[i]:
            continue
        (blocks if r["geometry"].area > bc.MAX_AREA else cands).append(r)
    if os.path.exists(args.out):
        os.remove(args.out)
    gpd.GeoDataFrame(cands, crs=src.crs).to_file(
        args.out, driver="GPKG", layer="candidates")
    if blocks:
        gpd.GeoDataFrame(blocks, crs=src.crs).to_file(
            args.out, driver="GPKG", layer="oversize_blocks")
    if work.review:
        gpd.GeoDataFrame(work.review, crs=src.crs).to_file(
            args.out, driver="GPKG", layer="needs_review")
    if work.core is not None:
        gpd.GeoDataFrame({"geometry": [work.core]}, crs=src.crs).to_file(
            args.out, driver="GPKG", layer="core_v4")

    logpath = os.path.join(AUDIT, f"changelog_pass{args.pass_no}.json")
    with open(logpath, "w") as fjson:
        json.dump(log, fjson, indent=1)
    ctpath = os.path.join(AUDIT, f"changed_tiles_pass{args.pass_no}.txt")
    with open(ctpath, "w") as ftxt:
        ftxt.write(",".join(sorted(changed_tiles)))

    from collections import Counter
    disp = Counter(e["disposition"] for e in log)
    bycat = Counter((e["category"], e["disposition"]) for e in log)
    print(f"dispositions: {dict(disp)}")
    for k, v in sorted(bycat.items()):
        print(f"  {k[0]:>16} {k[1]:>14} {v}")
    print(f"out: {len(cands)} candidates + {len(blocks)} oversize "
          f"+ {len(work.review)} needs_review")
    print(f"changed tiles ({len(changed_tiles)}): {ctpath}")
    print(f"changelog: {logpath}")


if __name__ == "__main__":
    main()
