#!/usr/bin/env python
"""Sliding-window inference (CPU ok) + vectorisation of machine candidates.
Usage: python infer_vectorize.py --ckpt ckpt_dir --which easburn|nyc
Writes probability maps (uint8 npz) and, for easburn, candidate GPKGs in EPSG:3857 (origin='machine_candidate_nyc_trained')."""
import argparse, json, time, sys, os
import numpy as np, torch, torch.nn.functional as F, cv2, rasterio, geopandas as gpd, pandas as pd
from rasterio import features
from shapely.geometry import shape, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(__file__))
from train_model import UNet  # noqa

RUN = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=f"{RUN}/ckpt")
p.add_argument("--which", default="easburn", choices=["easburn", "nyc"])
p.add_argument("--win", type=int, default=512)
p.add_argument("--overlap", type=int, default=64)
p.add_argument("--road-thr", type=float, default=0.4)
p.add_argument("--reuse-probs", action="store_true", help="skip inference; re-vectorise saved probability maps")
a = p.parse_args()
torch.set_num_threads(max(4, os.cpu_count() - 2))
T0 = time.time()


def predict(model, chans, valid=None):
    """chans uint8 [2,H,W] -> prob uint8 [H,W]; cosine-blended overlapping windows."""
    _, H, W = chans.shape
    win, ov = a.win, a.overlap; step = win - ov
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    ramp = np.minimum(np.arange(win) + 1, ov) / ov; wgt = np.minimum.outer(ramp, ramp) * np.minimum.outer(ramp[::-1], ramp[::-1])
    wgt = np.sqrt(wgt).astype(np.float32)
    ys = list(range(0, max(1, H - win), step)) + [max(0, H - win)]; xs = list(range(0, max(1, W - win), step)) + [max(0, W - win)]
    ys = sorted(set(ys)); xs = sorted(set(xs))
    tiles = [(y, x) for y in ys for x in xs if valid is None or valid[y:y + win, x:x + win].any()]
    with torch.no_grad():
        for i in range(0, len(tiles), 8):
            bt = tiles[i:i + 8]
            xb = np.stack([np.pad(chans[:, y:y + win, x:x + win], ((0, 0), (0, win - min(win, H - y)), (0, win - min(win, W - x)))) for y, x in bt])
            pr = torch.sigmoid(model(torch.from_numpy(xb).float() / 255)).numpy()[:, 0]
            for (y, x), pp in zip(bt, pr):
                h, w = min(win, H - y), min(win, W - x)
                acc[y:y + h, x:x + w] += pp[:h, :w] * wgt[:h, :w]; wsum[y:y + h, x:x + w] += wgt[:h, :w]
    return (np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 0) * 255).astype(np.uint8)


def load(task):
    m = UNet(2, 32); m.load_state_dict(torch.load(f"{a.ckpt}/{task}_best.pt", map_location="cpu")); m.eval(); return m


def trace_paths(skel):
    """Trace 8-connected skeleton into pixel paths between junctions/endpoints. Returns list of (path, end_degrees)."""
    H, W = skel.shape
    pts = np.argwhere(skel)
    S = set(map(tuple, pts))
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (p[0] + dy, p[1] + dx) in S]
    deg = {p: len(nb(p)) for p in S}
    nodes = {p for p, d in deg.items() if d != 2}
    used = set(); paths = []
    for n in nodes:
        for q in nb(n):
            e = (n, q) if n < q else (q, n)
            if e in used: continue
            path = [n, q]; used.add(e); prev, cur = n, q
            while cur not in nodes:
                nxt = [z for z in nb(cur) if z != prev]
                if not nxt: break
                prev, cur = cur, nxt[0]; e = (prev, cur) if prev < cur else (cur, prev)
                if e in used: break
                used.add(e); path.append(cur)
            paths.append((path, (deg[path[0]], deg[path[-1]])))
    # isolated loops without nodes
    for p0 in S:
        if p0 in nodes or any(((p0, q) if p0 < q else (q, p0)) in used for q in nb(p0)): continue
        path = [p0]; prev, cur = None, p0
        while True:
            nxt = [z for z in nb(cur) if z != prev]
            if not nxt: break
            e = (cur, nxt[0]) if cur < nxt[0] else (nxt[0], cur)
            if e in used: break
            used.add(e); prev, cur = cur, nxt[0]; path.append(cur)
        paths.append((path, (2, 2)))
    return paths


def prune_skeleton(skel, min_spur_px=30, iters=3):
    """Remove terminal branches (one end is an endpoint) shorter than min_spur_px; repeat."""
    skel = skel.copy()
    for _ in range(iters):
        removed = 0
        for path, (d0, d1) in trace_paths(skel):
            if (d0 == 1 or d1 == 1) and len(path) < min_spur_px:
                keep_end = [p for p, d in ((path[0], d0), (path[-1], d1)) if d > 1]
                for r, c in path:
                    if (r, c) not in keep_end: skel[r, c] = False
                removed += 1
        if removed == 0: break
    return skel


def skeleton_to_lines(skel, prob, tr, min_len_m=15.0, simplify_m=1.5):
    lines = []
    for path, _ in trace_paths(skel):
        if len(path) < 3: continue
        ls = LineString([tr * (c + 0.5, r + 0.5) for r, c in path])
        if ls.length >= min_len_m:
            conf = float(np.mean([prob[r, c] for r, c in path]) / 255)
            lines.append((ls.simplify(simplify_m), conf, ls.length))
    return lines


def vectorize_roads(prob, tr, valid, thr=128):
    ps = cv2.GaussianBlur(prob, (0, 0), 2.0)
    m = (ps >= thr) & (valid > 0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop thin bridges between buildings
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.isin(lab, np.where(st[:, cv2.CC_STAT_AREA] >= 400)[0][1:]) if n > 1 else m.astype(bool)
    skel = prune_skeleton(skeletonize(keep), min_spur_px=30, iters=3)
    lines = skeleton_to_lines(skel, prob, tr)
    g = gpd.GeoDataFrame({"confidence": [l[1] for l in lines], "length_m": [l[2] for l in lines],
                          "origin": "machine_candidate_nyc_trained", "model": "unet32_roads_nyc1776",
                          "geometry": [l[0] for l in lines]}, crs="EPSG:3857")
    return g, keep.astype(np.uint8)


def vectorize_buildings(prob, tr, valid, thr=128, min_area_m2=15.0):
    m = ((prob >= thr) & (valid > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    recs = []
    for geom, v in features.shapes(m, mask=m.astype(bool), transform=tr, connectivity=4):
        poly = shape(geom).buffer(0)
        if poly.area < min_area_m2: continue
        poly = poly.simplify(0.5, preserve_topology=True)
        if poly.is_empty or not poly.is_valid: continue
        # mean prob inside
        r = features.rasterize([(poly, 1)], out_shape=prob.shape, transform=tr, dtype=np.uint8) if poly.area < 5e4 else None
        conf = float(prob[r == 1].mean() / 255) if r is not None and (r == 1).any() else float(v)
        recs.append(dict(confidence=conf, area_m2=float(poly.area), geometry=poly))
    g = gpd.GeoDataFrame(recs, crs="EPSG:3857")
    g["origin"] = "machine_candidate_nyc_trained"; g["model"] = "unet32_buildings_nyc1776_weaklabels"
    g["oversize_flag"] = g.area_m2 > 1500
    return g


if a.which == "easburn":
    d = np.load(f"{RUN}/scratch/easburn_input.npz"); chans = d["chans"]; valid = d["drawn"]; tr = rasterio.Affine(*d["transform"])
else:
    d = np.load(f"{RUN}/scratch/nyc_full.npz"); chans = d["chans"]; valid = (d["off"] == 0).astype(np.uint8); tr = rasterio.Affine(*d["transform"])

out = {}
if a.reuse_probs and os.path.exists(f"{RUN}/scratch/{a.which}_probs.npz"):
    z = np.load(f"{RUN}/scratch/{a.which}_probs.npz"); out = {"roads": z["roads"], "buildings": z["buildings"]}
else:
    for task in ("roads", "buildings"):
        t = time.time(); pr = predict(load(task), chans, valid); out[task] = pr
        print(task, "inference s", round(time.time() - t, 1), "mean prob", pr.mean() / 255, flush=True)
    np.savez_compressed(f"{RUN}/scratch/{a.which}_probs.npz", roads=out["roads"], buildings=out["buildings"], transform=np.array(list(tr)[:6]))

t = time.time()
gr, rmask = vectorize_roads(out["roads"], tr, valid, thr=int(a.road_thr * 255))
gb = vectorize_buildings(out["buildings"], tr, valid)
print("vectorised roads", len(gr), "buildings", len(gb), "s", round(time.time() - t, 1), flush=True)
tag = "easburn" if a.which == "easburn" else "nyc_fullplate"
bname = f"{tag}_builtup_unet_candidates.gpkg" if tag == "easburn" else f"{tag}_building_candidates.gpkg"
gr.to_file(f"{RUN}/{tag}_road_candidates.gpkg", driver="GPKG"); gb.to_file(f"{RUN}/{bname}", driver="GPKG")
json.dump(dict(which=a.which, seconds=round(time.time() - T0, 1), n_roads=int(len(gr)), road_km=float(gr.length.sum() / 1000),
               n_buildings=int(len(gb)), n_oversize=int(gb.oversize_flag.sum()) if len(gb) else 0,
               all_valid_roads=bool(gr.is_valid.all()), all_valid_buildings=bool(gb.is_valid.all()),
               crs="EPSG:3857", building_threshold=0.5, road_threshold=a.road_thr, win=a.win, overlap=a.overlap), open(f"{RUN}/{tag}_infer_manifest.json", "w"), indent=1)
print("done", round(time.time() - T0, 1))
