#!/usr/bin/env python
"""Prepare NYC 1776 fire-map training tiles + Easburn inference array at a common 0.5 m/px (EPSG:3857) grid.

Two SEPARATE label stacks (per active_dataset.json requirements):
  roads     : street-CORRIDOR mask from 1776_nyc_vector_road_graph (roads.cga only; piers/greenspace/pavement excluded),
              target = centerline buffered by half-width clip(0.4*strtwdth, 2.5, 7) m, ignore band +2.5 m outside,
              off-plate ignored. v1 (thin 2 m centerline buffer) was abandoned: val IoU stalled ~0.10 because the graph
              itself sits ~1.1 m off the drawn corridor centre. Centerlines are recovered by skeletonising the corridor.
  buildings : WEAK labels from 1776_nyc_parcels_buildings + 1776_nyc_parcels_landmarks
              (excluding off_plate, blank_plate, modern_proxy_suspect, landmarks not on ink, id 54 fixed via buffer(0)),
              boundary ignore band +/-3 m, negatives valid ONLY inside the 10 exhaustive blocks and
              inside the road corridor (centerline buffer 3 m); everything else unlabeled -> ignore.
Split: per-pixel from block_coverage.gpkg (val = NE band, block polygons dilated 12 m to claim adjacent street halves).
Outputs: scratch/nyc_tiles.npz, scratch/nyc_full.npz (full-res arrays for val scoring), scratch/easburn_input.npz
"""
import json, hashlib, time, sys
import numpy as np, cv2, rasterio, geopandas as gpd, pandas as pd
from rasterio import features
from rasterio.enums import Resampling
from shapely.geometry import box
from shapely.ops import unary_union

T0 = time.time()
RUN = "/Users/gabriel/.openclaw/workspace-timewalker/Map_Reader/runs/nyc2easburn_2026-09-10"
TR = "/Users/gabriel/.openclaw/workspace-timewalker/Map_Reader/training/sunil_nyc_1776_fire_map_2026-09-10"
NYC = "/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif"
EAS = "/Users/gabriel/.openclaw/workspace-timewalker/philly_georef_work/tw_1776_philadelphia_map_easburn_plan_v2_cog.tif"
DRAWN = "/Users/gabriel/.openclaw/workspace-timewalker/Map_Reader/geo-sam/features_easburn_sam_b/drawn_city.geojson"
RES = 0.5           # m/px in EPSG:3857 units (ground ~0.38 m at 40N)
TILE, STRIDE = 320, 160
LANDMARKS_NOT_ON_INK = [17, 25, 26, 38, 39, 44, 54, 60, 63, 72]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 24), b''):
            h.update(ch)
    return h.hexdigest()


def ink_channels(gray):
    """Shared preprocessing for both plates. gray uint8 (paper bright, ink dark).
    ch0: background-normalised darkness (0 paper .. 255 ink); ch1: adaptive-threshold binary ink."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.blur(bg, (41, 41)).astype(np.float32) + 1.0
    norm = np.clip(gray.astype(np.float32) / bg, 0, 1)
    ch0 = ((1.0 - norm) * 255).astype(np.uint8)
    ch1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 18)
    return np.stack([ch0, ch1])


def resample_read(path, res, mask_geom=None):
    with rasterio.open(path) as s:
        f = s.res[0] / res
        out_h, out_w = int(round(s.height * f)), int(round(s.width * f))
        rgb = s.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
        tr = s.transform * rasterio.Affine.scale(s.width / out_w, s.height / out_h)
        meta = dict(path=path, sha256=sha256(path), src_size=[s.width, s.height], src_res=s.res[0], crs=str(s.crs),
                    src_transform=list(s.transform)[:6], out_size=[out_w, out_h], out_res=res, out_transform=list(tr)[:6])
    gray = cv2.cvtColor(np.moveaxis(rgb, 0, -1), cv2.COLOR_RGB2GRAY)
    return gray, tr, meta


def rasterize(geoms, shape, tr, all_touched=False):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, np.uint8)
    return features.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=tr, all_touched=all_touched, dtype=np.uint8)


manifest = {"res_m_per_px_3857": RES, "tile": TILE, "stride": STRIDE,
            "roads_target": "v3 corridor: only qa_roads-scorable edges (n>=3, width>=2 m) buffered clip(0.45*measured width,2,7) m, ignore band +2.5 m; unscorable graph edges ignored (12 m). v1 thin 2 m centerline: val IoU ~0.10; v2 all-edge corridor: val IoU ~0.20 (undrawn NE streets taught as roads)",
            "buildings_target": "weak polygons eroded 3 m = positives; negatives only exhaustive blocks + road corridor(3 m); ignore elsewhere"}

# ---------------- NYC ----------------
gray, tr, meta = resample_read(NYC, RES)
manifest["nyc_raster"] = meta
H, W = gray.shape
print("NYC resampled", W, H, f"{time.time()-T0:.0f}s")
chans = ink_channels(gray)
# off-plate: pure black region (large), opened to drop strokes, dilated a bit
black = (gray < 20).astype(np.uint8)
off = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8))
off = cv2.dilate(off, np.ones((21, 21), np.uint8))
print("off-plate frac", off.mean())

roads = gpd.read_file(f"{TR}/1776_nyc_vector_road_graph.gpkg").to_crs(3857)
roads["geometry"] = roads.geometry.force_2d() if hasattr(roads.geometry, "force_2d") else roads.geometry
core = roads[roads.rulefile == "roads.cga"]
other = roads[roads.rulefile != "roads.cga"]
road_line = unary_union(core.geometry.values)
# v3: only edges with a MEASURED white corridor on the plate (qa_roads: n>=3 samples, width>=2 m) are positives;
# graph edges with no drawn corridor (blank paper / planned streets / through hatching) are IGNORED (buffer 12 m),
# never taught as roads (v2 taught them -> val IoU capped ~0.2 because the NE val band is mostly undrawn streets).
qr = pd.read_csv(f"{TR}/qa_roads.csv").set_index("ogc_fid")
core = core.join(qr[["n", "median_width_m"]], on="ogc_fid")
scorable = core[(core.n >= 3) & (core.median_width_m >= 2.0)]
unscorable = core[~core.index.isin(scorable.index)]
GROUND_TO_3857 = 1 / 0.758  # cos(40.71 deg)
halfw = (scorable.median_width_m * GROUND_TO_3857 * 0.45).clip(2.0, 7.0)
corr = unary_union([g.buffer(h) for g, h in zip(scorable.geometry.values, halfw.values)])
corr_band = unary_union([g.buffer(h + 2.5) for g, h in zip(scorable.geometry.values, halfw.values)])
road_pos = rasterize([corr], (H, W), tr)
road_band = rasterize([corr_band], (H, W), tr)
other_ign = rasterize([unary_union(list(other.geometry.values) + list(unscorable.geometry.values)).buffer(12.0)], (H, W), tr)
print("road edges: scorable", len(scorable), "unscorable(ignored)", len(unscorable), "other rulefiles(ignored)", len(other))
road_tgt = road_pos
road_w = np.ones((H, W), np.uint8)
road_w[(road_band == 1) & (road_pos == 0)] = 0
road_w[other_ign == 1] = 0
road_w[off == 1] = 0
road_corridor = rasterize([road_line.buffer(3.0)], (H, W), tr)

bl = gpd.read_file(f"{TR}/1776_nyc_parcels_buildings.gpkg").to_crs(3857)
qa = pd.read_csv(f"{TR}/qa_buildings.csv").set_index("id")
qc = pd.read_csv(f"{TR}/qa_cells_buildings.csv").set_index("id")
bl = bl.join(qa[["qa_class", "on_plate"]], on="id").join(qc[["blank_plate"]], on="id")
keep = bl[(bl.on_plate == True) & (bl.blank_plate != True) & (bl.qa_class != "modern_proxy_suspect")]
lm = gpd.read_file(f"{TR}/1776_nyc_parcels_landmarks.gpkg").to_crs(3857)
lm["geometry"] = lm.geometry.buffer(0)
lm_split = pd.read_csv(f"{TR}/proposed_split_landmarks.csv").set_index("id")
lm = lm.join(lm_split[["on_plate"]], on="id")
lm_keep = lm[(lm.on_plate == True) & (~lm.id.isin(LANDMARKS_NOT_ON_INK))]
pos_geoms = list(keep.geometry.values) + list(lm_keep.geometry.values)
pos_u = unary_union(pos_geoms)
b_pos = rasterize([pos_u], (H, W), tr)
b_inner = rasterize([pos_u.buffer(-3.0)], (H, W), tr)
b_outer = rasterize([pos_u.buffer(3.0)], (H, W), tr)
blocks = gpd.read_file(f"{TR}/block_coverage.gpkg").to_crs(3857)
exh = rasterize(list(blocks[blocks.cls == "exhaustive"].geometry.values), (H, W), tr)
neg_valid = ((exh == 1) | (road_corridor == 1)).astype(np.uint8)
b_tgt = b_pos
b_w = np.zeros((H, W), np.uint8)
b_w[b_inner == 1] = 1                                  # confident positives (weak, eroded 3 m)
b_w[(neg_valid == 1) & (b_outer == 0)] = 1             # confident negatives
b_w[off == 1] = 0
print("buildings kept", len(keep), "landmarks kept", len(lm_keep), "pos px", int(b_pos.sum()), "valid px", int(b_w.sum()))

# split raster: 1 train, 2 val, 0 none — block polygons dilated 12 m
spl = np.zeros((H, W), np.uint8)
for lab, val in (("train", 1), ("val", 2)):
    g = blocks[blocks.split == lab].geometry.buffer(12.0)
    m = rasterize(list(g.values), (H, W), tr)
    spl[(m == 1) & (spl == 0)] = val
# val wins where dilations overlap (protect val from leakage)
mval = rasterize(list(blocks[blocks.split == "val"].geometry.buffer(12.0).values), (H, W), tr)
spl[mval == 1] = 2
print("split px train/val/none", [(spl == v).sum() for v in (1, 2, 0)])

np.savez_compressed(f"{RUN}/scratch/nyc_full.npz", chans=chans, road_tgt=road_tgt, road_w=road_w, b_tgt=b_tgt, b_w=b_w,
                    split=spl, off=off, transform=np.array(list(tr)[:6]))

# tiles
imgs, rt, rw, bt, bw, sp, org = [], [], [], [], [], [], []
for y in range(0, H - TILE + 1, STRIDE):
    for x in range(0, W - TILE + 1, STRIDE):
        sl = (slice(y, y + TILE), slice(x, x + TILE))
        s = spl[sl]
        n_tr, n_va = (s == 1).sum(), (s == 2).sum()
        if n_va > 0:
            split = 2
        elif n_tr > 0:
            split = 1
        else:
            continue
        rw_t = road_w[sl].copy(); bw_t = b_w[sl].copy()
        if split == 1:   # training tile: never learn from val pixels
            rw_t[s == 2] = 0; bw_t[s == 2] = 0
        else:            # val tile: evaluate only on val pixels
            rw_t[s != 2] = 0; bw_t[s != 2] = 0
        if rw_t.sum() == 0 and bw_t.sum() == 0:
            continue
        imgs.append(chans[:, y:y + TILE, x:x + TILE]); rt.append(road_tgt[sl]); rw.append(rw_t)
        bt.append(b_tgt[sl]); bw.append(bw_t); sp.append(split); org.append((y, x))
imgs = np.stack(imgs); sp = np.array(sp, np.uint8)
print("tiles", imgs.shape, "train", (sp == 1).sum(), "val", (sp == 2).sum())
np.savez_compressed(f"{RUN}/scratch/nyc_tiles.npz", img=imgs, road_tgt=np.stack(rt), road_w=np.stack(rw),
                    b_tgt=np.stack(bt), b_w=np.stack(bw), split=sp, origin=np.array(org))
manifest["nyc_tiles"] = {"n_train": int((sp == 1).sum()), "n_val": int((sp == 2).sum()),
                         "buildings_used": int(len(keep)), "landmarks_used": int(len(lm_keep)),
                         "roads_edges_used": int(len(scorable)), "roads_edges_ignored_no_drawn_corridor": int(len(unscorable)),
                         "roads_edges_ignored_other_rulefiles": int(len(other)),
                         "off_plate_frac": float(off.mean())}

# ---------------- Easburn ----------------
gray_e, tr_e, meta_e = resample_read(EAS, RES)
manifest["easburn_raster"] = meta_e
He, We = gray_e.shape
drawn = gpd.read_file(DRAWN).to_crs(3857)
dmask = rasterize(list(drawn.geometry.values), (He, We), tr_e)
gray_e = gray_e.copy(); gray_e[dmask == 0] = int(np.median(gray_e[dmask == 1]))  # outside drawn city -> paper tone
chans_e = ink_channels(gray_e)
np.savez_compressed(f"{RUN}/scratch/easburn_input.npz", chans=chans_e, drawn=dmask, transform=np.array(list(tr_e)[:6]))
print("Easburn", We, He, "drawn frac", dmask.mean(), f"{time.time()-T0:.0f}s")

# evidence of preprocessing (same window as earlier crops)
cv2.imwrite(f"{RUN}/evidence/preproc_nyc_center.png", np.hstack([gray[H//2-200:H//2+200, W//2-200:W//2+200], chans[0, H//2-200:H//2+200, W//2-200:W//2+200], chans[1, H//2-200:H//2+200, W//2-200:W//2+200]]))
cv2.imwrite(f"{RUN}/evidence/preproc_easburn_center.png", np.hstack([gray_e[He//2-200:He//2+200, We//2-200:We//2+200], chans_e[0, He//2-200:He//2+200, We//2-200:We//2+200], chans_e[1, He//2-200:He//2+200, We//2-200:We//2+200]]))
manifest["prep_seconds"] = round(time.time() - T0, 1)
json.dump(manifest, open(f"{RUN}/prep_manifest.json", "w"), indent=1)
print("done", manifest["prep_seconds"])
