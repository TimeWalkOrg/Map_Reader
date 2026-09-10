#!/usr/bin/env python
"""Hybrid Easburn building-instance candidates (run with geo-sam/env python).

The NYC-trained building head only learned 'built-up block vs street' from weak labels and does NOT transfer to
Easburn's convention (buildings = hatched/solid dark blocks). So instances come from the plate ink itself:
  1. binary ink (shared preprocessing, 0.5 m/px) -> close 5 px (fill hatch gaps) -> open 5 px (drop thin outlines/text strokes)
  2. remove predicted road corridors (NYC-trained road U-Net, prob >= 0.5, dilated 1 m) and everything outside the drawn city
  3. connected components -> polygons; keep 20 m2 <= area <= 6000 m2, darkness fill >= 0.25, solidity >= 0.55
  4. refine each blob with the existing Easburn Geo-SAM (SAM-B) embeddings: point + bbox prompt; accept SAM mask when
     IoU(blob, sam) >= 0.3 and sam area <= 3x blob area, else keep the blob.
origin = 'machine_candidate_nyc_trained' (the NYC-trained road model gates the search space); method column records provenance.
"""
import json, time, sys, os
import numpy as np, cv2, rasterio, geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from geosam import BoundingBox, FeatureQueryEngine, Points, PromptSet
from geosam.runtime import create_model_spec_from_checkpoint

RUN = os.path.dirname(os.path.abspath(__file__)); GS = os.path.abspath(f"{RUN}/../../geo-sam")
T0 = time.time()
d = np.load(f"{RUN}/scratch/easburn_input.npz"); ch = d["chans"]; drawn = d["drawn"]; tr = rasterio.Affine(*d["transform"])
p = np.load(f"{RUN}/scratch/easburn_probs.npz"); road = p["roads"]
ink = (ch[1] > 0).astype(np.uint8)
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k5)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)
roadmask = cv2.dilate((road >= 128).astype(np.uint8), np.ones((5, 5), np.uint8))
m[(roadmask == 1) | (drawn == 0)] = 0
n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
print("components", n - 1, f"{time.time()-T0:.0f}s", flush=True)
recs = []
dark = ch[0].astype(np.float32) / 255
for i in range(1, n):
    a_px = st[i, cv2.CC_STAT_AREA]; area = a_px * 0.25
    if area < 20 or area > 6000: continue
    x, y, w, h = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
    fill = float(dark[y:y + h, x:x + w][sub == 1].mean())
    if fill < 0.25: continue
    polys = [shape(g) for g, v in features.shapes(sub, mask=sub.astype(bool), transform=tr * rasterio.Affine.translation(x, y)) if v == 1]
    poly = unary_union(polys).buffer(0)
    if poly.is_empty: continue
    solidity = poly.area / poly.convex_hull.area if poly.convex_hull.area else 0
    if solidity < 0.55: continue
    recs.append(dict(blob=poly.simplify(0.5), area_m2=float(poly.area), fill=fill, solidity=float(solidity)))
print("blobs after filters", len(recs), f"{time.time()-T0:.0f}s", flush=True)

spec = create_model_spec_from_checkpoint(f"{GS}/models/sam_b.pt", model_id="sam_b", device="cpu")
engine = FeatureQueryEngine(f"{GS}/features_easburn_sam_b/tw_1776_philadelphia_map_easburn_plan_v2_cog", spec)
crs = engine.manifest.crs
out = []; n_sam = 0; t_sam = time.time()
try:
    for r in recs:
        blob = r["blob"]; click = blob.representative_point(); b = blob.buffer(1.0).bounds
        geom, method, score, iou = blob, "ink_blob", None, None
        try:
            res = engine.query(PromptSet(points=Points([[click.x, click.y]], labels=[1], crs=crs), bbox=BoundingBox(b[0], b[1], b[2], b[3], crs=crs)))
            mask = res.mask_array[0].astype(np.uint8)
            polys = [shape(g) for g, v in features.shapes(mask, mask=mask.astype(bool), transform=res.mask_transform) if v == 1]
            hit = [q for q in polys if q.covers(click)]
            sam = (unary_union(hit) if hit else (max(polys, key=lambda q: q.area) if polys else None))
            if sam is not None and not sam.is_empty:
                sam = sam.buffer(0); inter = sam.intersection(blob).area; iou = inter / (sam.area + blob.area - inter)
                score = float(res.scores.max())
                if iou >= 0.3 and sam.area <= 3 * blob.area:
                    geom, method = sam.simplify(0.5), "ink_blob+geosam_refine"; n_sam += 1
        except ValueError:
            pass  # no covering chip
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda q: q.area)
        out.append(dict(geometry=geom, method=method, confidence=round(0.5 * r["fill"] + 0.5 * r["solidity"], 3), ink_fill=round(r["fill"], 3),
                        solidity=round(r["solidity"], 3), blob_area_m2=round(r["area_m2"], 1), area_m2=round(float(geom.area), 1),
                        sam_score=None if score is None else round(score, 3), sam_blob_iou=None if iou is None else round(float(iou), 3)))
finally:
    engine.close()
g = gpd.GeoDataFrame(out, crs="EPSG:3857")
g["origin"] = "machine_candidate_nyc_trained"; g["model"] = "unet32_roads_nyc1776(gate)+ink_morphology+geosam_sam_b"
g["oversize_flag"] = g.area_m2 > 1500
g = g[g.is_valid & ~g.is_empty]
g.to_file(f"{RUN}/easburn_building_candidates.gpkg", driver="GPKG")
json.dump(dict(n_components=int(n - 1), n_blobs_after_filters=len(recs), n_candidates=int(len(g)), n_geosam_refined=n_sam,
               geosam_seconds=round(time.time() - t_sam, 1), total_seconds=round(time.time() - T0, 1),
               filters=dict(area_m2=[20, 6000], fill_min=0.25, solidity_min=0.55, close_px=5, open_px=5, road_gate="unet roads prob>=0.5 dilated 2px"),
               all_valid=bool(g.is_valid.all()), crs="EPSG:3857"), open(f"{RUN}/easburn_instance_manifest.json", "w"), indent=1)
print("done", len(g), "geosam refined", n_sam, f"{time.time()-T0:.0f}s")
