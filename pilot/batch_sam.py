#!/usr/bin/env python3
"""Overnight SAM automatic-mask candidate generator over the city-core
extent of the 1762 Clarkson & Biddle COG.

- SAM automatic mask generation (samgeo / segment-anything), ViT-H by
  default for accuracy (set SAM_MODEL=vit_b env var to fall back).
- 1024 px stride + 128 px margin: SAM resizes its input to 1024 px on the
  longest side internally, so 1280 px reads keep near-full map detail
  (a 2048 tile would be halved to 0.8 m/px before the model sees it).
- Blank-paper tiles skipped by ink density; masks must cover mostly ink.
- Per-tile GeoJSON checkpoints in results/sam_tiles/ -> resumable; merged
  GeoPackage rewritten every few tiles and at the end.

Output: pilot/results/candidates_sam.gpkg

Run detached:
  nohup env/bin/python pilot/batch_sam.py > pilot/results/batch_sam.log 2>&1 &
"""
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio import features as rfeatures
import geopandas as gpd
from shapely.geometry import shape, mapping

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_common as bc

STRIDE = 1024
MARGIN = 128
MODEL = os.environ.get("SAM_MODEL", "vit_h")
MIN_MASK_INK = 0.35        # mask must cover >= this fraction of ink pixels
DEDUPE_IOU = 0.5           # greedy per-tile dedupe of overlapping masks
TILE_DIR = os.path.join(HERE, "results", "sam_tiles")
OUT = os.path.join(HERE, "results", "candidates_sam.gpkg")
MERGE_EVERY = 3            # rewrite merged gpkg every N processed tiles
MAX_TILES = int(os.environ.get("SAM_MAX_TILES", "0"))  # 0 = no limit (testing)


def polys_from_mask(seg, transform):
    m = seg.astype(np.uint8)
    return [shape(g) for g, v in
            rfeatures.shapes(m, mask=seg, transform=transform) if v == 1]


def merge_tiles(crs):
    records = []
    for fn in sorted(os.listdir(TILE_DIR)):
        if not fn.endswith(".geojson"):
            continue
        with open(os.path.join(TILE_DIR, fn)) as f:
            fc = json.load(f)
        for feat in fc.get("features", []):
            rec = dict(feat["properties"])
            rec["geometry"] = shape(feat["geometry"])
            records.append(rec)
    if not records:
        return 0
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(OUT, driver="GPKG", layer="candidates")
    return len(gdf)


def main():
    t0 = time.time()
    os.makedirs(TILE_DIR, exist_ok=True)
    src = rasterio.open(bc.COG)
    core = bc.load_core_poly(src)
    tiles = list(bc.iter_tiles(src, core, STRIDE, MARGIN))
    print(f"model={MODEL}  {len(tiles)} tiles "
          f"(stride {STRIDE}px, margin {MARGIN}px)", flush=True)

    from samgeo import SamGeo
    sam = SamGeo(
        model_type=MODEL,
        automatic=True,
        sam_kwargs=dict(
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=60,
        ),
    )
    print(f"model loaded: {time.time()-t0:.1f}s", flush=True)

    done = skipped = 0
    times = []
    for i, (tid, win, own) in enumerate(tiles):
        ckpt = os.path.join(TILE_DIR, f"{tid}.geojson")
        if os.path.exists(ckpt):
            print(f"[{i+1}/{len(tiles)}] {tid} already done, skipping",
                  flush=True)
            continue
        t1 = time.time()
        data = src.read(window=win)
        gray = bc.gray_from_tile(data)
        ink = bc.ink_mask(gray)
        frac = float(ink.mean())
        if frac < bc.MIN_INK_FRACTION:
            with open(ckpt, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
            skipped += 1
            print(f"[{i+1}/{len(tiles)}] {tid} skip (ink {frac:.4f})",
                  flush=True)
            continue

        img = np.moveaxis(data[:3], 0, -1).copy()
        if data.shape[0] >= 4:
            img[data[3] == 0] = 255      # off-sheet -> white paper
        masks = sam.mask_generator.generate(img)
        masks.sort(key=lambda m: -m.get("predicted_iou", 0.0))

        transform = src.window_transform(win)
        kept_geoms, feats = [], []
        for m in masks:
            seg = m["segmentation"]
            npx = int(seg.sum())
            if npx == 0 or npx > 0.5 * seg.size:
                continue                  # background / whole-tile masks
            if float((gray[seg] < bc.INK_THRESHOLD).mean()) < MIN_MASK_INK:
                continue                  # mostly paper, not a building
            for geom in polys_from_mask(seg, transform):
                rp = geom.representative_point()
                if not own.contains(rp):  # tile-ownership dedupe
                    continue
                if not core.intersects(geom):
                    continue
                if not bc.passes_filters(geom):
                    continue
                dup = False
                for kg in kept_geoms:
                    inter = geom.intersection(kg).area
                    if inter > 0:
                        iou = inter / (geom.area + kg.area - inter)
                        if iou > DEDUPE_IOU:
                            dup = True
                            break
                if dup:
                    continue
                geom_s = geom.simplify(0.4, preserve_topology=True)
                kept_geoms.append(geom)
                feats.append({
                    "type": "Feature",
                    "properties": {
                        "tile": tid,
                        "area_m2_3857": round(geom_s.area, 1),
                        "sam_iou": round(float(m.get("predicted_iou", 0)), 3),
                        "sam_stability": round(
                            float(m.get("stability_score", 0)), 3),
                    },
                    "geometry": mapping(geom_s),
                })
        with open(ckpt, "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        dt = time.time() - t1
        times.append(dt)
        done += 1
        if MAX_TILES and done >= MAX_TILES:
            print(f"SAM_MAX_TILES={MAX_TILES} reached, stopping early",
                  flush=True)
            break
        remaining = sum(1 for t, _, _ in tiles
                        if not os.path.exists(
                            os.path.join(TILE_DIR, f"{t}.geojson")))
        eta_h = remaining * (sum(times) / len(times)) / 3600
        print(f"[{i+1}/{len(tiles)}] {tid} ink={frac:.3f} masks={len(masks)} "
              f"kept={len(feats)} {dt:.1f}s | avg {sum(times)/len(times):.0f}"
              f"s/tile, {remaining} left, ETA {eta_h:.1f}h", flush=True)
        if done % MERGE_EVERY == 0:
            n = merge_tiles(src.crs)
            print(f"  merged checkpoint -> {OUT} ({n} polygons)", flush=True)

    n = merge_tiles(src.crs)
    print(f"DONE. wrote {OUT}: {n} polygons; {skipped} blank tiles skipped; "
          f"total {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
