"""Cell-match QA + block coverage QA for the 1776 NYC fire-map plate.

On this plate a house lot is drawn as a white cell bounded by ink (the tick-divided frontage
strips); larger public buildings are solid black blobs. A polygon *traced from the plate* should
coincide with one white cell. For every polygon we therefore compute:
  * best_cell_iou   – IoU with the single best-matching drawn cell (cell dilated by half a stroke so
                      that it reaches the stroke centre-line, where a tracer would click);
  * union_cell_iou  – IoU with the union of all cells it overlaps by >= 20 % of the cell area;
  * cells_spanned   – how many drawn cells it covers by >= 20 % each (1 = one lot; >1 = merged rows);
  * blank_plate     – no ink within 6 px of the polygon boundary and no cell overlap (placed on a
                      blank part of the plate: not traceable from this raster).
Block coverage: blocks are polygonised from the road graph. For each block we count drawn cells
(white components 8–1500 ground m², not touching the block edge corridor) and how many are
covered >= 50 % by a building/landmark polygon. Classes: exhaustive (>= 90 % of >= 3 cells),
partial (30–90 %), sparse (< 30 %), no_drawn_cells, off_plate.
Run qa_alignment.py first (it caches /tmp/fire_ink.npy and /tmp/fire_plate.npy).
"""
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from PIL import Image, ImageDraw, ImageFont
from rasterio import features
from rasterio.transform import rowcol
from scipy import ndimage

BASE = Path(__file__).resolve().parent
EVID = BASE / "evidence"
RASTER = Path("/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif")
GROUND = math.cos(math.radians(40.71))


def main():
    ink = np.load("/tmp/fire_ink.npy"); plate = np.load("/tmp/fire_plate.npy")
    with rasterio.open(RASTER) as ds:
        T = ds.transform; px_m = T.a; shape = (ds.height, ds.width)
        gray = ds.read().mean(axis=0).astype(np.uint8)
    px_area_ground = (px_m * GROUND) ** 2
    # stroke half-width from the ink distance transform
    inner = ndimage.distance_transform_edt(ink)
    half_stroke = float(np.percentile(inner[ink & (inner > 0)], 90))
    print("stroke half-width px (p90 of inner EDT)", half_stroke)
    del inner
    dist_to_ink = ndimage.distance_transform_edt(~ink).astype(np.float32)
    # white cells: 8–1500 ground m² AND at least ~2.9 m wide (inner EDT max >= 5 px) so the thin
    # white strips between the burnt-district hatching are not mistaken for lots.
    white = plate & ~ink
    lab, n = ndimage.label(white)
    sizes = np.bincount(lab.ravel())
    cell_ids = np.where((sizes * px_area_ground >= 8) & (sizes * px_area_ground <= 1500))[0]
    cell_ids = cell_ids[cell_ids != 0]
    width_ok = ndimage.maximum(dist_to_ink, lab, index=cell_ids) >= 5.0
    n_thin = int((~width_ok).sum())
    cell_ids = cell_ids[width_ok]
    is_cell = np.zeros(len(sizes), bool); is_cell[cell_ids] = True
    cells = np.where(is_cell[lab], lab, 0).astype(np.int32)
    cell_areas = sizes[cell_ids] * px_area_ground
    print("white comps", n, "cells", len(cell_ids), "thin strips dropped", n_thin, "cell area ground m2 median", float(np.median(cell_areas)))
    # vectorise the cells once (as polygons) for IoU; dilate later by half stroke via buffer
    cell_polys = {}
    for geom, val in features.shapes(cells, mask=cells > 0, transform=T):
        v = int(val)
        g = shapely.geometry.shape(geom)
        cell_polys[v] = g if v not in cell_polys else cell_polys[v].union(g)
    cell_keys = np.array(list(cell_polys.keys()))
    cell_geoms = np.array([cell_polys[k] for k in cell_keys], dtype=object)
    cell_dil = shapely.buffer(cell_geoms, half_stroke * px_m, join_style="mitre")
    tree = shapely.STRtree(cell_geoms)
    b = gpd.read_file(BASE / "1776_nyc_parcels_buildings.gpkg")
    l = gpd.read_file(BASE / "1776_nyc_parcels_landmarks.gpkg")
    r = gpd.read_file(BASE / "1776_nyc_vector_road_graph.gpkg")
    fire = shapely.from_wkb(bytes.fromhex(pd.read_csv(BASE / "export" / "1776_nyc_vector_great_fire.csv", dtype=str).geom.iloc[0]))
    align_b = pd.read_csv(BASE / "qa_buildings.csv").set_index("id")
    align_l = pd.read_csv(BASE / "qa_landmarks.csv").set_index("id")

    def cell_scores(g, align):
        rows = []
        for pk, geom in zip(g.id, g.geometry):
            geom = shapely.make_valid(geom)
            rec = align.loc[int(pk)]
            if not rec.on_plate:
                rows.append(dict(id=int(pk), on_plate=False)); continue
            idx = tree.query(geom, predicate="intersects")
            best = 0.0; best_cell = None; spanned = []; 
            for j in idx:
                c = cell_geoms[j]; cd = cell_dil[j]
                ia = geom.intersection(c).area
                if ia / c.area >= 0.2:
                    spanned.append(j)
                inter = geom.intersection(cd).area
                iou = inter / (geom.area + cd.area - inter)
                if iou > best:
                    best, best_cell = iou, int(cell_keys[j])
            if spanned:
                u = shapely.union_all(cell_dil[spanned]); inter = geom.intersection(u).area
                uiou = inter / (geom.area + u.area - inter)
            else:
                uiou = 0.0
            blank = bool(rec.near6px == 0 and len(idx) == 0)
            rows.append(dict(id=int(pk), on_plate=True, best_cell_iou=best, union_cell_iou=uiou,
                             cells_spanned=len(spanned), blank_plate=blank, best_cell=best_cell,
                             in_fire_zone=bool(fire.intersects(geom.centroid)),
                             area_ground_m2=float(geom.area * GROUND ** 2)))
        df = pd.DataFrame(rows)
        on = df.on_plate
        cls = np.full(len(df), "off_plate", dtype=object)
        cls[on.values] = "loose_placement"
        cls[(on & (df.best_cell_iou >= 0.5)).values] = "traced_single_cell"
        cls[(on & (df.best_cell_iou < 0.5) & (df.union_cell_iou >= 0.5) & (df.cells_spanned >= 2)).values] = "traced_merged_cells"
        cls[(on & df.blank_plate.fillna(False)).values] = "blank_plate"
        df["cell_class"] = cls
        return df

    out = {"stroke_half_width_px": half_stroke, "n_drawn_cells_on_plate": int(len(cell_keys)),
           "thin_hatch_strips_dropped": n_thin,
           "drawn_cell_area_ground_m2": {"p10": float(np.percentile(cell_areas, 10)), "median": float(np.median(cell_areas)),
                                         "p90": float(np.percentile(cell_areas, 90))}}
    dfs = {}
    for name, g, al in (("buildings", b, align_b), ("landmarks", l, align_l)):
        df = cell_scores(g, al); dfs[name] = df
        df.to_csv(BASE / f"qa_cells_{name}.csv", index=False)
        on = df[df.on_plate]
        out[name] = {
            "n": int(len(df)),
            "best_cell_iou_median": float(on.best_cell_iou.median()),
            "best_cell_iou_ge_0_5": int((on.best_cell_iou >= 0.5).sum()),
            "best_cell_iou_ge_0_7": int((on.best_cell_iou >= 0.7).sum()),
            "union_cell_iou_median": float(on.union_cell_iou.median()),
            "cells_spanned_distribution": on.cells_spanned.clip(upper=5).value_counts().sort_index().to_dict(),
            "blank_plate": int(on.blank_plate.sum()),
            "in_fire_zone": int(on.in_fire_zone.astype(bool).sum()),
            "best_cell_iou_median_fire_zone": float(on[on.in_fire_zone.astype(bool)].best_cell_iou.median()) if on.in_fire_zone.any() else None,
            "best_cell_iou_median_outside_fire_zone": float(on[~on.in_fire_zone.astype(bool)].best_cell_iou.median()),
            "polygon_area_ground_m2_median": float(on.area_ground_m2.median()),
            "class_counts": df.cell_class.value_counts().to_dict(),
        }
        print(name, json.dumps(out[name]))

    # ---------------- block coverage ----------------
    lines = shapely.force_2d(r.geometry.values)
    noded = shapely.node(shapely.union_all(lines))
    blocks = list(shapely.get_parts(shapely.polygonize([noded])))
    blocks = [p for p in blocks if p.area * GROUND ** 2 > 200]
    print("blocks", len(blocks))
    bg = gpd.GeoDataFrame({"block_id": range(len(blocks))}, geometry=blocks, crs=3857)
    polys_all = pd.concat([b[["geometry"]], l[["geometry"]]], ignore_index=True)
    polys_all["geometry"] = shapely.make_valid(polys_all.geometry.values)
    ptree = shapely.STRtree(polys_all.geometry.values)
    rows = []
    corridor = 2.0 * half_stroke * px_m  # ignore cells hugging the block edge (street-line artefacts)
    for bid, blk in zip(bg.block_id, bg.geometry):
        rr, cc = rowcol(T, [blk.bounds[0], blk.bounds[2]], [blk.bounds[3], blk.bounds[1]])
        r0, r1 = int(np.clip(rr[0], 0, shape[0])), int(np.clip(rr[1] + 1, 0, shape[0]))
        c0, c1 = int(np.clip(cc[0], 0, shape[1])), int(np.clip(cc[1] + 1, 0, shape[1]))
        if r1 <= r0 or c1 <= c0:
            rows.append(dict(block_id=bid, cls="off_plate", n_cells=0, n_covered=0, n_polys=0, area_ground_m2=blk.area * GROUND ** 2)); continue
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        wt = rasterio.windows.transform(win, T)
        bmask = features.rasterize([(blk, 1)], out_shape=(r1 - r0, c1 - c0), transform=wt, fill=0, dtype=np.uint8).astype(bool)
        pl = plate[r0:r1, c0:c1]
        on_frac = float((pl & bmask).sum() / max(bmask.sum(), 1))
        inner = features.rasterize([(blk.buffer(-corridor), 1)], out_shape=bmask.shape, transform=wt, fill=0, dtype=np.uint8).astype(bool)
        cl = cells[r0:r1, c0:c1]
        ids_in = np.unique(cl[inner & (cl > 0)])
        n_cells = len(ids_in)
        pidx = ptree.query(blk, predicate="intersects")
        n_polys = int(len(pidx))
        n_cov = 0
        if n_cells:
            pmask = np.zeros(bmask.shape, bool)
            if n_polys:
                pmask = features.rasterize([(polys_all.geometry.values[j], 1) for j in pidx], out_shape=bmask.shape, transform=wt, fill=0, dtype=np.uint8).astype(bool)
            sub = np.where(np.isin(cl, ids_in), cl, 0)
            tot = np.bincount(sub.ravel(), minlength=sub.max() + 1)
            cov = np.bincount(sub[pmask].ravel(), minlength=sub.max() + 1)
            frac = cov[ids_in] / tot[ids_in]
            n_cov = int((frac >= 0.5).sum())
        if on_frac < 0.5:
            cls = "off_plate"
        elif n_cells == 0:
            cls = "no_drawn_cells"
        else:
            f_ = n_cov / n_cells
            cls = "exhaustive" if (f_ >= 0.9 and n_cells >= 3) else "partial" if f_ >= 0.3 else "sparse"
        rows.append(dict(block_id=bid, cls=cls, n_cells=n_cells, n_covered=n_cov, n_polys=n_polys,
                         covered_frac=(n_cov / n_cells if n_cells else None), area_ground_m2=blk.area * GROUND ** 2,
                         fire_zone_frac=float(blk.intersection(fire).area / blk.area)))
    cov = pd.DataFrame(rows)
    bg = bg.merge(cov, on="block_id")
    bg.to_file(BASE / "block_coverage.gpkg", driver="GPKG")
    cov.to_csv(BASE / "block_coverage.csv", index=False)
    # polygons inside exhaustive blocks
    exh = bg[bg.cls == "exhaustive"]
    etree = shapely.STRtree(exh.geometry.values)
    b_cent = b.geometry.centroid
    in_exh = np.array([len(etree.query(p, predicate="intersects")) > 0 for p in b_cent])
    summary = {
        "n_blocks": int(len(bg)),
        "class_counts": cov.cls.value_counts().to_dict(),
        "cells_total": int(cov.n_cells.sum()), "cells_covered": int(cov.n_covered.sum()),
        "cells_covered_frac_overall": float(cov.n_covered.sum() / max(cov.n_cells.sum(), 1)),
        "exhaustive_block_area_ground_m2": float(exh.area_ground_m2.sum()),
        "buildings_with_centroid_in_exhaustive_blocks": int(in_exh.sum()),
        "buildings_in_partial_blocks": int(sum(len(shapely.STRtree(bg[bg.cls == "partial"].geometry.values).query(p, predicate="intersects")) > 0 for p in b_cent)),
    }
    out["coverage"] = summary
    print("coverage", json.dumps(summary))
    (BASE / "qa_cells.json").write_text(json.dumps(out, indent=2, default=float))

    # ---------------- evidence PNGs ----------------
    f = 6
    small = gray[::f, ::f]
    img = Image.fromarray(small).convert("RGB")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    colors = {"exhaustive": (0, 170, 0, 110), "partial": (255, 170, 0, 110), "sparse": (220, 0, 0, 90),
              "no_drawn_cells": (120, 120, 120, 60), "off_plate": (0, 0, 0, 0)}
    def px(x, y):
        rr, cc = rowcol(T, x, y); return (cc / f, rr / f)
    for blk, c in zip(bg.geometry, bg.cls):
        for p in getattr(blk, "geoms", [blk]):
            d.polygon([px(x, y) for x, y in p.exterior.coords], fill=colors[c], outline=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d2 = ImageDraw.Draw(img)
    for geom in b.geometry:
        for p in geom.geoms:
            d2.polygon([px(x, y) for x, y in p.exterior.coords], outline=(200, 0, 200))
    font = ImageFont.load_default()
    d2.rectangle([4, 4, 330, 110], fill=(255, 255, 255))
    y0 = 10
    for k, v in colors.items():
        if k == "off_plate":
            continue
        d2.rectangle([10, y0, 30, y0 + 14], fill=v[:3]); d2.text((36, y0), f"{k}: {summary['class_counts'].get(k, 0)} blocks", fill=(0, 0, 0), font=font); y0 += 18
    d2.text((36, y0), "magenta = building polygons; block = road-graph face", fill=(0, 0, 0), font=font)
    img.save(EVID / "block_coverage_map.png")

    # cell-class contact sheet for buildings (stratified 64)
    df = dfs["buildings"].set_index("id")
    rng = np.random.default_rng(3)
    onb = b[df.loc[b.id, "on_plate"].values]
    bb = onb.total_bounds; k = 8
    cx = np.digitize(onb.geometry.centroid.x, np.linspace(bb[0], bb[2], k + 1)[1:-1]); cy = np.digitize(onb.geometry.centroid.y, np.linspace(bb[1], bb[3], k + 1)[1:-1])
    picks = []
    for _, idx in pd.Series(cx * k + cy, index=onb.index).groupby(lambda i: (cx * k + cy)[onb.index.get_loc(i)]).groups.items():
        picks.append(rng.choice(list(idx)))
    picks = picks[:64]
    crop = 200
    sheet = Image.new("RGB", (8 * crop, int(math.ceil(len(picks) / 8)) * (crop + 14)), "white")
    for i, ix in enumerate(picks):
        geom = b.geometry[ix]; pk = int(b.id[ix]); c = geom.centroid
        r0, c0 = rowcol(T, c.x, c.y); r0 = int(np.clip(r0 - crop // 2, 0, shape[0] - crop)); c0 = int(np.clip(c0 - crop // 2, 0, shape[1] - crop))
        tile = Image.fromarray(gray[r0:r0 + crop, c0:c0 + crop]).convert("RGB"); dt = ImageDraw.Draw(tile)
        rec = df.loc[pk]
        if rec.best_cell is not None and not (isinstance(rec.best_cell, float) and math.isnan(rec.best_cell)):
            cg = cell_polys[int(rec.best_cell)]
            for p in getattr(cg, "geoms", [cg]):
                dt.polygon([((rowcol(T, x, y)[1] - c0), (rowcol(T, x, y)[0] - r0)) for x, y in p.exterior.coords], outline=(0, 160, 255))
        for p in geom.geoms:
            dt.polygon([((rowcol(T, x, y)[1] - c0), (rowcol(T, x, y)[0] - r0)) for x, y in p.exterior.coords], outline=(255, 0, 0), width=2)
        x = (i % 8) * crop; y = (i // 8) * (crop + 14)
        sheet.paste(tile, (x, y))
        ImageDraw.Draw(sheet).text((x + 2, y + crop), f"{pk} {rec.cell_class[:14]} iou={rec.best_cell_iou:.2f} span={int(rec.cells_spanned)}", fill=(0, 0, 0), font=font)
    sheet.save(EVID / "buildings_cell_match_contact_sheet.png")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    on = dfs["buildings"][dfs["buildings"].on_plate]
    ax[0].hist(on.best_cell_iou, bins=25, color="#c33"); ax[0].axvline(0.5, color="k", ls="--"); ax[0].set_title("buildings: IoU with best drawn lot cell")
    ax[1].bar(*zip(*sorted(on.cells_spanned.clip(upper=6).value_counts().items())), color="#36c"); ax[1].set_title("drawn cells spanned (>=20% each; 6 = 6+)")
    fig.tight_layout(); fig.savefig(EVID / "buildings_cell_iou_hist.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
