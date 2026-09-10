#!/usr/bin/env python
"""Scoring for the NYC->Easburn pass.
 * Easburn buildings: one-to-one matching (Hungarian on IoU) of machine candidates vs Sunil's 51 manual parcels
   (held-out, positives-only). IoU>=0.5. Precision is reported over candidates inside the evaluation envelope only
   (parcels buffered 25 m) and labelled 'indicative' because unlabeled space is not confirmed negative.
 * Roads (Easburn vs PostGIS 1776_philadelphia_road_graph export if present; NYC val vs road graph in val blocks):
   centerline offset (sampled points on predicted lines -> nearest reference line), completeness = share of reference
   length within TOL of a prediction, correctness = share of predicted length within TOL of a reference.
 * Also runs scorer self-tests (exact / duplicate / split / merge / empty / disjoint) before scoring."""
import json, sys, os
import numpy as np, geopandas as gpd, pandas as pd
from shapely.geometry import box, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
from scipy.optimize import linear_sum_assignment

RUN = os.path.dirname(os.path.abspath(__file__))
TR = f"{RUN}/../../training"


def boundary_dist(a, b):
    """mean distance from vertices/sampled points of a's boundary to b's boundary (metres), symmetric average."""
    def side(x, y):
        bx = x.boundary; n = max(8, int(bx.length / 1.0)); pts = [bx.interpolate(i / n, normalized=True) for i in range(n)]
        return float(np.mean([y.boundary.distance(p) for p in pts]))
    return 0.5 * (side(a, b) + side(b, a))


def match_polygons(pred, ref, iou_thr=0.5, merge_frac=0.3):
    """pred, ref: lists of shapely polygons. Returns dict of metrics + per-ref rows."""
    P, R = len(pred), len(ref)
    iou = np.zeros((P, R)); inter_frac_ref = np.zeros((P, R)); inter_frac_pred = np.zeros((P, R))
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if not p.intersects(r): continue
            ia = p.intersection(r).area
            iou[i, j] = ia / (p.area + r.area - ia); inter_frac_ref[i, j] = ia / r.area; inter_frac_pred[i, j] = ia / p.area
    rows = []; tp = 0
    if P and R:
        ri, ci = linear_sum_assignment(-iou)
        matched = {j: i for i, j in zip(ri, ci) if iou[i, j] >= iou_thr}
    else:
        matched = {}
    for j, r in enumerate(ref):
        i = matched.get(j)
        covering = np.where(inter_frac_ref[:, j] >= merge_frac)[0]
        split_flag = len(np.where(inter_frac_pred[:, j] >= 0.5)[0]) >= 2 and inter_frac_ref[:, j].sum() >= 0.5
        merge_flag = False; oversize = False
        if i is not None:
            tp += 1
            merge_flag = (inter_frac_ref[i, :] >= merge_frac).sum() >= 2
            oversize = pred[i].area > 2.0 * r.area
        elif len(covering):
            k = covering[np.argmax(iou[covering, j])]
            merge_flag = (inter_frac_ref[k, :] >= merge_frac).sum() >= 2; oversize = pred[k].area > 2.0 * r.area
        rows.append(dict(ref_idx=j, matched=i is not None, pred_idx=int(i) if i is not None else None,
                         best_iou=float(iou[:, j].max()) if P else 0.0, match_iou=float(iou[i, j]) if i is not None else None,
                         boundary_dist_m=boundary_dist(pred[i], r) if i is not None else None,
                         merge_flag=bool(merge_flag), split_flag=bool(split_flag), oversize_flag=bool(oversize),
                         any_overlap=bool(iou[:, j].max() > 0) if P else False))
    df = pd.DataFrame(rows)
    return dict(n_pred=P, n_ref=R, tp=tp, precision=tp / P if P else None, recall=tp / R if R else None,
                f1=(2 * tp / (P + R)) if (P + R) else None,
                median_match_iou=float(df.match_iou.dropna().median()) if tp else None,
                median_best_iou=float(df.best_iou.median()) if R else None,
                median_boundary_dist_m=float(df.boundary_dist_m.dropna().median()) if tp else None,
                n_merge_flag=int(df.merge_flag.sum()), n_split_flag=int(df.split_flag.sum()), n_oversize_flag=int(df.oversize_flag.sum()),
                n_ref_any_overlap=int(df.any_overlap.sum())), df


def selftest():
    sq = lambda x, y, s=10: box(x, y, x + s, y + s)
    r = [sq(0, 0), sq(20, 0)]
    m, _ = match_polygons([sq(0, 0), sq(20, 0)], r); assert m["tp"] == 2 and m["precision"] == 1 and m["n_merge_flag"] == 0
    m, _ = match_polygons([sq(0, 0), sq(0, 0), sq(20, 0)], r); assert m["tp"] == 2 and abs(m["precision"] - 2 / 3) < 1e-9, m
    m, _ = match_polygons([box(0, 0, 3.5, 10), box(3.5, 0, 6.5, 10), box(6.5, 0, 10, 10)], r); assert m["tp"] == 0 and m["n_split_flag"] == 1, m
    m, _ = match_polygons([box(0, 0, 30, 10)], r); assert m["tp"] == 0 and m["n_merge_flag"] == 2 and m["n_oversize_flag"] == 2, m
    m, _ = match_polygons([], r); assert m["tp"] == 0 and m["recall"] == 0 and m["precision"] is None
    m, _ = match_polygons([sq(100, 100)], r); assert m["tp"] == 0 and m["precision"] == 0
    m, _ = match_polygons([box(1, 1, 11, 11)], [sq(0, 0)]); assert m["tp"] == 1 and m["median_boundary_dist_m"] < 1.5, m
    print("scorer self-tests passed")


def line_metrics(pred, ref, tol=3.0, sample=2.0):
    """pred, ref: GeoSeries of lines (metric CRS)."""
    ru = unary_union(list(ref.geometry.values)); pu = unary_union(list(pred.geometry.values))
    offs = []
    for g in pred.geometry.values:
        n = max(2, int(g.length / sample))
        for i in range(n):
            offs.append(ru.distance(g.interpolate(i / (n - 1) if n > 1 else 0, normalized=True)))
    offs = np.array(offs)
    def cov(lines, other, t):
        tot = sum(g.length for g in lines); inside = sum(g.intersection(other.buffer(t)).length for g in lines); return inside / tot if tot else None
    near = offs[offs <= 15]
    return dict(n_pred=int(len(pred)), pred_km=float(pred.length.sum() / 1000), n_ref=int(len(ref)), ref_km=float(ref.length.sum() / 1000),
                offset_median_m_all_pred_points=float(np.median(offs)), offset_p90_m_all_pred_points=float(np.percentile(offs, 90)),
                offset_median_m_pred_points_within_15m=float(np.median(near)) if len(near) else None,
                frac_pred_points_within_15m=float((offs <= 15).mean()),
                correctness_within_tol={f"{t}m": cov(list(pred.geometry.values), ru, t) for t in (3, 5, 8)},
                completeness_within_tol={f"{t}m": cov(list(ref.geometry.values), pu, t) for t in (3, 5, 8)},
                note="reference graphs omit alleys/minor lanes that the candidates include; correctness is therefore a lower bound")


if __name__ == "__main__":
    selftest()
    out = {}
    # ---- Easburn buildings vs 51 parcels
    ref = gpd.read_file(f"{TR}/sunil_parcels_2026-09-07/1776_philadelphia_building_parcels.gpkg").to_crs(3857)
    ref["geometry"] = ref.geometry.buffer(0)
    refs = [g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area) for g in ref.geometry.values]
    env = unary_union(list(ref.geometry.buffer(25).values))
    for tag, fn in (("hybrid_instances", "easburn_building_candidates.gpkg"), ("unet_builtup_raw", "easburn_builtup_unet_candidates.gpkg")):
        cand = gpd.read_file(f"{RUN}/{fn}").to_crs(3857)
        in_env = cand[cand.geometry.representative_point().within(env) | cand.geometry.intersects(unary_union(list(ref.geometry.values)))]
        m, df = match_polygons(list(in_env.geometry.values), refs)
        m["note"] = "precision indicative only: candidates counted if inside parcels buffered 25 m or touching a parcel; parcels are positives-only, not exhaustive"
        m["n_candidates_whole_plate"] = int(len(cand)); m["n_candidates_oversize_whole_plate"] = int(cand.oversize_flag.sum())
        m["geometry_validity"] = dict(all_valid=bool(cand.is_valid.all()), n_invalid=int((~cand.is_valid).sum()), crs=str(cand.crs))
        m3, _ = match_polygons(list(in_env.geometry.values), refs, iou_thr=0.3); m["at_iou_0.3"] = {k: m3[k] for k in ("tp", "precision", "recall")}
        out[f"easburn_buildings_vs_51_parcels__{tag}"] = m
        df.to_csv(f"{RUN}/easburn_parcel_match__{tag}.csv", index=False)
    # ---- Easburn roads vs PostGIS road graph export (optional)
    rg = f"{RUN}/scratch/1776_philadelphia_road_graph.gpkg"
    if os.path.exists(rg):
        refr = gpd.read_file(rg).to_crs(3857); predr = gpd.read_file(f"{RUN}/easburn_road_candidates.gpkg").to_crs(3857)
        drawn = gpd.read_file(f"{RUN}/../../geo-sam/features_easburn_sam_b/drawn_city.geojson").to_crs(3857).geometry.union_all()
        refr = refr[refr.intersects(drawn)]
        out["easburn_roads_vs_1776_philadelphia_road_graph"] = line_metrics(predr, refr)
        out["easburn_roads_vs_1776_philadelphia_road_graph"]["geometry_validity"] = dict(all_valid=bool(predr.is_valid.all()), crs=str(predr.crs))
    else:
        out["easburn_roads_vs_1776_philadelphia_road_graph"] = "skipped: road graph export not available"
    # ---- NYC val: roads (centerline) inside val blocks; buildings pixel agreement comes from train log
    if os.path.exists(f"{RUN}/nyc_fullplate_road_candidates.gpkg"):
        blocks = gpd.read_file(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/block_coverage.gpkg").to_crs(3857)
        valarea = unary_union(list(blocks[blocks.split == "val"].geometry.buffer(12).values))
        roads = gpd.read_file(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/1776_nyc_vector_road_graph.gpkg").to_crs(3857)
        roads = roads[roads.rulefile == "roads.cga"]; roads["geometry"] = roads.geometry.force_2d()
        qr = pd.read_csv(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/qa_roads.csv").set_index("ogc_fid")
        roads = roads.join(qr[["n", "median_width_m"]], on="ogc_fid")
        roads_drawn = roads[(roads.n >= 3) & (roads.median_width_m >= 2.0)]  # edges with a drawn corridor on the plate
        refv = roads[roads.intersects(valarea)].copy(); refv["geometry"] = refv.geometry.intersection(valarea); refv = refv[~refv.is_empty]
        refd = roads_drawn[roads_drawn.intersects(valarea)].copy(); refd["geometry"] = refd.geometry.intersection(valarea); refd = refd[~refd.is_empty]
        predn = gpd.read_file(f"{RUN}/nyc_fullplate_road_candidates.gpkg").to_crs(3857)
        predv = predn[predn.intersects(valarea)].copy(); predv["geometry"] = predv.geometry.intersection(valarea); predv = predv[~predv.is_empty]
        out["nyc_val_roads_vs_road_graph_all_edges"] = line_metrics(predv, refv)
        out["nyc_val_roads_vs_road_graph_drawn_corridor_edges_only"] = line_metrics(predv, refd)
        # NYC val buildings: candidates vs weak polygons in val blocks (weak labels -> agreement, not accuracy)
        bl = gpd.read_file(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/1776_nyc_parcels_buildings.gpkg").to_crs(3857)
        sp = pd.read_csv(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/proposed_split_buildings.csv").set_index("id")
        bl = bl.join(sp[["split", "cell_class"]], on="id"); blv = bl[(bl.split == "val") & (bl.cell_class != "blank_plate")]
        cb = gpd.read_file(f"{RUN}/nyc_fullplate_building_candidates.gpkg").to_crs(3857)
        cbv = cb[cb.representative_point().within(unary_union(list(blv.geometry.buffer(25).values)))]
        refs_n = [g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area) for g in blv.geometry.values]
        mn, _ = match_polygons(list(cbv.geometry.values), refs_n)
        mn["note"] = "agreement with WEAK val labels (not pixel-traced), candidates inside val polygons buffered 25 m"
        out["nyc_val_buildings_vs_weak_labels"] = mn
    json.dump(out, open(f"{RUN}/metrics.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
