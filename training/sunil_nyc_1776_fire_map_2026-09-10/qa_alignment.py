"""Alignment QA: do Sunil's 1776 NYC vectors sit on the ink of the fire-map plate?

Raster: NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif (EPSG:3857, 0.289 m/px
in projected units; ground scale ~0.219 m/px at 40.71 N).

Metrics (all polygons, not just a sample — the lookups are cheap):
  * boundary ink agreement  = fraction of densified boundary samples whose distance to the
    nearest ink pixel is <= 3 px (~0.65 m ground) / <= 6 px.
  * shift baseline          = the same score after shifting the polygon 4 m (3857 units) N/S/E/W;
    gain = actual - mean(shifted). Gain near 0 means the polygon is no better than chance on a
    dense plate; strongly positive gain means it was traced on the drawn footprints.
  * median boundary offset  = median distance-to-ink of boundary samples (px and ground m).
  * OSM proxy check         = max IoU against modern OSM building footprints (Overpass snapshot
    in export/); a polygon is flagged modern_proxy when IoU_osm >= 0.5 and ink agreement is poor.
Road graph: for points every 2 m along each edge, scan along the perpendicular for the first ink
pixel on each side; offset = (dR - dL)/2, corridor width = dL + dR.

Outputs: qa_alignment.json, per-feature CSVs, evidence/*.png. Run with Map_Reader/env/bin/python.
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
from rasterio.transform import rowcol
from scipy import ndimage

BASE = Path(__file__).resolve().parent
EVID = BASE / "evidence"
RASTER = Path("/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif")
GROUND_SCALE = math.cos(math.radians(40.71))  # 3857 metres -> ground metres
INK_T = 128
NEAR_PX = 3
FAR_PX = 6
SHIFT_M = 4.0


def load_plate():
    with rasterio.open(RASTER) as ds:
        rgb = ds.read()
        T = ds.transform
        px_m = T.a
    gray = rgb.mean(axis=0).astype(np.uint8)
    del rgb
    ink = gray < INK_T
    # plate mask: the off-plate region is pure black (0) and touches the image border.
    # Block-max downsample -> dark components touching the border = off-plate; the rest is plate.
    f = 8
    small = ndimage.maximum_filter(gray, size=f)[::f, ::f]
    lab, _ = ndimage.label(small < 40)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    big = ~np.isin(lab, list(border))
    big = ndimage.binary_erosion(big, iterations=2)
    plate = np.kron(big, np.ones((f, f), dtype=bool))[: gray.shape[0], : gray.shape[1]]
    if plate.shape != gray.shape:
        pad = np.zeros_like(gray, dtype=bool); pad[: plate.shape[0], : plate.shape[1]] = plate; plate = pad
    ink &= plate
    dist = ndimage.distance_transform_edt(~ink).astype(np.float32)
    return gray, ink, plate, dist, T, px_m


def boundary_samples(geom, step):
    pts = []
    for poly in getattr(geom, "geoms", [geom]):
        rings = [poly.exterior] + list(poly.interiors)
        for r in rings:
            n = max(int(r.length / step), 8)
            pts.append(np.array([r.interpolate(i / n, normalized=True).coords[0] for i in range(n)]))
    return np.vstack(pts)


def lookup(dist, T, xy, plate):
    rows, cols = rowcol(T, xy[:, 0], xy[:, 1])
    rows = np.asarray(rows); cols = np.asarray(cols)
    ok = (rows >= 0) & (rows < dist.shape[0]) & (cols >= 0) & (cols < dist.shape[1])
    d = np.full(len(xy), np.nan, dtype=np.float32)
    d[ok] = dist[rows[ok], cols[ok]]
    on_plate = np.zeros(len(xy), dtype=bool)
    on_plate[ok] = plate[rows[ok], cols[ok]]
    return d, on_plate


def score_polygons(g, dist, plate, T, px_m, osm_tree, osm_geoms):
    rows = []
    step = px_m * 0.5
    for pk, geom in zip(g.iloc[:, 0], g.geometry):
        geom = shapely.make_valid(geom)
        xy = boundary_samples(geom, step)
        d, onp = lookup(dist, T, xy, plate)
        frac_plate = float(onp.mean())
        if frac_plate < 0.5:
            rows.append(dict(id=int(pk), on_plate=False))
            continue
        d = d[onp]
        near = float((d <= NEAR_PX).mean()); far = float((d <= FAR_PX).mean())
        shifted = []
        for dx, dy in ((SHIFT_M, 0), (-SHIFT_M, 0), (0, SHIFT_M), (0, -SHIFT_M)):
            ds_, onp_ = lookup(dist, T, xy + (dx, dy), plate)
            shifted.append(float((ds_[onp_] <= NEAR_PX).mean()) if onp_.any() else np.nan)
        gain = near - float(np.nanmean(shifted))
        iou = 0.0; osm_id = None
        for j in osm_tree.query(geom, predicate="intersects"):
            o = osm_geoms[j]
            inter = geom.intersection(o).area
            v = inter / (geom.area + o.area - inter)
            if v > iou:
                iou, osm_id = v, int(j)
        rows.append(dict(id=int(pk), on_plate=True, near3px=near, near6px=far, gain=gain,
                         median_offset_px=float(np.median(d)), median_offset_m=float(np.median(d) * px_m * GROUND_SCALE),
                         p90_offset_px=float(np.percentile(d, 90)), osm_iou=iou, osm_idx=osm_id,
                         area_m2_ground=float(geom.area * GROUND_SCALE ** 2)))
    return pd.DataFrame(rows)


def classify(df):
    df = df.copy()
    cls = np.where(~df.on_plate, "off_plate", "aligned")
    poor = df.on_plate & ((df.near3px < 0.5) | (df.gain < 0.10))
    cls = np.where(poor, "weak_ink_support", cls)
    cls = np.where(df.on_plate & (df.osm_iou >= 0.5) & poor, "modern_proxy_suspect", cls)
    df["qa_class"] = cls
    return df


def stratified_sample(g, n, seed=0):
    b = g.total_bounds
    k = int(math.ceil(math.sqrt(n)))
    cx = np.digitize(g.geometry.centroid.x, np.linspace(b[0], b[2], k + 1)[1:-1])
    cy = np.digitize(g.geometry.centroid.y, np.linspace(b[1], b[3], k + 1)[1:-1])
    cells = pd.Series(cx * k + cy, index=g.index)
    rng = np.random.default_rng(seed)
    picks = []
    for _, idx in cells.groupby(cells).groups.items():
        picks.append(rng.choice(list(idx)))
    rng.shuffle(picks)
    picks = list(picks[:n])
    rest = [i for i in g.index if i not in set(picks)]
    while len(picks) < n and rest:
        picks.append(rest.pop(rng.integers(len(rest))))
    return g.loc[picks]


def contact_sheet(g, df, gray, T, osm, out, crop=200, cols=8, title=""):
    df = df.set_index("id")
    n = len(g); rows_ = int(math.ceil(n / cols))
    sheet = Image.new("RGB", (cols * crop, rows_ * (crop + 14)), "white")
    font = ImageFont.load_default()
    osm_tree = shapely.STRtree(osm.geometry.values)
    for k, (pk, geom) in enumerate(zip(g.iloc[:, 0], g.geometry)):
        c = geom.centroid
        r0, c0 = rowcol(T, c.x, c.y)
        r0 -= crop // 2; c0 -= crop // 2
        r0 = int(np.clip(r0, 0, gray.shape[0] - crop)); c0 = int(np.clip(c0, 0, gray.shape[1] - crop))
        tile = Image.fromarray(gray[r0:r0 + crop, c0:c0 + crop]).convert("RGB")
        d = ImageDraw.Draw(tile)

        def px(x, y):
            rr, cc = rowcol(T, x, y)
            return (cc - c0, rr - r0)
        for j in osm_tree.query(shapely.box(*shapely.affinity.scale(geom.envelope, 3, 3).bounds), predicate="intersects"):
            o = osm.geometry.values[j]
            for p in getattr(o, "geoms", [o]):
                d.polygon([px(x, y) for x, y in p.exterior.coords], outline=(60, 120, 255))
        for p in getattr(geom, "geoms", [geom]):
            d.polygon([px(x, y) for x, y in p.exterior.coords], outline=(255, 0, 0), width=2)
        rec = df.loc[int(pk)]
        lab = f"{pk} {rec.qa_class[:10]} n3={rec.get('near3px', float('nan')):.2f} g={rec.get('gain', float('nan')):+.2f}"
        x = (k % cols) * crop; y = (k // cols) * (crop + 14)
        sheet.paste(tile, (x, y))
        ImageDraw.Draw(sheet).text((x + 2, y + crop), lab, fill=(0, 0, 0), font=font)
    sheet.save(out)


def road_offsets(roads, ink, plate, T, px_m, max_scan=45):
    H, W = ink.shape
    out = []
    for pk, rule, geom in zip(roads.ogc_fid, roads.rulefile, roads.geometry):
        offs = []; widths = []
        for line in geom.geoms:
            line = shapely.force_2d(line)
            L = line.length
            if L < 4:
                continue
            for s in np.arange(2.0, L - 1.0, 2.0):
                p = line.interpolate(s); q = line.interpolate(min(s + 0.5, L))
                dx, dy = q.x - p.x, q.y - p.y
                nrm = math.hypot(dx, dy)
                if nrm == 0:
                    continue
                nx, ny = -dy / nrm, dx / nrm  # left normal (map units)
                r, c = rowcol(T, p.x, p.y)
                if not (0 <= r < H and 0 <= c < W) or not plate[r, c]:
                    continue
                dl = dr = None
                for k in range(1, max_scan):
                    if dl is None:
                        rr, cc = rowcol(T, p.x + nx * k * px_m, p.y + ny * k * px_m)
                        if 0 <= rr < H and 0 <= cc < W and ink[rr, cc]:
                            dl = k
                    if dr is None:
                        rr, cc = rowcol(T, p.x - nx * k * px_m, p.y - ny * k * px_m)
                        if 0 <= rr < H and 0 <= cc < W and ink[rr, cc]:
                            dr = k
                    if dl is not None and dr is not None:
                        break
                if dl is not None and dr is not None:
                    offs.append((dr - dl) / 2.0); widths.append(dl + dr)
        if offs:
            out.append(dict(ogc_fid=int(pk), rulefile=rule, n=len(offs), median_offset_px=float(np.median(offs)),
                            median_abs_offset_px=float(np.median(np.abs(offs))), median_width_px=float(np.median(widths))))
        else:
            out.append(dict(ogc_fid=int(pk), rulefile=rule, n=0))
    df = pd.DataFrame(out)
    df["median_abs_offset_m"] = df.median_abs_offset_px * px_m * GROUND_SCALE
    df["median_width_m"] = df.median_width_px * px_m * GROUND_SCALE
    return df


def main():
    EVID.mkdir(exist_ok=True)
    gray, ink, plate, dist, T, px_m = load_plate()
    print("plate px", int(plate.sum()), "ink px", int(ink.sum()), "px_m", px_m)
    b = gpd.read_file(BASE / "1776_nyc_parcels_buildings.gpkg")
    l = gpd.read_file(BASE / "1776_nyc_parcels_landmarks.gpkg")
    r = gpd.read_file(BASE / "1776_nyc_vector_road_graph.gpkg")
    osm = gpd.read_file(BASE / "export" / "osm_buildings_2026-09-10.gpkg")
    osm_geoms = osm.geometry.values; osm_tree = shapely.STRtree(osm_geoms)

    res = {"raster": str(RASTER), "pixel_size_3857_m": px_m, "ground_m_per_px": px_m * GROUND_SCALE,
           "near_px": NEAR_PX, "far_px": FAR_PX, "shift_m_3857": SHIFT_M}
    for name, g in (("buildings", b[["id", "geometry"]]), ("landmarks", l[["id", "geometry"]])):
        df = classify(score_polygons(g, dist, plate, T, px_m, osm_tree, osm_geoms))
        df.to_csv(BASE / f"qa_{name}.csv", index=False)
        on = df[df.on_plate]
        summary = {
            "n": int(len(df)), "off_plate": int((~df.on_plate).sum()),
            "near3px_median": float(on.near3px.median()), "near3px_p10": float(on.near3px.quantile(0.1)),
            "near6px_median": float(on.near6px.median()),
            "gain_median": float(on.gain.median()), "gain_p10": float(on.gain.quantile(0.1)),
            "boundary_offset_px_median_of_medians": float(on.median_offset_px.median()),
            "boundary_offset_ground_m_median_of_medians": float(on.median_offset_m.median()),
            "osm_iou_median": float(on.osm_iou.median()), "osm_iou_ge_0_5": int((on.osm_iou >= 0.5).sum()),
            "class_counts": df.qa_class.value_counts().to_dict(),
            "flagged_modern_proxy_ids": sorted(int(i) for i in df.loc[df.qa_class == "modern_proxy_suspect", "id"]),
            "flagged_weak_ink_ids": sorted(int(i) for i in df.loc[df.qa_class == "weak_ink_support", "id"]),
            "off_plate_ids": sorted(int(i) for i in df.loc[~df.on_plate, "id"]),
        }
        res[name] = summary
        print(name, json.dumps({k: v for k, v in summary.items() if not k.endswith("_ids")}, indent=None))
        # histogram
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].hist(on.near3px, bins=25, color="#c33"); ax[0].set_title(f"{name}: boundary within {NEAR_PX}px of ink")
        ax[1].hist(on.gain, bins=25, color="#36c"); ax[1].axvline(0, color="k"); ax[1].set_title("gain vs 4 m shifted baseline")
        ax[2].hist(on.median_offset_m, bins=25, color="#393"); ax[2].set_title("median boundary→ink offset (ground m)")
        fig.tight_layout(); fig.savefig(EVID / f"{name}_alignment_hist.png", dpi=110); plt.close(fig)
        n_s = 64 if name == "buildings" else len(g)
        samp = stratified_sample(g[df.set_index("id").loc[g.id, "on_plate"].values], n_s, seed=1)
        contact_sheet(samp, df, gray, T, osm, EVID / f"{name}_contact_sheet.png", crop=200 if name == "buildings" else 320)
        # also a sheet of the worst-scoring on-plate polygons
        worst_ids = on.sort_values("near3px").id.head(32).tolist()
        contact_sheet(g[g.id.isin(worst_ids)], df, gray, T, osm, EVID / f"{name}_worst32_contact_sheet.png", crop=200)

    rd = road_offsets(r, ink, plate, T, px_m)
    rd.to_csv(BASE / "qa_roads.csv", index=False)
    ok = rd[rd.n > 0]
    roads_summary = {
        "n_edges": int(len(rd)), "edges_measured": int(len(ok)),
        "by_rulefile": rd.groupby("rulefile").agg(n=("ogc_fid", "size"), measured=("n", lambda s: int((s > 0).sum())),
                                                  median_abs_offset_m=("median_abs_offset_m", "median"),
                                                  median_width_m=("median_width_m", "median")).round(2).reset_index().to_dict("records"),
        "median_abs_offset_px": float(ok.median_abs_offset_px.median()),
        "median_abs_offset_ground_m": float(ok.median_abs_offset_m.median()),
        "p90_abs_offset_ground_m": float(ok.median_abs_offset_m.quantile(0.9)),
        "median_corridor_width_ground_m": float(ok.median_width_m.median()),
        "edges_offset_over_2m": int((ok.median_abs_offset_m > 2).sum()),
    }
    res["roads"] = roads_summary
    print("roads", json.dumps({k: v for k, v in roads_summary.items() if k != "by_rulefile"}))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(ok.median_abs_offset_m.clip(upper=6), bins=30, color="#393"); ax[0].set_title("road edge: |centerline offset| from corridor centre (ground m)")
    ax[1].hist(ok.median_width_m.clip(upper=40), bins=30, color="#666"); ax[1].set_title("measured corridor width between ink (ground m)")
    fig.tight_layout(); fig.savefig(EVID / "roads_offset_hist.png", dpi=110); plt.close(fig)
    (BASE / "qa_alignment.json").write_text(json.dumps(res, indent=2))
    np.save("/tmp/fire_ink.npy", ink); np.save("/tmp/fire_plate.npy", plate)


if __name__ == "__main__":
    main()
