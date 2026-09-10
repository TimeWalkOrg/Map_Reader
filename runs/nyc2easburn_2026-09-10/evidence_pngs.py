#!/usr/bin/env python
"""Evidence PNGs: candidates over Easburn v2 (whole plate + closeups incl. the 51-parcel areas) and a NYC-val closeup."""
import os, numpy as np, cv2, rasterio, geopandas as gpd, pandas as pd
from rasterio.windows import Window
from rasterio import features
from shapely.geometry import box

RUN = os.path.dirname(os.path.abspath(__file__)); TR = f"{RUN}/../../training"
EAS = "/Users/gabriel/.openclaw/workspace-timewalker/philly_georef_work/tw_1776_philadelphia_map_easburn_plan_v2_cog.tif"
NYC = "/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif"
RED, GREEN, BLUE, MAG, YEL = (0, 0, 255), (0, 190, 0), (255, 80, 0), (255, 0, 255), (0, 200, 255)


def render(src, bounds, layers, out, scale=1.0, max_px=1800, title=None):
    """layers: list of (GeoDataFrame, colour, width_m, fill_alpha)"""
    with rasterio.open(src) as s:
        r0, c0 = s.index(bounds[0], bounds[3]); r1, c1 = s.index(bounds[2], bounds[1])
        r0, c0 = max(0, r0), max(0, c0); r1, c1 = min(s.height, r1), min(s.width, c1)
        w, h = c1 - c0, r1 - r0; f = min(1.0, max_px / max(w, h)) * scale
        ow, oh = int(w * f), int(h * f)
        a = s.read([1, 2, 3], window=Window(c0, r0, w, h), out_shape=(3, oh, ow))
        tr = s.window_transform(Window(c0, r0, w, h)) * rasterio.Affine.scale(w / ow, h / oh)
    img = cv2.cvtColor(np.moveaxis(a, 0, -1), cv2.COLOR_RGB2BGR).copy()
    px_m = tr.a
    for gdf, col, wm, alpha in layers:
        g = gdf[gdf.intersects(box(*bounds))]
        if len(g) == 0: continue
        if alpha > 0:
            m = features.rasterize([(geom, 1) for geom in g.geometry.values], out_shape=img.shape[:2], transform=tr, dtype=np.uint8)
            img[m == 1] = (img[m == 1] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
        wpx = max(1.0, wm / px_m)
        lines = g.geometry.boundary if g.geom_type.isin(["Polygon", "MultiPolygon"]).all() else g.geometry
        m = features.rasterize([(geom.buffer(wpx * px_m / 2), 1) for geom in lines.values if not geom.is_empty], out_shape=img.shape[:2], transform=tr, dtype=np.uint8)
        img[m == 1] = col
    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (255, 255, 255), -1)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out, img); print(out, img.shape)


cb = gpd.read_file(f"{RUN}/easburn_building_candidates.gpkg"); cr = gpd.read_file(f"{RUN}/easburn_road_candidates.gpkg")
ref = gpd.read_file(f"{TR}/sunil_parcels_2026-09-07/1776_philadelphia_building_parcels.gpkg").to_crs(3857)
drawn = gpd.read_file(f"{RUN}/../../geo-sam/features_easburn_sam_b/drawn_city.geojson").to_crs(3857)
rg = f"{RUN}/scratch/1776_philadelphia_road_graph.gpkg"
roads_ref = gpd.read_file(rg) if os.path.exists(rg) else None
E = "evidence"
os.makedirs(f"{RUN}/{E}", exist_ok=True)
whole = list(drawn.total_bounds); whole = [whole[0] - 100, whole[1] - 100, whole[2] + 100, whole[3] + 100]
render(EAS, whole, [(cb, GREEN, 3, 0.35), (cr, BLUE, 4, 0), (ref, RED, 6, 0)], f"{RUN}/{E}/easburn_whole_plate_candidates.png", max_px=2400,
       title="Easburn 1776 v2: machine candidates (green=buildings, blue=road centerlines) vs Sunil's 51 parcels (red) - NOT accepted geometry")
# closeups: Market St / High St block cluster (parcels 0-11), Front St waterfront cluster, and a Dock Creek area with road graph
clusters = {"closeup_market_second": ref.iloc[:12].total_bounds, "closeup_front_street_north": ref.iloc[12:30].total_bounds,
            "closeup_south_parcels": ref.iloc[30:].total_bounds}
for name, b in clusters.items():
    pad = 60; bb = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]
    layers = [(cb, GREEN, 1.0, 0.3), (cr, BLUE, 1.5, 0), (ref, RED, 1.2, 0)]
    if roads_ref is not None: layers.insert(0, (roads_ref, YEL, 1.0, 0))
    render(EAS, bb, layers, f"{RUN}/{E}/easburn_{name}.png", max_px=1600,
           title=f"{name}: green=bldg cand, blue=road cand, red=Sunil parcels" + (", yellow=PostGIS road graph" if roads_ref is not None else ""))
# NYC val closeup: pick a val block cluster with buildings
blocks = gpd.read_file(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/block_coverage.gpkg").to_crs(3857)
vb = blocks[(blocks.split == "val") & (blocks.n_polys > 8)].sort_values("n_polys", ascending=False).iloc[:6]
b = vb.total_bounds; bb = [b[0] - 30, b[1] - 30, b[2] + 30, b[3] + 30]
nb = gpd.read_file(f"{RUN}/nyc_fullplate_building_candidates.gpkg"); nr = gpd.read_file(f"{RUN}/nyc_fullplate_road_candidates.gpkg")
bl = gpd.read_file(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/1776_nyc_parcels_buildings.gpkg").to_crs(3857)
rgN = gpd.read_file(f"{TR}/sunil_nyc_1776_fire_map_2026-09-10/1776_nyc_vector_road_graph.gpkg").to_crs(3857); rgN = rgN[rgN.rulefile == "roads.cga"]; rgN["geometry"] = rgN.geometry.force_2d()
render(NYC, bb, [(rgN, YEL, 0.8, 0), (nb, GREEN, 0.8, 0.3), (nr, BLUE, 1.2, 0), (bl, MAG, 0.8, 0)], f"{RUN}/{E}/nyc_val_closeup.png", max_px=1600,
       title="NYC fire map VAL blocks: green=bldg cand, blue=road cand, magenta=Sunil weak polygons, yellow=road graph")
