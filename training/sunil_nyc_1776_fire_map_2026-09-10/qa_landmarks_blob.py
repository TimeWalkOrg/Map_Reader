"""Lighter landmark check: solid-blob agreement.
Landmarks on this plate are solid black shapes. For each landmark polygon compute the ink fraction
inside the polygon, the ink fraction in a 3 m ring just outside it, and the offset between the
polygon centroid and the centroid of the ink inside polygon.buffer(2 m).
Writes qa_landmarks_blob.csv and merges a summary into qa_alignment.json.
"""
import json, math
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd, rasterio, shapely
from rasterio import features
from rasterio.transform import rowcol
BASE = Path(__file__).resolve().parent
RASTER = "/Users/gabriel/NYC_Maps/raster/1776/1776_nyc_map_fire_map_upscaled.tif"
GROUND = math.cos(math.radians(40.71))
ink = np.load("/tmp/fire_ink.npy"); plate = np.load("/tmp/fire_plate.npy")
with rasterio.open(RASTER) as ds:
    T = ds.transform; px_m = T.a; H, W = ds.height, ds.width
l = gpd.read_file(BASE / "1776_nyc_parcels_landmarks.gpkg")
rows = []
for pk, name, geom in zip(l.id, l.name, l.geometry):
    geom = shapely.make_valid(geom)
    outer = geom.buffer(3.0); ring = outer.difference(geom); buf2 = geom.buffer(2.0)
    rr, cc = rowcol(T, [outer.bounds[0], outer.bounds[2]], [outer.bounds[3], outer.bounds[1]])
    r0, r1 = int(np.clip(rr[0], 0, H)), int(np.clip(rr[1] + 1, 0, H)); c0, c1 = int(np.clip(cc[0], 0, W)), int(np.clip(cc[1] + 1, 0, W))
    if r1 <= r0 or c1 <= c0:
        rows.append(dict(id=int(pk), name=name, on_plate=False)); continue
    wt = rasterio.windows.transform(rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), T)
    shp = (r1 - r0, c1 - c0)
    m_in = features.rasterize([(geom, 1)], out_shape=shp, transform=wt, dtype=np.uint8).astype(bool)
    m_ring = features.rasterize([(ring, 1)], out_shape=shp, transform=wt, dtype=np.uint8).astype(bool)
    m_b2 = features.rasterize([(buf2, 1)], out_shape=shp, transform=wt, dtype=np.uint8).astype(bool)
    pl = plate[r0:r1, c0:c1]; ik = ink[r0:r1, c0:c1]
    if (pl & m_in).sum() < 0.5 * max(m_in.sum(), 1):
        rows.append(dict(id=int(pk), name=name, on_plate=False)); continue
    fin = float(ik[m_in].mean()); fring = float(ik[m_ring].mean()) if m_ring.any() else np.nan
    ys, xs = np.nonzero(ik & m_b2)
    if len(ys):
        cy, cx = ys.mean() + r0 + 0.5, xs.mean() + c0 + 0.5
        X, Y = T * (cx, cy); c = geom.centroid
        off = math.hypot(X - c.x, Y - c.y) * GROUND
    else:
        off = np.nan
    rows.append(dict(id=int(pk), name=name, on_plate=True, inside_ink_frac=fin, ring_ink_frac=fring, contrast=fin - fring,
                     ink_centroid_offset_ground_m=off, area_ground_m2=geom.area * GROUND ** 2))
df = pd.DataFrame(rows)
on = df[df.on_plate].copy()
on["blob_class"] = np.where((on.inside_ink_frac >= 0.5) & (on.contrast >= 0.25), "on_solid_blob",
                    np.where(on.inside_ink_frac >= 0.3, "partial_blob", "not_on_blob"))
df = df.merge(on[["id", "blob_class"]], on="id", how="left"); df.loc[~df.on_plate, "blob_class"] = "off_plate"
df.to_csv(BASE / "qa_landmarks_blob.csv", index=False)
summ = {"n": int(len(df)), "class_counts": df.blob_class.value_counts().to_dict(),
        "inside_ink_frac_median": float(on.inside_ink_frac.median()), "ring_ink_frac_median": float(on.ring_ink_frac.median()),
        "ink_centroid_offset_ground_m_median_on_solid_blob": float(on[on.blob_class == "on_solid_blob"].ink_centroid_offset_ground_m.median()),
        "ink_centroid_offset_ground_m_p90_on_solid_blob": float(on[on.blob_class == "on_solid_blob"].ink_centroid_offset_ground_m.quantile(0.9)),
        "not_on_blob_ids": sorted(int(i) for i in df.loc[df.blob_class == "not_on_blob", "id"]),
        "on_solid_blob_ids": sorted(int(i) for i in df.loc[df.blob_class == "on_solid_blob", "id"])}
j = json.loads((BASE / "qa_alignment.json").read_text()); j["landmarks_solid_blob_check"] = summ
(BASE / "qa_alignment.json").write_text(json.dumps(j, indent=2))
print(json.dumps(summ))
print(df[df.on_plate].sort_values("inside_ink_frac")[["id", "name", "inside_ink_frac", "ring_ink_frac", "ink_centroid_offset_ground_m", "blob_class"]].to_string())
