#!/usr/bin/env python3
"""Compare SAM footprints vs hand-traced ground truth: per-building IoU
+ an overlay PNG (map crop, ground truth green, SAM red)."""
import json
import geopandas as gpd
import rasterio
import rasterio.plot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import shape

CROP = "crop_1762_core.tif"
GT = "ground_truth.geojson"
SAM = "results/sam_output.gpkg"
PNG = "results/overlay.png"
MD = "results/results.md"

with open(GT) as f:
    fc = json.load(f)
gt = {f["properties"]["id"]: (f["properties"]["name"], shape(f["geometry"]))
      for f in fc["features"]}

sam = gpd.read_file(SAM, layer="sam_footprints")

rows = []
for _, r in sam.iterrows():
    name, g = gt[r["id"]]
    s = r.geometry
    if s is None or s.is_empty:
        rows.append((r["id"], name, 0.0, 0.0, 0.0, r["predict_s"]))
        continue
    inter = g.intersection(s).area
    union = g.union(s).area
    iou = inter / union if union else 0.0
    prec = inter / s.area if s.area else 0.0   # how much of SAM poly is real building
    rec = inter / g.area if g.area else 0.0    # how much of building SAM covered
    rows.append((r["id"], name, iou, prec, rec, r["predict_s"]))

with open(MD, "w") as f:
    f.write("# Pilot results — SAM vs hand-traced (1762 Clarkson & Biddle)\n\n")
    f.write("| id | building | IoU | precision | recall | predict (s) |\n")
    f.write("|---|---|---|---|---|---|\n")
    for fid, name, iou, prec, rec, dt in rows:
        f.write(f"| {fid} | {name} | {iou:.3f} | {prec:.3f} | {rec:.3f} | {dt} |\n")
    mean_iou = np.mean([r[2] for r in rows])
    f.write(f"\nMean IoU: **{mean_iou:.3f}**\n")
print(open(MD).read())

# overlay
with rasterio.open(CROP) as src:
    img = src.read()
    extent = rasterio.plot.plotting_extent(src)
img = np.transpose(img[:3], (1, 2, 0)) if img.shape[0] >= 3 else img[0]

fig, ax = plt.subplots(figsize=(16, 16 * (extent[3]-extent[2]) / (extent[1]-extent[0])))
ax.imshow(img, extent=extent, cmap=None if img.ndim == 3 else "gray")
for fid, (name, g) in gt.items():
    gpd.GeoSeries([g]).plot(ax=ax, facecolor="none", edgecolor="lime", linewidth=2.5)
sam.plot(ax=ax, facecolor="red", alpha=0.35, edgecolor="red", linewidth=1.5)
for fid, name, iou, *_ in rows:
    g = gt[fid][1]
    c = g.centroid
    ax.annotate(f"{name.split('(')[0].strip()}\nIoU {iou:.2f}",
                (c.x, c.y), color="yellow", fontsize=9, ha="center",
                path_effects=None, weight="bold")
ax.legend(handles=[Patch(edgecolor="lime", facecolor="none", label="hand-traced ground truth"),
                   Patch(facecolor="red", alpha=0.35, label="SAM output")],
          loc="lower right")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"wrote {PNG}")
