# Map_Reader

TimeWalk toolset for turning **georeferenced historical maps (COGs)** into
**building footprints in PostGIS**, using SAM (Segment Anything) to accelerate
tracing. Walkthrough site: **https://map-reader.timewalk.live**

## Why

TimeWalk rebuilds historical cities (Manhattan 1776, Philadelphia 1776, Boston)
in Unreal Engine. Footprints come from period maps georeferenced into
EPSG:3857 COGs (see the `NYC_Maps` repo) and hand-traced into the `timewalk`
PostGIS schema. Hand-tracing is accurate but slow. SAM can propose the outline
from a click or box; a human verifies. This repo collects the tooling, a
scored pilot, and the reproducible workflow.

## The end-to-end workflow

1. **Georeference** the map to EPSG:3857 COG (standard per Sunil 2026-08-19:
   30+ label-verified GCPs, TPS warp, OSM road-overlay verification). COGs
   live in `TimeWalk/NYC_Maps` (Git LFS), never here.
2. **Verify the map actually shows building footprints** before extracting
   (lesson learned from the 1777 Pelham Boston map, which has none).
3. **SAM-assisted trace** — either:
   - *Interactive:* QGIS + [Geo-SAM plugin](https://github.com/coolzhao/Geo-SAM)
     (encode once, then <1 s click-to-polygon), or
   - *Scripted:* [samgeo](https://samgeo.gishub.org) as in `pilot/run_sam.py`.
4. **Cleanup + geometry policy** (Sunil, 2026-08-19, FINAL):
   - **Demolished buildings:** trace the drawn ink on the aligned period
     raster (full plot for burial grounds/yards).
   - **Surviving buildings:** use the true modern **OSM footprint**, even if
     the period ink disagrees (plate warp / schematic drawing).
5. **Load to PostGIS** into a candidates table, review in QGIS vs the raster,
   then merge approved rows into the era parcel table:
   ```bash
   ogr2ogr -f PostgreSQL \
     PG:"host=<host> port=5432 dbname=postgres user=postgres sslmode=require" \
     pilot/results/sam_output.gpkg sam_footprints \
     -nln timewalk.sam_footprint_candidates \
     -nlt MULTIPOLYGON -t_srs EPSG:3857 -lco GEOMETRY_NAME=geom
   ```

## Install (exact commands used, Mac mini, 2026-08-21)

```bash
git clone https://k7oth9.gitea.cloud/TimeWalk/Map_Reader.git
cd Map_Reader
python3 -m venv env                    # Python 3.11.3
env/bin/pip install segment-geospatial torch torchvision
# GOTCHA (x86-64 Mac): torch 2.2.2 is the last Intel-mac build and is
# incompatible with numpy 2.x ("Could not infer dtype of numpy.uint8"):
env/bin/pip install "numpy<2" "opencv-python-headless<4.12"
```

Installed versions that work together: `segment-geospatial 1.4.1`,
`torch 2.2.2` (CPU), `numpy 1.26.4`, `rasterio 1.4.4`, `geopandas 1.1.4`,
`opencv-python-headless 4.11.0`. SAM ViT-B checkpoint (375 MB) auto-downloads
to `~/.cache/torch/hub/checkpoints/` on first run.

**QGIS Geo-SAM plugin:** cloned into
`~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/Geo-SAM`
(QGIS 3.40.5 LTR). Before enabling it in QGIS, its deps must be installed into
*QGIS's own* Python:
`/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3 -m pip install torch torchgeo segment-anything rtree`

**GOTCHA:** Homebrew's `gdal_translate` is currently broken on this Mac
(abseil dylib mismatch) — the pilot crops with `rasterio` instead
(`pilot/crop.py`). QGIS also bundles working GDAL binaries.

## Pilot: 1762 Clarkson & Biddle (Philadelphia)

Scored SAM against 4 hand-traced landmarks
(`timewalk."1776_philadelphia_parcels_landmarks"`) on a 2298×1148 px crop of
the city core. Prompts: padded ground-truth bbox + centroid point per building
(simulating a user's rough box + click). Scripts: `pilot/crop.py`,
`pilot/run_sam.py`, `pilot/compare_iou.py`. Full numbers + analysis:
[`pilot/results/results.md`](pilot/results/results.md), overlay evidence:
[`pilot/results/overlay.png`](pilot/results/overlay.png).

| building | IoU |
|---|---|
| Pennsylvania State House / Independence Hall | 0.220 |
| Christ Church | 0.442 |
| High Street Market shambles | 0.081 |
| Old Gaol (Old Stone Prison) | 0.281 |
| **mean** | **0.256** |

**Honest verdict:** too low for unattended auto-tracing — SAM grabs whole
hatched blocks, thin strips punish IoU, and (by policy) surviving-building
ground truth is OSM footprints which *intentionally* disagree with the ink.
But the **speed** result makes interactive use compelling: one ~5 s encode
per window, then **0.06–0.09 s per building prompt** on CPU. That's a
click-to-polygon tracing accelerant, not a replacement for the human.
Upgrade paths: ViT-H, negative points to split blocks, MapSAM-style
fine-tuning on our own traced pairs.

## Related tools

- [Geo-SAM](https://github.com/coolzhao/Geo-SAM) — QGIS interactive SAM
- [samgeo / segment-geospatial](https://samgeo.gishub.org) — scriptable SAM for rasters
- [MapSAM](https://github.com/xue-xia/MapSAM) — SAM fine-tuned for historical maps
- [MapReader](https://github.com/maps-as-data/MapReader) — patch classification at corpus scale
- [mapKurator](https://github.com/machines-reading-maps/map-kurator-system) — map text/label recognition
- [NYPL Map Vectorizer](https://github.com/nypl-spacetime/map-vectorizer) — legacy CV pipeline (2013)

## Repo layout

```
pilot/            crop.py, run_sam.py, compare_iou.py, ground_truth.geojson
pilot/results/    results.md, overlay.png, sam_output.gpkg
site/             static walkthrough page → map-reader.timewalk.live
```

Not committed: the COG (lives in NYC_Maps via LFS), model checkpoints, `env/`.
