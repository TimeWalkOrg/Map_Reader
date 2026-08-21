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

## Overnight candidate batch (full 1762 map)

Two independent candidate generators run over the whole city-core extent of
the 1762 COG, so review starts from draft polygons instead of a blank canvas
(QGIS: load the GPKG over the COG, fix/delete/accept):

- `pilot/batch_threshold.py` — no-ML NYPL-map-vectorizer-style baseline:
  grayscale ink threshold (<165) + morphological close(3)/open(5),
  polygonized per tile. Seconds to run. Output
  `pilot/results/candidates_threshold.gpkg` (committed): layer `candidates`
  plus layer `oversize_blocks` (contiguous rows/blocks over the 2000 m² cap —
  useful to hand-split). Known noise: street-name lettering.
- `pilot/batch_sam.py` — SAM automatic mask generation (ViT-H, 2.4 GB
  checkpoint), 1024 px-stride tiles (SAM's internal resize is 1024 px, so
  bigger tiles would lose map detail), masks must cover ≥35 % ink, per-tile
  resumable checkpoints in `pilot/results/sam_tiles/`. ~70–90 s/tile on the
  Mac mini CPU. Output `pilot/results/candidates_sam.gpkg` (not committed).

Shared plumbing in `pilot/batch_common.py`: a hand-digitized city-core
polygon (excludes the REFERENCES legend, the two Schuylkill inset maps, the
cartouche, and the river/ships), blank-tile skip by ink density, tile
ownership dedupe (each polygon kept by the tile whose stride cell contains
its representative point; 128 px read margin so boundary buildings are never
cut), and filters: keep 10–2000 m² (EPSG:3857 terms), drop slivers
(min-rotated-rect width < 2 m or aspect > 12).

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

## Tool landscape — six tools evaluated

Six tools were assessed against one question: *can it turn ink on a
georeferenced historical map into building-footprint polygons we can load into
PostGIS?* Two of them (MapReader, mapKurator) answer "no, but they solve an
adjacent problem we also have," and they're included so a reviewer can see the
full option space rather than only the shortlist.

### 1. Geo-SAM — *best immediate win*
<https://github.com/coolzhao/Geo-SAM>

A QGIS plugin (available in the **official QGIS plugin repository**, so
installable from inside QGIS) that runs SAM's image encoder over a raster once —
the slow, offline step — then serves near-instant interactive segmentation from
point and box prompts directly on the QGIS canvas. Prompt response is
sub-second **on CPU**, which our pilot independently confirms (0.06–0.09 s per
prompt). Version 2 adds SAM2/SAM3 backbones. This is the right tool for a human
tracer working building-by-building today: click, get a draft polygon, correct
it, save, next. No GPU, no training data, no annotation effort required.

### 2. MapSAM / MapSAM2 — *path to full automation*
<https://github.com/Xue-Xia/MapSAM> · papers:
[arXiv:2411.06971](https://arxiv.org/abs/2411.06971),
[arXiv:2510.27547](https://arxiv.org/abs/2510.27547)

ETH Zurich research fine-tuning SAM specifically for **historical map** feature
extraction (buildings, vineyards, railways on Siegfried maps), using
parameter-efficient adaptation. The important property for us is that it is
**prompt-free**: after fine-tuning it does batch extraction across a whole sheet
without a human clicking each building. That is the only route on this list to
actual automation rather than acceleration. Cost of entry: it needs a **GPU** and
a corpus of **annotated tiles** in the target map's drawing style. Both are
reachable — PC-5090 has an RTX 5090, and the annotated tiles are exactly the
by-product of doing an interactive Geo-SAM pass. That makes the two tools
sequential rather than competing.

### 3. samgeo / segment-geospatial — *scriptable backbone*
<https://github.com/opengeos/segment-geospatial> · docs <https://samgeo.gishub.org>

Qiusheng Wu's MIT-licensed Python package wrapping SAM for geospatial rasters:
it handles georeferencing, point/box/text prompts, tiling for large images,
batch runs, and vectorization to GeoPackage/Shapefile. Since 2025 it also ships
a QGIS plugin. This is what the pilot in this repo actually runs, and it's the
right layer for anything that must be **reproducible, scriptable, and
CI-able** — evaluation harnesses, bulk re-runs, and the eventual MapSAM
inference wrapper. It is the backbone; Geo-SAM is the cockpit.

### 4. NYPL Map Vectorizer — *historically exact, technically abandoned*
<https://github.com/nypl-spacetime/map-vectorizer>

The closest historical precedent to what TimeWalk is doing: NYPL used this
pipeline to auto-vectorize fire-insurance atlases and produced **170,000+
building footprints** that fed the crowdsourced *Building Inspector* project —
same problem, same human-verification loop we are copying. It is, however,
Python-2 era and effectively abandoned; do not plan to run it as-is. Its
enduring value is methodological: the **color-threshold → `gdal_polygonize`**
approach remains a legitimate **no-ML baseline**, and on maps with clean flat
color fills it can beat SAM outright while being trivially explainable. Worth
keeping as a comparison arm.

### 5. MapReader — *triage, not extraction*
<https://github.com/maps-as-data/MapReader>

Turing Institute library for computational analysis of large map **corpora**. It
cuts sheets into patches and *classifies* them (e.g. "contains buildings,"
"railspace"). It does not produce polygons, so it cannot do our core job. It is
genuinely useful one step earlier: when facing dozens or hundreds of scanned
sheets, it answers "which sheets and which regions are worth georeferencing and
tracing at all" — the triage question that burned us on the 1777 Pelham Boston
map.

### 6. mapKurator — *text, not geometry*
<https://github.com/knowledge-computing/mapkurator-system>

University of Minnesota pipeline for **text spotting** on historical maps —
detecting and recognizing labels at scale (applied to the David Rumsey
collection). Complementary rather than competing: it populates *attributes*
(street names, place names, owner names) for footprints that the SAM tools
produce as *geometry*. A natural companion once the geometry pipeline is
producing volume.

### Ranked verdict

| rank | tool | role | needs |
|---|---|---|---|
| 1 | **Geo-SAM** | best immediate win — interactive tracing in QGIS, today | CPU only |
| 2 | **MapSAM/MapSAM2** | path to full automation — prompt-free batch extraction | GPU + annotated tiles |
| 3 | **samgeo** | scriptable backbone — reproducible runs, evaluation, glue | CPU only |
| — | NYPL vectorizer | no-ML baseline worth comparing against; don't run as-is | legacy |
| — | MapReader | corpus triage: which sheets are worth the effort | — |
| — | mapKurator | attribute enrichment from map labels | — |

**The recommended sequence follows directly from that ranking:** run the
interactive Geo-SAM pass now to get real footprints into PostGIS; keep every
accepted polygon as a training pair; once enough have accumulated, fine-tune
MapSAM on PC-5090's RTX 5090 using those pairs and move to prompt-free batch
extraction; keep samgeo as the scripted layer that runs and scores both.

## Repo layout

```
pilot/            crop.py, run_sam.py, compare_iou.py, ground_truth.geojson
pilot/results/    results.md, overlay.png, sam_output.gpkg
site/             static walkthrough page → map-reader.timewalk.live
```

Not committed: the COG (lives in NYC_Maps via LFS), model checkpoints, `env/`.

## Note for reviewers

This repo is written to be **self-contained and independently reviewable** —
Ted intends to have other AI systems audit the work and the process. Accordingly:

- Every install command in this README is one that was **actually run** on the
  Mac mini on 2026-08-21, including the three failures worth knowing about
  (numpy 2.x vs torch 2.2.2, broken Homebrew GDAL, `rasterio.plot` import).
- The pilot metrics are **unretouched**. Mean IoU 0.256 is a poor score and is
  reported as such, with the overlay PNG included so the numbers can be checked
  against the imagery rather than taken on trust. The box-only run (0.186) is
  reported alongside the box+point run rather than quietly dropped.
- Choices are argued, not asserted: the tool ranking above states what each tool
  needs and why it does or doesn't fit, so a reviewer can disagree with the
  conclusion on the evidence given.
- Reproduction path: clone, run the install block, then
  `pilot/crop.py` → `pilot/run_sam.py` → `pilot/compare_iou.py`. The only input
  not in this repo is the COG (in `TimeWalk/NYC_Maps`, Git LFS) and the ground
  truth, which is regenerated from PostGIS by the query documented in
  `pilot/ground_truth.geojson`'s provenance below.

Ground truth was extracted with:

```sql
SELECT json_build_object('type','FeatureCollection','features', json_agg(
  json_build_object('type','Feature',
    'properties', json_build_object('id', id, 'name', name),
    'geometry', ST_AsGeoJSON(geom)::json)))
FROM timewalk."1776_philadelphia_parcels_landmarks"
WHERE id IN (1,2,13,19);
```
