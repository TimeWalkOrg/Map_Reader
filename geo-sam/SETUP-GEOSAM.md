# Geo-SAM click-to-polygon setup — 1762 Clarkson & Biddle (Philadelphia)

One click inside a hatched building on the 1762 plate → a draft polygon in
QGIS, in well under a second. The slow part (SAM image encoding) has already
been done for you on the Mac mini; this folder ships those pre-encoded
features plus everything you need to load them.

**Audience:** Sunil. **Map:** `tw_1762_philadelphia_map_clarkson_biddle_v2_cog.tif`
(the v2 COG from `NYC_Maps`, 39-GCP TPS georef, EPSG:3857). **Coverage:** the
full built-up strip — city core **plus Southwark and the Northern Liberties**
(156 chips of 1024 px, stride 512; legend, cartouche, insets and river skipped).

---

## 1. Install the Geo-SAM plugin (v2.0)

Geo-SAM is now in the **official QGIS plugin repository** — no manual clone
needed.

1. QGIS ≥ 3.20 (tested on 3.40 LTR) → **Plugins → Manage and Install Plugins**.
2. Search **"Geo SAM"** → Install.

## 2. Install its Python dependencies (built-in installer)

1. **Plugins → Geo-SAM Tools → Geo-SAM Settings → Dependencies tab.**
2. Click **Install Missing** (torch, ultralytics, geosam, etc. go into a
   plugin-private folder — nothing touches the QGIS Python).
   First install downloads several hundred MB; give it a few minutes.
3. **Restart QGIS** when it finishes.

## 3. Download the SAM Base model

1. **Geo-SAM Settings → Model Management tab.**
2. Download **SAM Base** (~375 MB).

> ⚠️ It must be **SAM Base** (`sam_b`) — the features in this package were
> encoded with that model, and the plugin locks the model to whatever the
> manifest records. Other models will not work with these features (but see
> "Live Encoding" below).

## 4. Unpack the features and fix paths (one-time)

1. Unzip `tw_1762_geosam_features_sam_b.zip` anywhere, e.g. `D:/TimeWalk/geosam/`.
   You should get:

   ```
   tw_1762_philadelphia_map_clarkson_biddle_v2_cog/
   ├── manifest.parquet
   └── features/
       ├── chip_000000.pt … chip_000155.pt   (156 files, ~4.2 MB each)
   ```

2. The manifest stores absolute paths from the encoding machine, so run the
   included fixer once. Easiest: **Plugins → Python Console** in QGIS, then:

   ```python
   FEATURE_DIR = r"D:/TimeWalk/geosam/tw_1762_philadelphia_map_clarkson_biddle_v2_cog"
   exec(open(r"D:/TimeWalk/geosam/fix_manifest_paths.py").read())
   ```

   It prints `OK: rewrote 156 feature paths …`. (Any Python with
   geopandas+pyarrow also works: `python fix_manifest_paths.py <folder>`.)

## 5. Load features + raster

1. Add the **1762 v2 COG** raster to your project (you already have it in the
   PostGIS project via `NYC_Maps/maps/tw_1762_philadelphia_map_clarkson_biddle_v2_cog.tif`).
2. Open the **Geo-SAM Segmentation** tool (toolbar icon).
3. **Input/Output tab → source selector: switch "Live Encoding" → "Pre-encoded".**
4. Pick the `tw_1762_philadelphia_map_clarkson_biddle_v2_cog` folder → **Load**.
   You should see a success message with 156 chips / SAM Base.
5. Click **Zoom to** if you're not already on Philadelphia.
6. Set the **output file** in the Input/Output tab (see §7 for the GeoPackage
   convention).

## 6. Click workflow

- **Left-click inside a hatched building** (foreground point) → draft polygon
  appears instantly.
- Polygon grabbed too much (merged row / street)? **Add background points**
  (negative clicks) on the parts to exclude, or draw a **bounding box** to
  constrain it.
- **`S`** = save the current polygon, **`C`** = clear prompts and move on.
- **Preview mode** makes the proposal follow your cursor before you commit.
- Dense party-wall rows often segment as one blob — a box prompt across a
  single house, plus a background point on each neighbor, splits them well.
- Expect SAM edges to hug the ink contour, not idealized corners; per the
  2026-08-19 geometry policy, drawn ink is exactly what we trace for
  demolished buildings (surviving buildings get OSM footprints later anyway).

## 7. Saving to GeoPackage with `source='geosam'`

Save clicks into a GeoPackage so they double as **training labels** for the
future MapSAM fine-tune (every accepted polygon = one training pair).

Option A — let the plugin write its shapefile, then append to a GPKG:

```bash
ogr2ogr -f GPKG -append -nln footprints_geosam -nlt MULTIPOLYGON \
  philly_1762_geosam.gpkg geosam_output.shp \
  -sql "SELECT *, 'geosam' AS source, 'tw_1762_clarkson_biddle_v2' AS source_map FROM geosam_output"
```

Option B — pre-make the GPKG layer with defaulted fields and point the plugin
output at it: layer `footprints_geosam`, fields `source` (text, default
`'geosam'`), `source_map` (text, default `'tw_1762_clarkson_biddle_v2'`),
`reviewed` (bool, default false). Then run QGIS field defaults on save.

Keep everything you click — even rejects are useful negatives for training.
When a batch is reviewed, we load it to PostGIS the usual way
(`timewalk.sam_footprint_candidates`, see `Map_Reader/README.md`).

## 8. Outside the encoded area / your RTX 5090

The pre-encoded coverage is the built strip only. If you need the Schuylkill
insets, map margins, or another sheet entirely, just switch the source
selector back to **Live Encoding** — on your 5090 the on-the-fly encode is a
few seconds per view and you don't need this package at all. Pre-encoded mode
is mainly a gift to CPU-only machines and for instant startup.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "No cached chip fully covers the requested query" | You clicked outside the encoded strip — Live Encoding mode, or ask Gabriel to encode more area. |
| Load fails / missing chip files | Re-run step 4's path fixer; check all 156 `.pt` files unzipped. |
| Model dropdown locked to SAM Base but not downloaded | Settings → Model Management → download SAM Base. |
| Deps install fails behind proxy | Settings → Dependencies → Open Folder shows the target dir; a plain `pip install --target <dir> torch ultralytics geosam` also works. |

## Provenance

- Encoded 2026-09-01 on the TimeWalk Mac mini, CPU (`encode_1762.py` in this
  folder — replicates the plugin's Image Encoder algorithm 1:1 via the
  `geosam` 0.1.3 library; torch 2.2.2, checkpoint `sam_b.pt` = official
  `sam_vit_b_01ec64.pth`).
- Chip filter: hand-digitized city-core polygon from
  `Map_Reader/pilot/batch_common.py` (includes Southwark + Northern
  Liberties), 414 grid chips → 156 encoded.
- ~14.5 s/chip on CPU; full run ≈ 38 min; features ≈ 650 MB.
