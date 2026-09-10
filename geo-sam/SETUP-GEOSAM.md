# Geo-SAM click-to-polygon setup — Philadelphia pre-encoded packages

One click inside a hatched building on the plate → a draft polygon in QGIS,
in well under a second. The slow part (SAM image encoding) has already been
done on the TimeWalk Mac mini; each package ships those pre-encoded features
plus everything needed to load them.

**Audience:** Sunil. **Model for BOTH packages:** SAM Base (`sam_b` / vit_b).

| Package | Map (from `NYC_Maps/maps/`, Git LFS) | Chips | Release |
| --- | --- | --- | --- |
| **Easburn 1776 v2 — PRIMARY** | `tw_1776_philadelphia_map_easburn_plan_v2_cog.tif` (EPSG:3857, 8864×14614 px) | **213** × 1024 px, stride 512 | [`geosam-easburn-1776-v1`](https://k7oth9.gitea.cloud/TimeWalk/NYC_Maps/releases/tag/geosam-easburn-1776-v1) |
| Clarkson & Biddle 1762 v2 — legacy | `tw_1762_philadelphia_map_clarkson_biddle_v2_cog.tif` (39-GCP TPS, EPSG:3857) | 156 × 1024 px, stride 512 | [`geosam-1762-v1`](https://k7oth9.gitea.cloud/TimeWalk/NYC_Maps/releases/tag/geosam-1762-v1) |

**Coverage (both):** the full drawn city — Northern Liberties, city core,
Southwark down to the Fort, including the wharf line. Skipped: river,
Windmill Island, the Delaware Bay inset (Easburn), legend/cartouche/insets
(C&B), blank margins. The Easburn chip grid is its own (`chip_grid.geojson`
inside the zip); nothing is shared with the C&B package — never point one
package at the other raster.

Per the 2026-09-07 decision, **Easburn is the primary map and the raster
Sunil's manual parcels are drawn on**; use the Easburn package for all new
work. The 1762 package stays available for comparisons.

---

## 1. Install the Geo-SAM plugin (v2.0)

Geo-SAM is in the **official QGIS plugin repository** — no manual clone.

1. QGIS ≥ 3.20 (tested 3.40 LTR and Sunil's 3.44.11) → **Plugins → Manage and
   Install Plugins**.
2. Search **"Geo SAM"** → Install. (Plugin shows as "GeoSAM" v2; it also
   knows about sam2.1 models — ignore those, see §3.)

## 2. Install its Python dependencies (built-in installer)

1. **Plugins → Geo-SAM Tools → Geo-SAM Settings → Dependencies tab.**
2. Click **Install Missing** (torch, ultralytics, geosam, etc. go into a
   plugin-private folder — nothing touches the QGIS Python). First install
   downloads several hundred MB.
3. **Restart QGIS** when it finishes.

## 3. Get the SAM Base model — NOT sam2.1_s

Both packages were encoded with **SAM Base** (`sam_b`, vit_b). The plugin
locks the model to whatever the manifest records; any other model will refuse
the features.

- **Geo-SAM Settings → Model Management → download "SAM Base"** (~375 MB), *or*
- let the Segmentation tool prompt you (next step).

> ⚠️ **Gotcha (hit 2026-09-03):** when you open the Segmentation tool it
> defaults the model selector to **`sam2.1_s`** and offers to download it.
> Click **No** and pick **SAM Base** in the dropdown. The model is actually
> loaded from the **Segmentation tool's model dropdown**, not from Model
> Management (Model Management only downloads files).

## 4. Unpack the features and fix paths (one-time per package)

1. Unzip the package anywhere, e.g. `D:/TimeWalk/geosam/`. You get:

   ```
   tw_1776_philadelphia_map_easburn_plan_v2_cog/      ← Easburn package
   ├── manifest.parquet
   ├── chip_grid.geojson, drawn_city.geojson, encode_easburn_run.json  (provenance)
   └── features/
       ├── chip_000000.pt … chip_000212.pt   (213 files, ~4.2 MB each)

   tw_1762_philadelphia_map_clarkson_biddle_v2_cog/   ← 1762 package
   ├── manifest.parquet
   └── features/  chip_000000.pt … chip_000155.pt   (156 files)
   ```

2. `manifest.parquet` stores **absolute paths from the encoding machine**, so
   run the included `fix_manifest_paths.py` once per package. In QGIS
   **Plugins → Python Console**, paste this — it opens two pickers, so there
   is nothing to edit by hand:

   ```python
   from qgis.PyQt.QtWidgets import QFileDialog
   FEATURE_DIR = QFileDialog.getExistingDirectory(None, "Pick the folder that contains manifest.parquet")
   FIXER = QFileDialog.getOpenFileName(None, "Pick fix_manifest_paths.py", "", "Python (*.py)")[0]
   exec(open(FIXER).read())
   ```

   Expected output: `OK: rewrote 213 feature paths …` (Easburn) or
   `… 156 feature paths …` (1762). Any Python with geopandas + pyarrow also
   works: `python fix_manifest_paths.py <folder-with-manifest.parquet>`.

   > ⚠️ Do **not** paste `C:/path/to/...` placeholders literally (happened
   > twice on 2026-09-03) — use the picker snippet above.

## 5. Load features + raster — ORDER MATTERS

1. Add the matching COG raster to your project (Easburn:
   `NYC_Maps/maps/tw_1776_philadelphia_map_easburn_plan_v2_cog.tif`; it is
   already in the PostGIS project).
2. Open the **Geo-SAM Segmentation** tool (toolbar icon).
3. **Model dropdown → SAM Base** (click *No* to the sam2.1_s download offer).
4. **Input/Output tab → source selector: switch "Live Encoding" → "Pre-encoded".**
5. Pick the feature **FOLDER** → **Load**.

   > ⚠️ The picker is a **folder** picker, not a file picker. Select the
   > `tw_1776_..._v2_cog` folder that *contains* `manifest.parquet` — **not**
   > the `features/` subfolder (hit twice on 2026-09-03).

   You should see a success message: **213 chips / SAM Base** (Easburn) or
   156 chips (1762).
6. Only now click **Load/Create** for the Segmentation Result layer and set
   the **output file** (see §7).

   > ⚠️ Plugin bug: creating the result layer *before* features are loaded
   > throws `AttributeError: 'Selector' has no 'img_crs_manager'`
   > (widgetTool.py:2564). Sequence is always **model → features → result
   > layer**. If you hit it, close the tool, reopen, redo in order.
7. **Zoom to** the layer if you're not already on Philadelphia.

## 6. Click workflow

- **Left-click inside a hatched building** (foreground point) → draft polygon
  appears instantly.
- Polygon grabbed too much (merged row / street)? **Add background points**
  (negative clicks) on the parts to exclude, or draw a **bounding box** to
  constrain it. A box across a single house plus a background point on each
  neighbor splits dense party-wall rows well.
- **`S`** = save the current polygon, **`C`** = clear prompts and move on.
- **Preview mode** makes the proposal follow your cursor before you commit.
- Expect SAM edges to hug the ink contour, not idealized corners; per the
  2026-08-19 geometry policy, drawn ink is exactly what we trace for
  demolished buildings (surviving buildings get OSM footprints later anyway).

## 7. Saving results (`source='geosam'`)

**2026-09-07 update:** Sunil's manual `1776_philadelphia_building_parcels`
(51 parcels, now in PostGIS as `timewalk."1776_philadelphia_building_parcels"`)
is the selected training source. Geo-SAM output is **excluded** from training
— save Geo-SAM results separately for review, not as training labels. See
[training selection](../training/README.md).

Option A — let the plugin write its shapefile, then append to a GPKG:

```bash
ogr2ogr -f GPKG -append -nln footprints_geosam -nlt MULTIPOLYGON \
  philly_1776_geosam.gpkg geosam_output.shp \
  -sql "SELECT *, 'geosam' AS source, 'tw_1776_easburn_plan_v2' AS source_map FROM geosam_output"
```

Option B — pre-make the GPKG layer with defaulted fields and point the plugin
output at it: layer `footprints_geosam`, fields `source` (text, default
`'geosam'`), `source_map` (text, default `'tw_1776_easburn_plan_v2'` or
`'tw_1762_clarkson_biddle_v2'`), `reviewed` (bool, default false).

Rejected proposals are not automatically negative training examples. Do not
add this layer to the selected manual-label dataset. Reviewed batches load to
PostGIS the usual way (`timewalk.sam_footprint_candidates`, see
`Map_Reader/README.md`).

## 8. Outside the encoded area / your RTX 5090

Pre-encoded coverage is the drawn city only. For the inset, margins, or
another sheet, switch the source selector back to **Live Encoding** — on the
5090 an on-the-fly encode is a few seconds per view and you don't need these
packages at all. Pre-encoded mode is mainly for CPU-only machines and instant
startup.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Model selector shows `sam2.1_s` / asks to download it | Click No; choose **SAM Base** in the Segmentation tool dropdown. |
| "No cached chip fully covers the requested query" | You clicked outside the encoded area (river/inset/margin) — Live Encoding mode, or ask Gabriel to encode more area. |
| Load fails / missing chip files | Re-run §4's path fixer; check all `.pt` files unzipped (213 Easburn / 156 C&B); make sure you picked the folder with `manifest.parquet`, not `features/`. |
| `AttributeError: 'Selector' has no 'img_crs_manager'` | Result layer created before features loaded — redo in order model → features → result layer. |
| Features load but polygons land in the wrong place | Wrong raster for the package (Easburn features on the C&B raster or vice versa). Each package is pinned to its own COG. |
| Deps install fails behind proxy | Settings → Dependencies → Open Folder shows the target dir; a plain `pip install --target <dir> torch ultralytics geosam` also works. |

## Provenance

**Easburn 1776 v2 (`geosam-easburn-1776-v1`)**
- Encoded 2026-09-10 on the TimeWalk Mac mini, CPU, `encode_easburn.py` in
  this folder — replicates the plugin's Image Encoder algorithm 1:1 via the
  `geosam` 0.1.3 library (torch 2.2.2; checkpoint `sam_b.pt` = official
  `sam_vit_b_01ec64.pth`, sha256 `ec2df627…8912`).
- Source COG sha256 `7f2d55093bc871e547014cd71b2e2d046505274521b7d21abaa8fec0ac1ae99c`
  (byte-identical to `NYC_Maps/maps/…easburn_plan_v2_cog.tif`).
- Chip filter: hand-digitized drawn-city polygon (`DRAWN_CITY_PX` in the
  script; `drawn_city.geojson` in the zip): 476 grid chips → 213 encoded.
- Verified with `verify_easburn.py`: one positive click at each of Sunil's
  51 manual parcels against the cache — see `verify_result.json` in the
  release notes / repo.

**Clarkson & Biddle 1762 v2 (`geosam-1762-v1`)**
- Encoded 2026-09-01, `encode_1762.py`; chip filter = city-core polygon from
  `Map_Reader/pilot/batch_common.py`; 414 grid chips → 156 encoded; ~14.5 s/chip
  CPU; ≈650 MB.
