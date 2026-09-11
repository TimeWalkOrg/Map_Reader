# 1762 Society Hill — source-alignment fix and review package

Date: 2026-09-06. Requested by Ted in #ai. **Local candidates only; no PostGIS,
shared QGIS project, or OpenClaw configuration changes.**

## Results

- Re-extracted the full city core from Clarkson–Biddle **v2**, using the v4
  extractor. **726 candidates + 28 oversize review features**, all nonempty,
  valid EPSG:3857 polygons. Runtime: 13.5 seconds on this machine, including
  source hashing. This is extraction time, NOT human cleanup time.
- Ran a same-code **v1 control**, with the same parameters and no carry-over:
  676 candidates + 27 oversize features, 13.6 seconds. Count increases do
  not establish increased accuracy.
- Built three contiguous **sampling areas** between Second and Third Streets:
  B1 Spruce–Pine (includes Union Street internally), B2 Pine–Lombard,
  B3 Lombard–Cedar/modern South. These are street-centerline AOIs, not
  inferred building boundaries. B1 includes smaller historical street blocks.
  The earlier 80–150-building estimate was not borne out in this sample:
  there are **25 candidates**, distributed 13/9/3, and no oversize features.
- Local v1→v2 raster phase-correlation shifts: **4.29 / 5.26 / 4.74 m**
  in UTM zone 18N; correlation responses 0.43/0.48/0.37. Approximate local
  translation diagnostics only: NOT independent georeferencing errors or
  footprint accuracy. The warps are nonuniform; no global vector shift applied.
- Inspected B1/B2 before/after overlays: corrected registration is visibly
  better, but small buildings are still missed and some street lettering is
  still detected. Output remains **unreviewed**.

## What was fixed

The current v4 batch entrypoint, landmark crop, and Geo-SAM encoder now share
the v2 source resolver (`pilot/source_config.py`). Set `MAPS_DIR` to the
directory containing the COGs on another machine. V4 also accepts `--cog`.
No v1 fallback. V4 defaults to a new `candidates_v4_v2.gpkg` filename and
refuses to overwrite existing outputs. Carry-over is disabled unless explicitly
requested with `--carry-from`; the prior file's source hash AND output hash
must match its manifest. The old v3 file cannot silently enter a v2 run.

Each output includes source filename/SHA-256, run ID, extractor version and
`reviewed=false` per feature, plus a manifest with code hashes, raster grid,
counts, timestamp, and output hash. This extractor requires the calibrated
Clarkson–Biddle raster grid; other grids fail rather than reinterpret its
pixel-based core mask.

**Historical scripts remain historical.** `batch_common.COG` and the old
v3/v5 visualization, findings, and repair scripts still refer to v1. Do not
use those legacy commands to review these v2 outputs, or replay old v5
findings on v2. This is a fresh **v4-on-v2 baseline, not a re-audited v5**.
The new package/commands are the v2 workflow. Existing committed results were
not overwritten. Geo-SAM was not re-encoded or tested inside QGIS this turn.

## Review in QGIS

1. Open `review.gpkg`: `benchmark_blocks`, `candidates`, `oversize_blocks`,
   and **empty** `reference_map_trace`. CRS is EPSG:3857.
2. Load `B1_v2.tif`, `B2_v2.tif`, `B3_v2.tif` beneath the vector layers.
   They are self-contained georeferenced crops of the source COG. Full-source
   filename/hash are in `candidates_v4_v2.manifest.json`. Geo-SAM's existing
   feature cache must be used with the matching full v2 source, not assumed
   to work with newly cropped rasters.
3. **Hide candidates before independent tracing.** Sunil (or another human
   reviewer) traces every building's map-ink footprint in each AOI from blank,
   using Geo-SAM/manual edits. Do not copy candidates into the reference layer.
   Include a whole footprint when its representative point lies in the AOI;
   do not clip it at the boundary. Inspect the full map for edge cases outside
   a crop. Record ambiguous building divisions explicitly before declaring
   a block complete; agree what counts as one building for both workflows.
4. Fill each reference's `reference_id`, `block_id`, `target=map_trace`,
   `reviewed_by`, `reviewed_at` (ISO date/time), `source_map` and `cog_sha256`.
   Copy the latter two from the candidate manifest, not from a different map.
   Modern survivor footprints belong to a **different target** and must not
   enter this layer. This scorer intentionally handles map-trace targets only.
5. Mark a block `complete=true` in `reference_completion.json` only after
   reviewing the entire AOI, including omissions/blank areas. Record reviewer
   and ISO date/time there too. The reference layer currently has **zero rows**
   and all completion flags are **false**; this is not a measured baseline yet.
6. In a **separate copy** of candidates, time accept/fix/delete/add review.
   Log actual tracing, correction and review seconds plus clicks in
   `timing.csv` for both workflows; blanks mean unmeasured, not zero.
   Record machine/model, cache warm/cold state and setup time in notes.
   Record accepted-unmodified counts separately. Prefer different reviewers
   or counterbalanced block order to reduce familiarity bias.

## Reproduce / score

From the Map_Reader root, with its existing `env` environment:

```
env/bin/python pilot/extract_candidates_v4.py --out pilot/results/NEW_RUN/candidates_v4_v2.gpkg
env/bin/python pilot/extract_candidates_v4.py --cog /path/to/tw_1762_philadelphia_map_clarkson_biddle_cog.tif --out pilot/results/NEW_RUN/control_v4_v1.gpkg
env/bin/python pilot/prepare_benchmark.py --directory pilot/results/NEW_RUN
env/bin/python pilot/score_benchmark.py --directory pilot/results/NEW_RUN
```

For the delivered package, substitute `benchmark_2026-09-06` for `NEW_RUN`
in the **score** command only; never regenerate over human review work.
The source COGs are large and excluded from this ZIP; crops are included.
Overlay PNGs are previews. The output scorer prints JSON only after completion
checks pass: one-to-one matching at IoU >= 0.5, precision/recall, matched IoUs,
sampled symmetric boundary distances in UTM metres, and split/merge flags.
Oversize features count as predictions. Timing remains separately logged;
no speedup is claimed without complete timing for both methods.

## Verification and stopping point

- Both output hashes and all recorded code hashes verified.
- Both full outputs: all geometry valid/nonempty, expected CRS.
- Seven scoring checks passed: exact, duplicate, split, merge, empty
  predictions, empty references, disjoint geometries.
- Existing-output and cross-raster carry-over rejection checked.
- Scorer correctly exits with an error on this incomplete human baseline.
- Python compilation and `git diff --check` passed.
- GPKGs re-opened with pyogrio/geopandas; interactive QGIS usage unverified.

**Remaining dependency:** human-reviewed reference polygons and timing.
No precision/recall, cleanup speedup, or fine-tuning recommendation can be
reported yet. No production publication and no expansion beyond the pilot.

Related audit clarification: the v2 GCPs are absent from the audited repos,
but a local `philly_georef_work/cb_v2/cb_gcps_v2.json` does exist. It was
located this turn; its georeferencing quality was not re-audited here.
