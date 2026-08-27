# v5 — Full-sheet vision audit of the 1762 building-footprint vectors

**Date:** 2026-08-27 · **Input:** `candidates_v4.gpkg` (688 candidates + 27
oversize blocks, commit c9c695b) · **Output:** `candidates_v5.gpkg`
(**798 candidates + 26 oversize_blocks + 24 needs_review**)

Trigger: Sunil's QC screenshots on v4 (2026-08-27) — hatched buildings still
missing, wrong geometry around enclosed hatch areas, polygons on street
lettering/streets. Rule of the map: parcels belong ONLY around hatched-line
areas.

## The audit loop

1. **Tile** (`pilot/audit_vision.py tile`) — the city core is cut into
   **132 content-bearing tiles** (704 px ≈ 283 m ≈ 2 blocks, stride 512 px,
   192 px overlap; blank tiles skipped). Each tile stores its georeferencing
   in `audit_v5/tiles_manifest.json`, so pixel findings map back to
   EPSG:3857.
2. **Render pairs** — per tile: raw crop + identical crop with the vector
   overlay (green candidates / orange oversize), with a faint labelled
   128 px grid so the judge can report pixel bboxes.
3. **Vision judge** — each pair goes through a vision model with a strict
   rubric (four categories: `missing_feature`, `false_positive`,
   `shape_mismatch`, `missing_split`; strict-JSON reply with per-finding
   pixel bbox + confidence). One tile pair per prompt.
4. **Verify + fix** (`pilot/apply_fixes_v5.py`) — every finding is
   re-verified against **raster evidence** before anything changes
   (structure-tensor hatch fraction, dark-fill rectangularity, ink-overhang
   measurement, relaxed dark-divider split). Confident fixes are applied;
   ambiguity goes to the `needs_review` layer with a note; findings
   contradicted by the raster are logged as dismissed. Nothing is silently
   guessed.
5. **Re-audit** changed tiles only; iterate (3 passes total).

## Pass-by-pass

| pass | tiles judged | "issues" verdicts | findings (deduped) | auto-fixed | needs_review | dismissed |
|---|---|---|---|---|---|---|
| 1 | 132 (full sheet) | 89 | 288 → 265 | **111** (10 del / 98 add / 3 reshape) | 11 | 143 |
| 2 | 80 (changed) | 73 | 193 → 184 | **30** (5 del / 18 add / 7 reshape) | 9 | 145 |
| 3 | 39 (changed) | 36 | 95 → 90 | **16** (2 del / 11 add / 3 reshape) | 3 | 71 |

Auto-fix volume converged 111 → 30 → 16 (max-3-passes stop). A post-fix
sweep re-ran the v4 hatch-core trim on all vision-added polygons: **29
additions trimmed** of lettering/street-line tails; 1 flagged for review
instead (real building whose outline drapes onto the Lombard-street
lettering flourish — resisted automated tidying, see `needs_review`).

**Dismissals are dominated by judge noise, and that is by design:** 61 % of
`missing_feature` reports pointed at buildings already covered by a polygon
(bbox imprecision at this zoom) or at tree scribbles/shading with no hatch
signature; the raster gate absorbed them. The two-stage design (cheap
eager vision + strict computational verification) is what makes the loop
safe to run unattended.

## Fixes by category (all passes)

- **false_positive → 17 deleted** — polygons on street lettering (incl. the
  script-letter knots Sunil circled), tree scribbles, blank ground.
  Deletion gate: relaxed-hatch fraction < 0.06 under the polygon AND not a
  compact rectangular dark fill (rectangularity test separates solid-fill
  buildings from letter blobs).
- **missing_feature → 127 added** (121 survive; a handful were re-judged
  and removed in later passes). Add gate: an ink component in the bbox with
  hatch fraction ≥ 0.22 (or solid rectangular dark fill), < 25 % already
  covered; regularized + ink-clamped, `reg_method='v5vision*'`.
- **shape_mismatch → 13 reshaped** — measured overhang beyond the ink
  > 20 % → re-clamped; missing > 25 % of the hatch core → extended.
- **missing_split → 0 auto-split, 15 → needs_review** — the relaxed
  dark-divider detector found no confident internal divider in any flagged
  block; per policy these went to review rather than being cut on a guess.

## Net change v4 → v5

| | v4 | v5 |
|---|---|---|
| candidates | 688 | **798** (672 kept, 91 added, 28 added+trimmed, 7 reshaped) |
| oversize_blocks | 27 | **26** |
| needs_review | — | **24** (15 missing_split, 6 missing_feature, 3 shape_mismatch) |

Sunil's five screenshot locations were recovered on the sheet by
multi-scale template matching (scores 0.80–0.91; the fifth was pinned via
its skewed oversize block) and are covered by before/after QC crops:
`qc_v5_sunil_*.png` + `qc_v5_recall_society_hill_south.png` (densest
recall cluster, 7 buildings added on one tile).

**Registration sanity** (`measure_offset.py` on v5): median offset
**dx 0.01 m / dy 0.00 m** vs the source ink — unchanged from v4.

## Honest residuals — what v5 does NOT claim

- **Recall of faint/small hatched houses is still incomplete.** Pass 3
  judges still reported 78 `missing_feature` findings on the 39 re-audited
  tiles; 61 were dismissed on raster evidence, but some dismissals
  ("already covered", "weak hatch") hide genuine small buildings whose
  hatching is too faint/fine for the tensor gate. Expect on the order of
  1–2 real misses per dense tile in the worst areas (Southwark, Northern
  Liberties edges).
- **Oversize blocks remain hand-split work.** No confident divider = no
  cut; the 15 `missing_split` review flags mark where the judge saw parcel
  boundaries the detector could not confirm.
- **A few added outlines are tight to the hatch core but not pretty** —
  ortho-regularization hugs ragged hatch edges; they are correct in
  position/extent but will benefit from hand-tidying during review.
- The **28 changed tiles after pass 3 were not re-audited** (max-passes
  stop); the pass-3 fix set was small and evidence-gated, but it is not
  vision-verified.

## Reusing the harness for ROADS (design note, per Sunil)

The tiler, manifest, pair renderer, and judge plumbing are
feature-agnostic: `audit_vision.py` takes `--feature`, and each feature
type in `FEATURES` supplies (a) the vector layers + overlay colors and (b)
the rubric text. A roads pass needs only: a `roads` entry with a
road-centerline/casing rubric and layer spec, plus a road-specific verify
module in the fixer (the raster evidence functions — tensor fields, ink
mask, divider detector — are already importable). Tile size/stride and the
bbox→map-coords plumbing carry over unchanged.

## Reproduce

```bash
env/bin/python pilot/audit_vision.py tile --gpkg pilot/results/candidates_v4.gpkg
# judge tile pairs (vision model, strict-JSON rubric) -> findings_passN_*.jsonl
env/bin/python pilot/apply_fixes_v5.py --pass 1 \
  --in pilot/results/candidates_v4.gpkg \
  --findings 'pilot/results/audit_v5/findings_pass1_*.jsonl' \
  --out pilot/results/candidates_v5.gpkg
env/bin/python pilot/audit_vision.py render --gpkg pilot/results/candidates_v5.gpkg \
  --tiles "$(cat pilot/results/audit_v5/changed_tiles_pass1.txt)" --suffix p2
# ...repeat judge+fix for passes 2/3, then:
env/bin/python pilot/postfix_v5_trim.py
env/bin/python pilot/render_v5_qc.py
env/bin/python pilot/measure_offset.py pilot/results/candidates_v5.gpkg
```
