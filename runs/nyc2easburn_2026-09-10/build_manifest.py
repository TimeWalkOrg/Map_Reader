#!/usr/bin/env python
"""Assemble run_manifest.json: raster hashes, code commit, params, timings, counts, output hashes (skill step 3)."""
import json, hashlib, os, subprocess, glob, time
import geopandas as gpd

RUN = os.path.dirname(os.path.abspath(__file__))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""): h.update(ch)
    return h.hexdigest()
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
prep = J(f"{RUN}/prep_manifest.json")
logs = {t: J(f"{RUN}/ckpt/{t}_train_log.json") for t in ("roads", "buildings")}
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RUN, capture_output=True, text=True).stdout.strip()
outputs = {}
for f in sorted(glob.glob(f"{RUN}/*.gpkg")):
    g = gpd.read_file(f)
    outputs[os.path.basename(f)] = dict(sha256=sha(f), rows=int(len(g)), crs=str(g.crs), all_valid=bool(g.is_valid.all()), n_empty=int(g.is_empty.sum()),
                                        geom_types=sorted(g.geom_type.unique().tolist()), origin=sorted(g.origin.unique().tolist()) if "origin" in g else None)
man = {
    "run": "nyc2easburn_2026-09-10", "status": "machine_candidate_run_NOT_accepted_geometry",
    "requested_by": "Sunil Prabhu, Slack C08BNQJQB1P thread 1789070222.107089",
    "code_commit_before_run": "ac8bae1 (Map_Reader HEAD when run started)", "code_commit_at_manifest": git,
    "training_source": {"dataset": "training/sunil_nyc_1776_fire_map_2026-09-10 (GPKG exports, manifest.json sha256s)", "raster": prep["nyc_raster"] if prep else None,
                        "label_use": "buildings+landmarks = WEAK location labels (eroded 3 m, boundary ignore); negatives only in 10 exhaustive blocks + road corridors; roads = corridor mask from qa_roads-scorable edges only",
                        "split": "block-level spatial (val = NE band, 41 blocks) per proposed_split_*.csv; train tiles contain no val pixels", "tiles": prep["nyc_tiles"] if prep else None},
    "target_raster": prep["easburn_raster"] if prep else None,
    "common_grid": {"res_m_per_px_epsg3857": 0.5, "preprocessing": "grayscale -> background-normalised darkness (close41+blur41) + adaptive-threshold binary ink (block 41, C 18); 2 channels"},
    "models": {t: (dict(arch="UNet base32 depth5, 7.76M params, 2-ch input", device=l.get("gpu"), torch=l.get("torch"), iters=l.get("iters_done"), seconds=l.get("seconds"),
                        best_val_iou_weak_labels=l.get("best_val_iou"), n_train_tiles=l.get("n_train_tiles"), n_val_tiles=l.get("n_val_tiles"),
                        checkpoint=f"ckpt/{t}_best.pt", checkpoint_sha256=sha(f"{RUN}/ckpt/{t}_best.pt") if os.path.exists(f"{RUN}/ckpt/{t}_best.pt") else None) if l else None) for t, l in logs.items()},
    "roads_iterations": {"v1": "thin 2 m centerline buffer on all roads.cga edges -> val IoU ~0.10 (log ckpt/roads_v1_train_log.json)",
                          "v2": "corridor (0.4*strtwdth) on all edges -> val IoU ~0.20 (undrawn NE streets taught as roads)",
                          "v3 (shipped)": "corridor on 530 qa_roads-scorable edges only, unscorable edges ignored -> best val IoU 0.42 @ iter 750"},
    "inference": {"easburn": J(f"{RUN}/easburn_infer_manifest.json"), "nyc_fullplate": J(f"{RUN}/nyc_fullplate_infer_manifest.json"),
                  "easburn_instances": J(f"{RUN}/easburn_instance_manifest.json")},
    "geosam_features": {"path": "geo-sam/features_easburn_sam_b (SAM-B, 213 chips, encode_easburn_run.json)", "source_sha256_matches_target": True},
    "metrics": J(f"{RUN}/metrics.json"),
    "outputs": outputs,
    "review_status": "No human review of candidates yet; interactive QGIS verification incomplete; nothing written to PostGIS or the QGIS project.",
    "manifest_written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
}
json.dump(man, open(f"{RUN}/run_manifest.json", "w"), indent=1, default=str)
print(json.dumps({k: man[k] for k in ("code_commit_at_manifest", "outputs")}, indent=1, default=str))
