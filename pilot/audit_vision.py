#!/usr/bin/env python3
"""Full-sheet vision-audit harness for Map_Reader vector layers (v5 pass).

Cuts the city core of a georeferenced COG into overlapping high-zoom tiles,
renders per-tile image pairs (raw crop | crop + vector overlay), and stores a
georeferenced tile manifest so pixel-space findings from a vision judge can
be mapped back to map (EPSG:3857) coordinates.

The harness is FEATURE-GENERIC (see FEATURES below): the tiling, rendering,
manifest and rubric plumbing take a feature-type key, so the same loop can
audit ROAD vectors (or wharves, etc.) from this map later — only a rubric
and a layer spec need to be added.

Subcommands:
  tile    --gpkg PATH [--feature buildings] [--outdir DIR]
          Build tile grid + manifest + render all pairs.
  render  --gpkg PATH --tiles T012,T044,... [--suffix p2]
          Re-render overlay (and raw if missing) for specific tiles after
          the vector layer changed (re-audit passes).
  rubric  [--feature buildings]
          Print the vision-judge rubric prompt for a tile pair.

Run:  env/bin/python pilot/audit_vision.py tile --gpkg pilot/results/candidates_v4.gpkg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box as sbox
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))

TILE = 704          # px rendered tile (704 px * 0.402 m/px ~ 283 m ~ 2 blocks)
STRIDE = 512        # px stride -> 192 px overlap between neighbours
MIN_INK = 0.004     # skip tiles blanker than this unless they carry vectors

# ---------------------------------------------------------------------------
# feature-type configs (add "roads" etc. here for future passes)
# ---------------------------------------------------------------------------
FEATURES = {
    "buildings": {
        # (layer name, BGR outline colour, thickness)
        "layers": [("candidates", (0, 200, 0), 2),
                   ("oversize_blocks", (0, 140, 255), 2)],
        "rubric": """You are auditing machine-extracted BUILDING-FOOTPRINT vectors drawn over a 1762 engraved map of Philadelphia (Clarkson & Biddle). You get TWO images of the SAME map area:
- IMAGE 1 (RAW): the map crop with no overlay.
- IMAGE 2 (OVERLAY): the identical crop with vector outlines: GREEN = building candidates, ORANGE = oversize blocks awaiting hand-split. Faint blue grid lines are drawn every 128 px with pixel labels on the top/left edges to help you report coordinates.

MAP CONVENTIONS: buildings are shown as areas filled with fine parallel diagonal HATCH lines (lower-left to upper-right), occasionally as solid dark fills (public buildings). Streets are blank corridors, often carrying italic street-name LETTERING. Trees/orchards are small round scribbles; creeks and shading are wavy lines; dashed lines are lot boundaries, not buildings. RULE: parcels belong ONLY on hatched (or solid-filled) building areas.

Report ONLY clear, decisive problems, using these categories:
1. "missing_feature"  - a hatched/solid-filled building with NO green/orange outline over it (ignore anything smaller than ~10x10 px, tree scribbles, and hatching that is clearly water/shading along the river).
2. "false_positive"   - an outline enclosing ONLY non-building content: street lettering, blank street/open ground, trees, water. (An outline that includes SOME hatching is NOT this category.)
3. "shape_mismatch"   - an outline whose shape is clearly wrong for the hatched area it sits on: it overhangs far onto blank street/lettering (by >~25% of its area), is skewed/rotated relative to the hatched block, or misses a large part (>~25%) of its building's hatching.
4. "missing_split"    - one single outline spanning what the map draws as TWO OR MORE distinct buildings/parcels separated by a clear straight dark ink line crossing the hatched block.

Respond with STRICT JSON only (no markdown fences, no commentary):
{"tile": "<TILE_ID>", "verdict": "ok" | "issues", "findings": [{"category": "...", "bbox_px": [x0, y0, x1, y1], "confidence": "high"|"medium"|"low", "note": "<max 15 words>"}]}

bbox_px are pixel coordinates in the 704x704 images (use the blue grid: labels mark pixels). Be conservative: if unsure, either omit the finding or mark confidence "low". An empty findings list with verdict "ok" is a perfectly good answer. Do not report stylistic quibbles; only the four categories above. TILE_ID is: """,
    },
    # "roads": {...}  # future: road centerlines/casings rubric goes here
}


def _draw_grid(vis):
    h, w = vis.shape[:2]
    for p in range(128, max(h, w), 128):
        if p < w:
            vis[:, p, :] = (vis[:, p, :].astype(np.int32) * 2 // 3
                            + np.array([85, 40, 0], np.int32) // 3 * 1
                            ).astype(np.uint8)
            cv2.putText(vis, str(p), (p - 18, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 60, 0), 1,
                        cv2.LINE_AA)
        if p < h:
            vis[p, :, :] = (vis[p, :, :].astype(np.int32) * 2 // 3
                            + np.array([85, 40, 0], np.int32) // 3 * 1
                            ).astype(np.uint8)
            cv2.putText(vis, str(p), (2, p + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 60, 0), 1,
                        cv2.LINE_AA)


def _render_pair(src, gdfs, feature, tid, c0, r0, w, h, outdir, suffix=""):
    win = Window(c0, r0, w, h)
    gray = bc.gray_from_tile(src.read(window=win))
    tr = src.window_transform(win)
    inv = ~tr
    x0, y0 = tr * (0, h)
    x1, y1 = tr * (w, 0)
    bbox = sbox(x0, y0, x1, y1)

    raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _draw_grid(raw)
    cv2.putText(raw, f"{tid} RAW", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 220), 2, cv2.LINE_AA)

    ovl = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_polys = 0
    for (lname, color, thick) in FEATURES[feature]["layers"]:
        gdf = gdfs.get(lname)
        if gdf is None or len(gdf) == 0:
            continue
        idx = list(gdf.sindex.intersection((x0, y0, x1, y1)))
        for geom in gdf.geometry.iloc[idx]:
            if not geom.intersects(bbox):
                continue
            n_polys += 1
            polys = (geom.geoms if geom.geom_type == "MultiPolygon"
                     else [geom])
            for p in polys:
                for ring in [p.exterior] + list(p.interiors):
                    pts = np.array([inv * (xx, yy) for xx, yy in ring.coords])
                    cv2.polylines(ovl, [np.round(pts).astype(np.int32)],
                                  True, color, thick, cv2.LINE_AA)
    _draw_grid(ovl)
    cv2.putText(ovl, f"{tid} OVERLAY", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)

    sfx = f"_{suffix}" if suffix else ""
    p_raw = os.path.join(outdir, "tiles", f"{tid}_raw{sfx}.png")
    p_ovl = os.path.join(outdir, "tiles", f"{tid}_ovl{sfx}.png")
    if suffix and not os.path.exists(p_raw):
        # raw never changes between passes; reuse pass-1 raw if present
        p_raw0 = os.path.join(outdir, "tiles", f"{tid}_raw.png")
        p_raw = p_raw0 if os.path.exists(p_raw0) else p_raw
    if not os.path.exists(p_raw):
        cv2.imwrite(p_raw, raw)
    cv2.imwrite(p_ovl, ovl)
    return p_raw, p_ovl, n_polys, gray


def cmd_tile(args):
    src = rasterio.open(bc.COG)
    core = bc.load_core_poly(src)
    gdfs = _load_gdfs(args.gpkg, args.feature)
    outdir = args.outdir
    os.makedirs(os.path.join(outdir, "tiles"), exist_ok=True)

    ncols = int(np.ceil(src.width / STRIDE))
    nrows = int(np.ceil(src.height / STRIDE))
    manifest = {"gpkg": os.path.abspath(args.gpkg), "feature": args.feature,
                "tile_px": TILE, "stride_px": STRIDE,
                "cog": bc.COG, "crs": str(src.crs), "tiles": []}
    n = 0
    for tr_ in range(nrows):
        for tc in range(ncols):
            c0 = tc * STRIDE
            r0 = tr_ * STRIDE
            w = min(TILE, src.width - c0)
            h = min(TILE, src.height - r0)
            if w < 128 or h < 128:
                continue
            x0, y0 = src.transform * (c0, r0 + h)
            x1, y1 = src.transform * (c0 + w, r0)
            if not sbox(x0, y0, x1, y1).intersects(core):
                continue
            tid = f"T{tr_:02d}{tc:02d}"
            p_raw, p_ovl, n_polys, gray = _render_pair(
                src, gdfs, args.feature, tid, c0, r0, w, h, outdir)
            ink = float(bc.ink_mask(gray).mean())
            if ink < MIN_INK and n_polys == 0:
                for p in (p_raw, p_ovl):
                    if os.path.exists(p):
                        os.remove(p)
                continue
            t = src.window_transform(Window(c0, r0, w, h))
            manifest["tiles"].append({
                "id": tid, "col0": c0, "row0": r0, "w": w, "h": h,
                "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                "bounds": [x0, y0, x1, y1],
                "ink_frac": round(ink, 4), "n_polys": n_polys,
                "raw": os.path.abspath(p_raw), "ovl": os.path.abspath(p_ovl),
            })
            n += 1
            if n % 25 == 0:
                print(f"{n} tiles rendered...", flush=True)
    mpath = os.path.join(outdir, "tiles_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {mpath}: {n} tiles")


def _load_gdfs(gpkg, feature):
    import pyogrio
    have = {l[0] for l in pyogrio.list_layers(gpkg)}
    gdfs = {}
    for (lname, _c, _t) in FEATURES[feature]["layers"]:
        if lname in have:
            gdfs[lname] = gpd.read_file(gpkg, layer=lname)
    return gdfs


def cmd_render(args):
    src = rasterio.open(bc.COG)
    gdfs = _load_gdfs(args.gpkg, args.feature)
    outdir = args.outdir
    mpath = os.path.join(outdir, "tiles_manifest.json")
    manifest = json.load(open(mpath))
    want = set(args.tiles.split(","))
    for t in manifest["tiles"]:
        if t["id"] not in want:
            continue
        p_raw, p_ovl, n_polys, _ = _render_pair(
            src, gdfs, args.feature, t["id"], t["col0"], t["row0"],
            t["w"], t["h"], outdir, suffix=args.suffix)
        print(f"{t['id']}: {p_raw} | {p_ovl} ({n_polys} polys)")


def cmd_rubric(args):
    print(FEATURES[args.feature]["rubric"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tile", "render", "rubric"):
        p = sub.add_parser(name)
        p.add_argument("--feature", default="buildings",
                       choices=list(FEATURES))
        p.add_argument("--outdir",
                       default=os.path.join(HERE, "results", "audit_v5"))
        if name in ("tile", "render"):
            p.add_argument("--gpkg", required=True)
        if name == "render":
            p.add_argument("--tiles", required=True)
            p.add_argument("--suffix", default="p2")
    args = ap.parse_args()
    {"tile": cmd_tile, "render": cmd_render, "rubric": cmd_rubric}[args.cmd](
        args)


if __name__ == "__main__":
    main()
