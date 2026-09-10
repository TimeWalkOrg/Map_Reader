"""Propose a block-level spatial train/val split (no training is run).
Validation = a contiguous band of road-graph blocks at the north-east end of the plate holding
~20 % of the on-plate building polygons; everything else on-plate = train. Polygons with no
enclosing block face (e.g. outside the road graph) are 'unassigned'. Writes proposed_split_buildings.csv,
proposed_split_landmarks.csv, adds `split` to block_coverage.gpkg and a summary to split_summary.json.
"""
import json, math
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd, shapely
BASE = Path(__file__).resolve().parent
GROUND = math.cos(math.radians(40.71))
bg = gpd.read_file(BASE / "block_coverage.gpkg")
b = gpd.read_file(BASE / "1776_nyc_parcels_buildings.gpkg"); l = gpd.read_file(BASE / "1776_nyc_parcels_landmarks.gpkg")
qa_b = pd.read_csv(BASE / "qa_buildings.csv").set_index("id"); cell_b = pd.read_csv(BASE / "qa_cells_buildings.csv").set_index("id")
qa_l = pd.read_csv(BASE / "qa_landmarks.csv").set_index("id"); cell_l = pd.read_csv(BASE / "qa_cells_landmarks.csv").set_index("id")
tree = shapely.STRtree(bg.geometry.values)
def block_of(geoms):
    out = []
    for g in geoms:
        c = g.centroid; idx = tree.query(c, predicate="within")
        out.append(int(bg.block_id.iloc[idx[0]]) if len(idx) else -1)
    return np.array(out)
b["block_id"] = block_of(b.geometry); l["block_id"] = block_of(l.geometry)
# NE axis: project block centroids onto the plate's long axis (x + y in 3857 increases toward NE)
usable_blocks = bg[bg.cls != "off_plate"].copy()
cent = usable_blocks.geometry.centroid; usable_blocks["ne_score"] = cent.x + cent.y
counts = b[b.id.map(qa_b.on_plate)].groupby("block_id").size()
usable_blocks["n_onplate_polys"] = usable_blocks.block_id.map(counts).fillna(0).astype(int)
usable_blocks = usable_blocks.sort_values("ne_score", ascending=False)
target = 0.20 * usable_blocks.n_onplate_polys.sum(); cum = usable_blocks.n_onplate_polys.cumsum()
val_blocks = set(usable_blocks.loc[cum <= target, "block_id"])
bg["split"] = np.where(bg.cls == "off_plate", "excluded_off_plate", np.where(bg.block_id.isin(val_blocks), "val", "train"))
bg.to_file(BASE / "block_coverage.gpkg", driver="GPKG")
def assign(df, qa, cell):
    s = pd.DataFrame({"id": df.id, "block_id": df.block_id})
    s["on_plate"] = s.id.map(qa.on_plate).astype(bool)
    s["alignment_class"] = s.id.map(qa.qa_class); s["cell_class"] = s.id.map(cell.cell_class)
    s["block_coverage_class"] = s.block_id.map(bg.set_index("block_id").cls).fillna("no_block")
    s["split"] = s.block_id.map(bg.set_index("block_id").split).fillna("unassigned")
    s.loc[~s.on_plate, "split"] = "excluded_off_plate"
    s["negatives_allowed_in_block"] = s.block_coverage_class == "exhaustive"
    return s
sb = assign(b, qa_b, cell_b); sl = assign(l, qa_l, cell_l)
sb.to_csv(BASE / "proposed_split_buildings.csv", index=False); sl.to_csv(BASE / "proposed_split_landmarks.csv", index=False)
traced = sb.cell_class.isin(["traced_single_cell", "traced_merged_cells"])
summ = {
  "val_blocks": sorted(int(x) for x in val_blocks), "n_val_blocks": len(val_blocks),
  "n_train_blocks": int(((bg.split == "train")).sum()),
  "buildings": {"train": int((sb.split == "train").sum()), "val": int((sb.split == "val").sum()),
                "excluded_off_plate": int((sb.split == "excluded_off_plate").sum()), "unassigned": int((sb.split == "unassigned").sum()),
                "traced_cell_match_train": int((traced & (sb.split == "train")).sum()), "traced_cell_match_val": int((traced & (sb.split == "val")).sum()),
                "in_exhaustive_blocks": int(sb.negatives_allowed_in_block.sum()),
                "traced_and_in_exhaustive_blocks": int((traced & sb.negatives_allowed_in_block).sum())},
  "landmarks": {"train": int((sl.split == "train").sum()), "val": int((sl.split == "val").sum()),
                "excluded_off_plate": int((sl.split == "excluded_off_plate").sum()), "unassigned": int((sl.split == "unassigned").sum())},
}
(BASE / "split_summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ))
