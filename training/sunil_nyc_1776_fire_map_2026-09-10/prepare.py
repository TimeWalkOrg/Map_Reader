"""Build lossless GeoPackages from the read-only PostGIS snapshot exports.

Inputs (export/): `psql \\copy` CSVs of timewalk."<layer>" (all columns, geom as hex
EWKB, NULL encoded as \\N), snapshot_utc.txt, columns.csv, indexes.csv.
Outputs: <layer>.gpkg (all columns + geometry, EPSG:3857, FID column named
`gpkg_fid` so the road graph's own numeric `fid` attribute survives untouched)
and validation.json. Geometry round-trip is verified byte-exact (WKB).
No database connection is made here. Run with Map_Reader/env/bin/python.
"""
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

BASE = Path(__file__).resolve().parent
EXPORT = BASE / "export"
LAYERS = {
    "1776_nyc_parcels_buildings": "id",
    "1776_nyc_parcels_landmarks": "id",
    "1776_nyc_vector_road_graph": "ogc_fid",
}
INT_TYPES = {"integer", "bigint", "smallint"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cols = pd.read_csv(EXPORT / "columns.csv")
    snapshot = (EXPORT / "snapshot_utc.txt").read_text().strip()
    report = {"snapshot_utc": snapshot, "layers": {}}
    for name, pk in LAYERS.items():
        csv_path = EXPORT / f"{name}.csv"
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=["\\N"])
        meta = cols[cols.table_name == name].sort_values("ordinal_position")
        assert list(df.columns) == list(meta.column_name), name
        geom = shapely.from_wkb([bytes.fromhex(h) for h in df["geom"]])
        # SRID from EWKB header (byte order 01 little endian; flags at bytes 1..4; srid at bytes 5..8)
        srid_set = {int.from_bytes(bytes.fromhex(h[10:18]), "little") for h in df["geom"]}
        assert srid_set == {3857}, srid_set
        attrs = df.drop(columns=["geom"]).copy()
        for _, r in meta.iterrows():
            c = r.column_name
            if c == "geom":
                continue
            if r.data_type in INT_TYPES:
                attrs[c] = pd.to_numeric(attrs[c]).astype("Int64")
            elif r.data_type == "numeric":
                attrs[c] = pd.to_numeric(attrs[c]).astype("float64")
            elif r.data_type == "boolean":
                attrs[c] = attrs[c].map({"t": True, "f": False}).astype("boolean")
            # date / text / varchar stay as ISO strings (lossless)
        g = gpd.GeoDataFrame(attrs, geometry=geom, crs="EPSG:3857")
        gpkg = BASE / f"{name}.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        g.to_file(gpkg, layer=name, driver="GPKG", layer_options={"FID": "gpkg_fid"})
        back = gpd.read_file(gpkg, layer=name)
        assert len(back) == len(g) and back.crs == g.crs
        assert list(back.columns) == list(g.columns), (list(back.columns), list(g.columns))
        exact = bool(np.all(shapely.to_wkb(back.geometry.values) == shapely.to_wkb(g.geometry.values)))
        assert exact, name
        # attribute round trip: compare the raw CSV text with the GPKG read-back,
        # normalising NULL spellings and integer-valued floats.
        def norm(s):
            return s.map(lambda v: "" if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
                         else (str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() else str(v)))
        raw = df.drop(columns=["geom"])
        b = back.drop(columns="geometry")
        attr_mismatch = 0
        for c in raw.columns:
            dtype = meta.set_index("column_name").data_type[c]
            if dtype == "numeric":
                x, y = pd.to_numeric(raw[c]), pd.to_numeric(b[c])
                attr_mismatch += int((~((x == y) | (x.isna() & y.isna()))).sum())
                continue
            x, y = norm(raw[c]), norm(b[c])
            if dtype == "boolean":
                x = x.map({"t": "True", "f": "False", "": ""})
            attr_mismatch += int((x != y).sum())

        valid = g.is_valid
        gtypes = sorted(set(g.geom_type))
        has_z = bool(shapely.has_z(g.geometry.values).all())
        metric = g.to_crs(32618)
        norm = shapely.to_wkb(shapely.normalize(g.geometry.values))
        dup_geoms = int(pd.Series(norm).duplicated().sum())
        entry = {
            "source_table": f'timewalk."{name}"',
            "primary_key": pk,
            "row_count": int(len(g)),
            "distinct_pk": int(g[pk].nunique()),
            "columns": [c for c in df.columns],
            "column_types": {r.column_name: r.data_type for _, r in meta.iterrows()},
            "srid": 3857,
            "geometry_types": gtypes,
            "has_z": has_z,
            "valid": int(valid.sum()),
            "invalid_ids": [int(i) for i in g.loc[~valid, pk]],
            "invalid_reasons": [shapely.is_valid_reason(x) for x in g.loc[~valid].geometry],
            "empty": int(g.is_empty.sum()),
            "exact_duplicate_geometries": dup_geoms,
            "bounds_epsg3857": [float(v) for v in g.total_bounds],
            "bounds_wgs84": [float(v) for v in g.to_crs(4326).total_bounds],
            "csv_sha256": sha256(csv_path),
            "gpkg_sha256": sha256(gpkg),
            "gpkg_geometry_roundtrip_exact": exact,
            "gpkg_attribute_mismatch_cells": attr_mismatch,
        }
        if gtypes[0].endswith("Polygon"):
            area = metric.area
            entry["area_utm18n_m2"] = {
                "min": float(area.min()), "median": float(area.median()),
                "max": float(area.max()), "total": float(area.sum()),
            }
            entry["vertices_per_feature_mean"] = float(shapely.get_num_coordinates(g.geometry.values).mean())
            fixed = shapely.make_valid(metric.geometry.values)
            tree = shapely.STRtree(fixed)
            pairs = tree.query(fixed, predicate="overlaps")
            ov = []
            for i, j in zip(*pairs):
                if i < j:
                    a_ = shapely.intersection(fixed[i], fixed[j]).area
                    if a_ > 1.0:
                        ov.append({"id_a": int(g[pk].iloc[i]), "id_b": int(g[pk].iloc[j]), "area_m2": round(float(a_), 2)})
            entry["overlapping_pairs_over_1_m2"] = len(ov)
            entry["overlaps_over_1_m2"] = sorted(ov, key=lambda d: -d["area_m2"])[:25]
            entry["overlap_total_m2"] = round(float(sum(d["area_m2"] for d in ov)), 2)
        else:
            entry["total_length_utm18n_m"] = float(metric.length.sum())
        report["layers"][name] = entry
        print(name, len(g), gtypes, "valid", int(valid.sum()), "dups", dup_geoms, "roundtrip", exact, "attr_mismatch", attr_mismatch)
    report["export_sha256"] = {p.name: sha256(p) for p in sorted(EXPORT.iterdir())}
    (BASE / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
