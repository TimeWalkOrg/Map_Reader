#!/usr/bin/env python3
"""Rewrite feature_path entries in a Geo-SAM manifest.parquet after moving it.

The Geo-SAM encoder stores ABSOLUTE feature paths in manifest.parquet, so a
feature folder copied to another machine won't load until the paths are fixed.
Run this ONCE after unzipping, pointing it at the folder that contains
manifest.parquet. Requires geopandas + pyarrow (both are installed by the
Geo-SAM plugin, so the QGIS Python console works).

CLI:
    python fix_manifest_paths.py /path/to/tw_1762_philadelphia_map_clarkson_biddle_v2_cog

QGIS Python console (Plugins > Python Console):
    FEATURE_DIR = r"C:/path/to/tw_1762_philadelphia_map_clarkson_biddle_v2_cog"
    exec(open(r"C:/path/to/fix_manifest_paths.py").read())
"""

from pathlib import Path


def fix(feature_dir) -> None:
    import geopandas as gpd

    feature_dir = Path(feature_dir).expanduser().resolve()
    manifest_path = feature_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.parquet not found in {feature_dir}")
    m = gpd.read_parquet(manifest_path)
    m["feature_path"] = m["chip_id"].map(
        lambda cid: str(feature_dir / "features" / f"{cid}.pt")
    )
    missing = [p for p in m["feature_path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} feature files missing, e.g. {missing[0]} — "
            "is the features/ folder next to manifest.parquet?"
        )
    m.to_parquet(manifest_path)
    print(f"OK: rewrote {len(m)} feature paths in {manifest_path}")


if "FEATURE_DIR" in dir():  # QGIS console usage: set FEATURE_DIR, then exec this file
    fix(FEATURE_DIR)  # noqa: F821
elif __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        fix(sys.argv[1])
    else:
        print(__doc__)
