"""Legacy Clarkson & Biddle experiment source, not the primary project map.

Easburn v2 is primary (Sunil, 2026-09-07); see training/active_dataset.json.
Keep this map-specific experiment pinned: its extents/encodings are not
portable to Easburn. Historical v1 audit scripts remain frozen.
"""
import hashlib
import os
from pathlib import Path

MAPS_DIR = Path(os.environ.get('MAPS_DIR', Path.home() / 'NYC_Maps' / 'maps'))
COG = MAPS_DIR / 'tw_1762_philadelphia_map_clarkson_biddle_v2_cog.tif'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
