"""Build a source-locked three-block review package; no database writes."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon
from pyproj import Transformer

from batch_common import gray_from_tile
from source_config import COG, sha256

# Street-centerline AOIs, not building ground truth. Coordinates from the
# local 1776 road graph; Second/Pine from intersections_3857.json because
# the divided road graph has two crossings there. These define the sample
# only and do NOT supply evidence of 1762 building positions.
EAST = [(-8365095.332164237, 4857880.692221376),
        (-8365137.889605569, 4857678.286419562),
        (-8365166.932860715, 4857547.145923692),
        (-8365179.545359024, 4857400.2537811445)]
WEST = [(-8365312.082344762, 4857916.353240448),
        (-8365346.702706399, 4857712.785136993),
        (-8365368.298687612, 4857583.473642349),
        (-8365393.0116145685, 4857434.678936688)]
STREETS = ['Spruce', 'Pine', 'Lombard', 'Cedar (modern South)']


def draw(gray, frame, transform, title):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    inv = ~transform
    for geom in frame.geometry:
        for p in ([geom] if geom.geom_type == 'Polygon' else geom.geoms):
            for ring in [p.exterior, *p.interiors]:
                pts = np.round([inv * xy for xy in ring.coords]).astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 150, 0), 1, cv2.LINE_AA)
    vis = cv2.copyMakeBorder(vis, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    args = ap.parse_args()
    out = args.directory
    if (out / 'review.gpkg').exists():
        ap.error('Review package already exists; never overwrite human edits.')
    inputs = {}
    for name in ['candidates_v4_v2', 'control_v4_v1']:
        p = out / (name + '.gpkg')
        m = json.loads(p.with_suffix('.manifest.json').read_text())
        if m['output_sha256'] != sha256(p):
            raise ValueError('Candidate file differs from its manifest')
        inputs[name] = m
    if inputs['candidates_v4_v2']['cog_sha256'] != sha256(COG):
        raise ValueError('Current v2 raster differs from candidate source')
    v1 = COG.with_name(COG.name.replace('_v2_cog', '_cog'))
    if inputs['control_v4_v1']['cog_sha256'] != sha256(v1):
        raise ValueError('Control raster differs from candidate source')
    rows = [dict(block_id=f'B{i+1}', name=f'Second–Third / {STREETS[i]}–{STREETS[i+1]}',
                 geometry=Polygon([EAST[i], WEST[i], WEST[i+1], EAST[i+1]])) for i in range(3)]
    blocks = gpd.GeoDataFrame(rows, crs=3857)
    blocks.to_file(out / 'review.gpkg', layer='benchmark_blocks', driver='GPKG')
    for layer in ['candidates', 'oversize_blocks']:
        all_c = gpd.read_file(out / 'candidates_v4_v2.gpkg', layer=layer)
        pieces = []
        for _, block in blocks.iterrows():
            subset = all_c[all_c.geometry.representative_point().map(block.geometry.covers)].copy()
            subset['block_id'] = block.block_id
            subset['candidate_id'] = [f'{layer}-{i}' for i in subset.index]
            pieces.append(subset)
        import pandas as pd
        selected = gpd.GeoDataFrame(pd.concat(pieces), crs=3857)
        selected.to_file(out / 'review.gpkg', layer=layer, driver='GPKG')
    # Empty independent reference layer: deliberately NOT copied from candidates.
    import pyogrio
    import pandas as pd
    ref = gpd.GeoDataFrame({k: pd.Series(dtype='str') for k in
        ['reference_id','block_id','target','reviewed_by','reviewed_at','source_map','cog_sha256']},
        geometry=gpd.GeoSeries([], crs=3857), crs=3857)
    pyogrio.write_dataframe(ref, out / 'review.gpkg', layer='reference_map_trace',
                            driver='GPKG', geometry_type='MultiPolygon')
    old = gpd.read_file(out / 'control_v4_v1.gpkg', layer='candidates')
    new = gpd.read_file(out / 'candidates_v4_v2.gpkg', layer='candidates')
    metric = Transformer.from_crs(3857, 32618, always_xy=True)
    diagnostics = []
    with rasterio.open(COG) as current, rasterio.open(v1) as previous:
        for _, block in blocks.iterrows():
            bbox = block.geometry.buffer(25).bounds
            win = from_bounds(*bbox, current.transform).round_offsets().round_lengths()
            tr = current.window_transform(win)
            data = current.read(window=win)
            gray = gray_from_tile(data)
            before = gray_from_tile(previous.read(window=win))
            h,w = gray.shape
            hann = cv2.createHanningWindow((w,h), cv2.CV_32F)
            (dx,dy), response = cv2.phaseCorrelate(before.astype(np.float32)*hann,
                                                   gray.astype(np.float32)*hann)
            x,y = tr * (w/2,h/2)
            xa,ya = metric.transform(x,y)
            xb,yb = metric.transform(x+dx*tr.a,y+dy*tr.e)
            diagnostics.append(dict(block_id=block.block_id, dx_px=dx, dy_px=dy,
                response=response, displacement_m_utm=float(np.hypot(xb-xa,yb-ya)),
                note='Local raster phase correlation, NOT building accuracy or independent georeferencing error',
                old_candidates=int(old.geometry.representative_point().map(block.geometry.covers).sum()),
                new_candidates=int(new.geometry.representative_point().map(block.geometry.covers).sum())))
            cv2.imwrite(str(out / f'{block.block_id}_comparison.png'), np.hstack([
                draw(gray, old[old.intersects(block.geometry.buffer(25))],tr,'v1 extraction on v2 (misaligned)'),
                draw(gray, new[new.intersects(block.geometry.buffer(25))],tr,'v2 extraction on v2 (unreviewed)')]))
            cv2.imwrite(str(out / f'{block.block_id}_source.png'), gray)
            profile = current.profile.copy()
            profile.update(driver='GTiff',height=h,width=w,transform=tr,compress='deflate')
            with rasterio.open(out / f'{block.block_id}_v2.tif','w',**profile) as dst:
                dst.write(data)
    with (out / 'timing.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['block_id','workflow','reviewer','status','tracing_seconds','correction_seconds','review_seconds','clicks','notes'])
        for b in blocks.block_id:
            for workflow in ['geosam_from_blank','batch_corrected']:
                writer.writerow([b,workflow,'','not_started','','','','',''])
    completion = {b:dict(complete=False,reviewed_by='',reviewed_at='',target='map_trace',
                        cog_sha256=inputs['candidates_v4_v2']['cog_sha256']) for b in blocks.block_id}
    (out / 'reference_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    (out / 'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__ == '__main__':
    main()
