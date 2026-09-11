"""Score only completed, human-reviewed map-ink references, in UTM metres.

One-to-one maximum-cardinality matching at IoU >= threshold, then maximum
IoU as tie-break. Oversize proposals count as predictions, not silently
excluded. Surviving modern footprints are NOT map-ink ground truth.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from source_config import sha256


def score(pred, reference, threshold=.5):
    ious = np.zeros((len(pred), len(reference)))
    for i,p in enumerate(pred):
        for j,r in enumerate(reference):
            ious[i,j] = p.intersection(r).area / p.union(r).area
    matched = []
    if ious.size:
        valid = ious >= threshold
        # One extra valid match outweighs every possible IoU tie-break.
        weights = valid * (min(ious.shape)+1+ious)
        ii,jj = linear_sum_assignment(weights, maximize=True)
        matched = [(i,j) for i,j in zip(ii,jj) if valid[i,j]]
    tp = len(matched)
    distances=[]
    for i,j in matched:
        a,b = pred[i].boundary,reference[j].boundary
        # Symmetric mean nearest-boundary distance, sampled every <=1m.
        d=[]
        for x,y in [(a,b),(b,a)]:
            n=max(2,int(np.ceil(x.length))+1)
            d.extend(x.interpolate(t,normalized=True).distance(y) for t in np.linspace(0,1,n))
        distances.append(float(np.mean(d)))
    return dict(predictions=len(pred), references=len(reference), tp=tp,
        fp=len(pred)-tp, fn=len(reference)-tp,
        precision=tp/len(pred) if pred else None,
        recall=tp/len(reference) if reference else None,
        matched_iou=[float(ious[i,j]) for i,j in matched],
        matched_mean_boundary_distance_m=distances,
        possible_splits=int(((ious>0.1).sum(axis=0)>1).sum()),
        possible_merges=int(((ious>0.1).sum(axis=1)>1).sum()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--iou',type=float,default=.5)
    args=ap.parse_args()
    if not 0 < args.iou <= 1:
        ap.error('IoU must be in (0,1].')
    d=args.directory
    manifest=json.loads((d/'candidates_v4_v2.manifest.json').read_text())
    if manifest['output_sha256'] != sha256(d/'candidates_v4_v2.gpkg'):
        ap.error('Frozen predictions were modified; use a separate layer for correction.')
    completion=json.loads((d/'reference_completion.json').read_text())
    blocks=gpd.read_file(d/'review.gpkg',layer='benchmark_blocks').to_crs(32618)
    if set(completion) != set(blocks.block_id):
        ap.error('Reference completion must cover every benchmark block.')
    for b,c in completion.items():
        if not (c.get('complete') is True and c.get('reviewed_by','').strip()
                and c.get('reviewed_at','').strip() and c.get('target')=='map_trace'
                and c.get('cog_sha256')==manifest['cog_sha256']):
            ap.error(f'{b}: complete human-reviewed map-trace baseline required; no scores emitted.')
    ref=gpd.read_file(d/'review.gpkg',layer='reference_map_trace').to_crs(32618)
    if not ref.geometry.is_valid.all() or ref.geometry.is_empty.any() or ref.geometry.isna().any():
        ap.error('Reference geometry must be valid and nonempty.')
    for key in ['reference_id','reviewed_by','reviewed_at']:
        if ref[key].isna().any() or ref[key].astype(str).str.strip().eq('').any():
            ap.error(f'Missing reference {key}.')
    if ref.reference_id.duplicated().any():
        ap.error('Duplicate reference IDs.')
    if not (ref.target.eq('map_trace').all() and ref.cog_sha256.eq(manifest['cog_sha256']).all()
            and ref.source_map.eq(manifest['source_map']).all()
            and ref.block_id.isin(blocks.block_id).all()):
        ap.error('Reference target, source, or block assignment mismatch.')
    for _,row in ref.iterrows():
        b=blocks[blocks.block_id==row.block_id].geometry.iloc[0]
        if not b.covers(row.geometry.representative_point()):
            ap.error('Reference must belong to its declared block by representative point.')
    pred=gpd.GeoDataFrame(pd.concat([gpd.read_file(d/'candidates_v4_v2.gpkg',layer=l)
                        for l in ['candidates','oversize_blocks']]),crs=3857).to_crs(32618)
    results={}
    for _,b in blocks.iterrows():
        p=pred[pred.geometry.representative_point().map(b.geometry.covers)]
        r=ref[ref.block_id==b.block_id]
        results[b.block_id]=score(list(p.geometry),list(r.geometry),args.iou)
    totals={k:sum(r[k] for r in results.values()) for k in ['tp','fp','fn']}
    totals['precision']=totals['tp']/(totals['tp']+totals['fp']) if totals['tp']+totals['fp'] else None
    totals['recall']=totals['tp']/(totals['tp']+totals['fn']) if totals['tp']+totals['fn'] else None
    print(json.dumps(dict(target='map_trace',metric_crs='EPSG:32618',iou_threshold=args.iou,
        blocks=results,totals=totals,reference_file_sha256=sha256(d/'review.gpkg'),
        note='Timing is recorded separately in timing.csv; not inferred from geometry.'),indent=2))


if __name__=='__main__':
    main()
