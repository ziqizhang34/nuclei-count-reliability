#!/usr/bin/env python3
"""Run connected-component and U-Net+watershed post-processing ablations.

Required manifest columns:
  id, split, gt_instance_path, and either prob_path or seed-specific probability columns.

Image files may be .npy, .png, .tif, or .tiff. Probability maps should be floating-point arrays
in [0, 1] or integer images that can be normalized to [0, 1]. Ground-truth instance maps must
contain 0 as background and positive integer instance IDs.
"""
from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io, measure, morphology, segmentation, feature
from metrics import foreground_metrics, overlap_diagnostics, label_count, game_metric


def read_array(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        arr = np.load(path)
    else:
        arr = io.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def normalize_prob(arr):
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0 if arr.max() <= 255 else arr / arr.max()
    return np.clip(arr, 0, 1)


def cc_labels(prob, threshold, min_size, connectivity):
    mask = prob >= threshold
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size), connectivity=int(connectivity))
    return measure.label(mask, connectivity=int(connectivity))


def watershed_labels(prob, threshold, min_size, min_distance, connectivity):
    mask = prob >= threshold
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size), connectivity=int(connectivity))
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32)
    dist = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(dist, labels=mask, min_distance=int(min_distance), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    markers = measure.label(markers > 0, connectivity=int(connectivity))
    if markers.max() == 0:
        markers = measure.label(mask, connectivity=int(connectivity))
    labels = segmentation.watershed(-dist, markers, mask=mask)
    labels = morphology.remove_small_objects(labels, min_size=int(min_size))
    return measure.label(labels > 0, connectivity=int(connectivity))


def eval_one(gt_inst, pred_lab):
    gt_count = label_count(gt_inst)
    pred_count = label_count(pred_lab)
    error = pred_count - gt_count
    row = dict(gt_count=gt_count, pred_count=pred_count, error=error, abs_error=abs(error), sq_error=error*error)
    row.update(foreground_metrics(gt_inst, pred_lab))
    for lev in [0, 1, 2, 4]:
        row[f'game{lev}'] = game_metric(gt_inst, pred_lab, lev)
    row.update(overlap_diagnostics(gt_inst, pred_lab))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='CSV with id, split, gt_instance_path, and probability map paths')
    ap.add_argument('--out', default='results/postprocessing_ablation')
    ap.add_argument('--prob-column', default='prob_path', help='Column containing probability map path. For seed-specific runs use e.g. unet_prob_seed42_path')
    ap.add_argument('--thresholds', nargs='+', type=float, default=[0.50])
    ap.add_argument('--min-sizes', nargs='+', type=int, default=[20])
    ap.add_argument('--min-distances', nargs='+', type=int, default=[5])
    ap.add_argument('--connectivity', type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, m in manifest.iterrows():
        if not str(m.get(args.prob_column, '')).strip():
            continue
        gt = read_array(m['gt_instance_path']).astype(np.int32)
        prob = normalize_prob(read_array(m[args.prob_column]))
        for th, ms in itertools.product(args.thresholds, args.min_sizes):
            lab = cc_labels(prob, th, ms, args.connectivity)
            r = eval_one(gt, lab); r.update(id=m['id'], split=m.get('split','test'), method='unet_cc', threshold=th, min_size=ms, min_distance=np.nan)
            rows.append(r)
            for md in args.min_distances:
                lab = watershed_labels(prob, th, ms, md, args.connectivity)
                r = eval_one(gt, lab); r.update(id=m['id'], split=m.get('split','test'), method='unet_watershed', threshold=th, min_size=ms, min_distance=md)
                rows.append(r)
    if not rows:
        raise SystemExit('No rows were evaluated. Check manifest paths and --prob-column.')
    df = pd.DataFrame(rows)
    df.to_csv(out/'postprocessing_ablation_per_image.csv', index=False)
    summary = df.groupby(['method','threshold','min_size','min_distance'], dropna=False).agg(
        n=('id','count'), mae=('abs_error','mean'), rmse=('error', lambda x: np.sqrt(np.mean(np.square(x)))),
        bias=('error','mean'), dice=('dice','mean'), iou=('iou','mean'), recall=('instance_recall','mean'),
        missed=('missed_objects','mean'), merge=('merge_errors','mean')
    ).reset_index().sort_values(['mae','merge'])
    summary.to_csv(out/'postprocessing_ablation_summary.csv', index=False)
    print(f'Wrote {out}')
    print(summary.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
