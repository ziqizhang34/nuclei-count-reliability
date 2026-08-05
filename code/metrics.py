#!/usr/bin/env python3
"""Shared metrics for nuclei counting experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def foreground_metrics(gt_inst: np.ndarray, pred_lab: np.ndarray) -> dict:
    gt_fg = gt_inst > 0
    pred_fg = pred_lab > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    denom_dice = gt_fg.sum() + pred_fg.sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    return {
        'dice': float(2 * inter / denom_dice) if denom_dice else 1.0,
        'iou': float(inter / union) if union else 1.0,
    }


def label_count(lbl: np.ndarray) -> int:
    vals = np.unique(lbl)
    return int(np.sum(vals > 0))


def centroids(lbl: np.ndarray):
    ids = [int(x) for x in np.unique(lbl) if x > 0]
    out = []
    for i in ids:
        yy, xx = np.nonzero(lbl == i)
        if len(yy):
            out.append((i, float(yy.mean()), float(xx.mean())))
    return out


def game_metric(gt_inst: np.ndarray, pred_lab: np.ndarray, level: int) -> float:
    h, w = gt_inst.shape[:2]
    cells = 2 ** int(level)
    gt_cent = centroids(gt_inst)
    pr_cent = centroids(pred_lab)
    err = 0.0
    for gy in range(cells):
        y0 = int(np.floor(gy * h / cells)); y1 = int(np.floor((gy + 1) * h / cells))
        for gx in range(cells):
            x0 = int(np.floor(gx * w / cells)); x1 = int(np.floor((gx + 1) * w / cells))
            gc = sum(1 for _, y, x in gt_cent if y0 <= y < y1 and x0 <= x < x1)
            pc = sum(1 for _, y, x in pr_cent if y0 <= y < y1 and x0 <= x < x1)
            err += abs(pc - gc)
    return float(err)


def overlap_diagnostics(gt_inst: np.ndarray, pred_lab: np.ndarray, *, iou_threshold: float = 0.10, coverage_threshold: float = 0.10) -> dict:
    """Compute explanatory instance diagnostics from an overlap graph.

    The routine intentionally separates primary one-to-one matching from many-to-one
    and one-to-many explanatory events. It is not a replacement for AP/PQ/AJI.
    """
    gt_ids = [int(x) for x in np.unique(gt_inst) if x > 0]
    pr_ids = [int(x) for x in np.unique(pred_lab) if x > 0]
    n_gt, n_pr = len(gt_ids), len(pr_ids)
    gt_count, pred_count = n_gt, n_pr
    if n_gt == 0 and n_pr == 0:
        return dict(gt_instances=0, pred_components=0, true_positive_instances=0, missed_objects=0,
                    false_components=0, merge_errors=0, split_errors=0, instance_precision=1.0,
                    instance_recall=1.0, mean_matched_iou=1.0)
    if n_gt == 0:
        return dict(gt_instances=0, pred_components=pred_count, true_positive_instances=0, missed_objects=0,
                    false_components=pred_count, merge_errors=0, split_errors=0, instance_precision=0.0,
                    instance_recall=1.0, mean_matched_iou=0.0)
    if n_pr == 0:
        return dict(gt_instances=gt_count, pred_components=0, true_positive_instances=0, missed_objects=gt_count,
                    false_components=0, merge_errors=0, split_errors=0, instance_precision=1.0,
                    instance_recall=0.0, mean_matched_iou=0.0)

    gt_area = {i: int((gt_inst == i).sum()) for i in gt_ids}
    pr_area = {j: int((pred_lab == j).sum()) for j in pr_ids}
    iou_mat = np.zeros((n_gt, n_pr), dtype=float)
    cover_gt = np.zeros((n_gt, n_pr), dtype=float)
    cover_pr = np.zeros((n_gt, n_pr), dtype=float)
    for a, gi in enumerate(gt_ids):
        gm = gt_inst == gi
        for b, pj in enumerate(pr_ids):
            pm = pred_lab == pj
            inter = int(np.logical_and(gm, pm).sum())
            if inter == 0:
                continue
            union = gt_area[gi] + pr_area[pj] - inter
            iou_mat[a, b] = inter / union if union else 0.0
            cover_gt[a, b] = inter / gt_area[gi] if gt_area[gi] else 0.0
            cover_pr[a, b] = inter / pr_area[pj] if pr_area[pj] else 0.0

    # primary one-to-one IoU matching
    cost = -iou_mat
    gi_idx, pj_idx = linear_sum_assignment(cost)
    matched = [(a, b) for a, b in zip(gi_idx, pj_idx) if iou_mat[a, b] >= iou_threshold]
    matched_gt = {a for a, _ in matched}
    matched_pr = {b for _, b in matched}

    graph = (cover_gt >= coverage_threshold) | (cover_pr >= coverage_threshold)
    missed = int(sum(1 for a in range(n_gt) if not graph[a, :].any()))
    false = int(sum(1 for b in range(n_pr) if not graph[:, b].any()))
    merge_events = int(sum(max(0, int(graph[:, b].sum()) - 1) for b in range(n_pr)))
    split_events = int(sum(max(0, int(graph[a, :].sum()) - 1) for a in range(n_gt)))
    tp = int(len(matched))
    return dict(
        gt_instances=gt_count,
        pred_components=pred_count,
        true_positive_instances=tp,
        missed_objects=missed,
        false_components=false,
        merge_errors=merge_events,
        split_errors=split_events,
        instance_precision=float(tp / pred_count) if pred_count else 1.0,
        instance_recall=float(tp / gt_count) if gt_count else 1.0,
        mean_matched_iou=float(np.mean([iou_mat[a, b] for a, b in matched])) if matched else 0.0,
    )
