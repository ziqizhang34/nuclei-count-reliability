#!/usr/bin/env python3
"""Reproduce image-level summaries, density-stratified tables, and figures from per-image CSV rows."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rmse(x):
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def percentile_ci(values, statistic=np.mean, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    reps = np.empty(n_boot)
    for i in range(n_boot):
        reps[i] = statistic(rng.choice(arr, size=len(arr), replace=True))
    return float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2))


def summarize(df):
    rows = []
    for keys, g in df.groupby(['dataset', 'method', 'split'], dropna=False):
        dataset, method, split = keys
        row = {'dataset': dataset, 'method': method, 'split': split,
               'n_images': g['id'].nunique(), 'n_rows': len(g),
               'seeds_completed': g['seed'].nunique() if 'seed' in g else np.nan}
        row['MAE_mean'] = float(g['abs_error'].mean())
        row['RMSE_mean'] = rmse(g['error'])
        row['Bias_mean'] = float(g['error'].mean())
        row['nMAE_mean'] = float(g['abs_error'].sum() / g['gt_count'].sum())
        for metric, col, stat in [
            ('MAE','abs_error',np.mean), ('RMSE','error',lambda x: np.sqrt(np.mean(np.square(x)))),
            ('Bias','error',np.mean), ('Dice','dice',np.mean), ('IoU','iou',np.mean),
            ('GAME0','game0',np.mean), ('GAME1','game1',np.mean), ('GAME2','game2',np.mean), ('GAME4','game4',np.mean)]:
            lo, hi = percentile_ci(g[col], statistic=stat)
            row[f'{metric}_ci95_low'] = lo
            row[f'{metric}_ci95_high'] = hi
            if f'{metric}_mean' not in row:
                row[f'{metric}_mean'] = float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def density_summary(df):
    tmp = df.copy()
    tmp['density_group'] = pd.cut(tmp['gt_count'], bins=[-np.inf, 10, 50, np.inf], labels=['Low (<=10)', 'Medium (11-50)', 'High (>50)'])
    return tmp.groupby('density_group', observed=True).agg(
        n=('id','count'), gt_mean=('gt_count','mean'), pred_mean=('pred_count','mean'),
        mae=('abs_error','mean'), mae_median=('abs_error','median'),
        ae_q1=('abs_error', lambda x: np.quantile(x, 0.25)),
        ae_q3=('abs_error', lambda x: np.quantile(x, 0.75)),
        rmse=('error', rmse), bias=('error','mean'), dice=('dice','mean'), iou=('iou','mean'),
        recall=('instance_recall','mean'), missed=('missed_objects','mean'), merge=('merge_errors','mean')
    ).reset_index()


def make_figures(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 160, 'savefig.dpi': 300, 'font.size': 10})
    labels = ['MAE','RMSE','GAME(0)','GAME(2)','GAME(4)']
    means = [df['abs_error'].mean(), rmse(df['error']), df['game0'].mean(), df['game2'].mean(), df['game4'].mean()]
    fig, ax = plt.subplots(figsize=(6.2,3.6)); ax.bar(labels, means)
    ax.set_ylabel('Error value'); ax.set_title('Global and spatial counting errors'); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/'fig1_global_spatial_errors.png'); plt.close(fig)
    cols = ['missed_objects','false_components','merge_errors','split_errors']
    fig, ax = plt.subplots(figsize=(6.2,3.6)); ax.boxplot([df[c] for c in cols], labels=['Missed\nobjects','False\ncomponents','Merge\nevents','Split\nevents'], showmeans=True)
    ax.set_ylabel('Count per image'); ax.set_title('Failure diagnostics'); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/'fig2_failure_diagnostics_boxplot.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2,5.0)); ax.scatter(df['gt_count'], df['pred_count'], s=18, alpha=0.75)
    maxv=max(df['gt_count'].max(), df['pred_count'].max()); ax.plot([0,maxv],[0,maxv], linestyle='--', linewidth=1)
    ax.set_xlabel('Ground-truth count'); ax.set_ylabel('Predicted count'); ax.set_title('Ground-truth versus predicted counts'); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/'fig3_gt_vs_pred_counts.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2,3.6)); ax.hist(df['error'], bins=24); ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('Prediction error (predicted - ground truth)'); ax.set_ylabel('Number of images'); ax.set_title('Distribution of count errors'); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/'fig4_error_distribution.png'); plt.close(fig)
    tmp=df.copy(); tmp['density_group']=pd.cut(tmp['gt_count'], bins=[-np.inf,10,50,np.inf], labels=['Low (<=10)','Medium (11-50)','High (>50)'])
    groups=['Low (<=10)','Medium (11-50)','High (>50)']
    fig, ax = plt.subplots(figsize=(6.2,3.6)); ax.boxplot([tmp.loc[tmp['density_group'].astype(str)==g,'abs_error'] for g in groups], labels=groups, showmeans=True)
    ax.set_ylabel('Absolute error'); ax.set_title('Density-stratified absolute error'); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/'fig5_density_stratified_abs_error_boxplot.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/raw_run_image_rows.csv')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    out = Path(args.out); tables = out/'tables'; figs = out/'figures'
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    summarize(df).to_csv(tables/'metric_summary_image_level_ci_recomputed.csv', index=False)
    density_summary(df).to_csv(tables/'density_stratified_summary_with_iqr.csv', index=False)
    df.sort_values('abs_error', ascending=False).head(5).to_csv(tables/'top_error_cases.csv', index=False)
    make_figures(df, figs)
    print(f'Wrote results under {out}')

if __name__ == '__main__':
    main()
