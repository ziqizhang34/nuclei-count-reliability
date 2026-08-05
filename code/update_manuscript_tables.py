#!/usr/bin/env python3
"""Create manuscript-ready Markdown tables from ablation summary CSVs."""
from pathlib import Path
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True, help='postprocessing_ablation_summary.csv')
    ap.add_argument('--out', default='results/tables/ablation_table.md')
    args = ap.parse_args()
    df = pd.read_csv(args.summary).sort_values(['mae','merge']).head(12)
    cols = ['method','threshold','min_size','min_distance','n','mae','rmse','bias','dice','iou','recall','missed','merge']
    table = df[cols].to_markdown(index=False, floatfmt='.3f')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table, encoding='utf-8')
    print(table)

if __name__ == '__main__':
    main()
