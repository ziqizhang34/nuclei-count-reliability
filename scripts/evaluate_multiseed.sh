#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/converted/bbbc038}"
RUN_ROOT="${2:-runs/unet}"
OUT_ROOT="${3:-results/reproduced}"

for SEED in 42 123 2024; do
  nuclei-evaluate \
    --data-dir "$DATA_DIR" \
    --weights "$RUN_ROOT/seed_$SEED/best_unet_seg.pt" \
    --out-dir "$OUT_ROOT/seed_$SEED" \
    --split test \
    --threshold 0.50 \
    --min-area 8 \
    --connectivity 2 \
    --iou-match-threshold 0.50 \
    --proxy-coverage-threshold 0.10
done
