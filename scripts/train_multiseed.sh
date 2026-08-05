#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/converted/bbbc038}"
OUT_ROOT="${2:-runs/unet}"

for SEED in 42 123 2024; do
  nuclei-train \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_ROOT/seed_$SEED" \
    --epochs 100 \
    --batch-size 4 \
    --learning-rate 0.0001 \
    --weight-decay 0.01 \
    --seed "$SEED"
done
