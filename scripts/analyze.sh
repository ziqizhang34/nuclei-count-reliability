#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-results/per_seed}"
OUT_DIR="${2:-results/generated}"

nuclei-analyze \
  --results-root "$RESULTS_ROOT" \
  --out-dir "$OUT_DIR" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 42
