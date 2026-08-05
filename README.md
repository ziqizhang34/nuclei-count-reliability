# Component-count-aware reliability diagnosis for U-Net nuclei counting


> **Component-count-aware reliability diagnosis for U-Net-based nuclei counting in optical microscopy images**

The study audits how U-Net foreground masks become nuclei counts after thresholding, small-component filtering, connected-component labeling, and one-to-one instance matching. The reported experiment uses a fixed 100-image BBBC038 test subset and three training seeds: 42, 123, and 2024.

## What is included

- The compact U-Net architecture and BCE-plus-Dice training code.
- BBBC038 conversion code using the fixed split manifest committed in `data/splits/`.
- Evaluation code for count error, Dice, IoU, spatial GAME metrics, missed/false components, merge/split proxies, and instance recall.
- Optional export of probability maps, binary masks, component labels, and overlap-edge logs.
- Per-image evaluation CSVs and training histories for all three reported seeds.
- Analysis code that regenerates Tables 1-4 and 6 and Figures 1-3.
- The published proxy-threshold sensitivity values in Table 5, plus a script to recompute them from saved component-label artifacts.

## What is intentionally not included

- Raw BBBC038 images or third-party dataset archives.
- Converted image arrays.
- PyTorch checkpoints.
- MCNN, CSRNet, BBBC039, NuInsSeg, synthetic-demo, pilot, watershed-ablation, or old cross-paradigm benchmark code, because none of those are part of the final manuscript.
- Earlier manuscript drafts, review-response notes, versioned figures, debug runs, and duplicate unseeded results.

This keeps the repository small and prevents old experimental branches from being mistaken for the evidence used in the paper.

## Quick reproduction of published tables and figures

```bash
conda env create -f environment.yml
conda activate nuclei-count-reliability
pip install -e .

nuclei-analyze \
  --results-root results/per_seed \
  --out-dir results/generated
```

Generated files will appear under:

```text
results/generated/tables/
results/generated/figures/
```

## Full rerun from the official BBBC038 archive

1. Download the official BBBC038 `stage1_train.zip` archive.
2. Convert it using the committed fixed split manifest:

```bash
nuclei-prepare \
  --zip-path /path/to/stage1_train.zip \
  --out-dir data/converted/bbbc038 \
  --split-manifest data/splits/bbbc038_fixed_split.csv
```

3. Train the three seeds:

```bash
bash scripts/train_multiseed.sh data/converted/bbbc038 runs/unet
```

4. Evaluate each checkpoint:

```bash
bash scripts/evaluate_multiseed.sh data/converted/bbbc038 runs/unet results/reproduced
```

To retain first-principles audit artifacts, add `--save-artifacts` to the evaluation command. Probability maps and component labels can be large, so they are ignored by Git by default.

## Reported operating point

The committed configuration in `configs/paper.yaml` records the reported settings:

- probability threshold: 0.50
- minimum component area: 8 pixels
- instance-match IoU threshold: 0.50
- primary merge/split coverage threshold: 0.10
- sensitivity thresholds: 0.05, 0.10, and 0.20
- bootstrap resamples: 10,000
- training seeds: 42, 123, and 2024

## Repository scope and verification

The included per-seed evaluation CSVs reproduce the manuscript's seed-level MAE, RMSE, bias, nMAE, Dice, IoU, instance-recall, missed-object, false-component, and merge/split summaries. Table 5 requires overlap information under alternative proxy thresholds; its published values are included, and `proxy_sensitivity.py` can recompute them when component labels and GT instance maps are available.

No open-source license has been selected. Add a license before making the repository public if reuse is intended.
