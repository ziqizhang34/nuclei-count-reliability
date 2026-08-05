# Reproducibility notes

## Evidence included in this repository

The three `eval_unet_seg_test.csv` files contain one row per test image and seed. They support the reported count, overlap, spatial, and instance-level summaries without requiring the model checkpoints.

The fixed split manifest contains all 670 BBBC038 sample IDs and their train/validation/test assignments. The test split contains exactly 100 IDs.

## Table 5

Merge/split proxies depend on the overlap coverage threshold. The published values for 0.05, 0.10, and 0.20 are committed in `results/tables/table5_proxy_threshold_sensitivity.csv`.

To regenerate those values from first principles, evaluate each checkpoint with `--save-artifacts`, then run `nuclei_counting.proxy_sensitivity` using the saved component labels and converted GT instance maps.

## Small-mask filtering and foreground overlap

The cleaned evaluation code computes Dice and IoU after the minimum-area filter, matching the stated operating point. The originally archived CSVs were generated with the thresholded mask before the area filter for foreground overlap, while counting and instance diagnostics used the filtered components. The difference is below 0.0003 in mean Dice/IoU and does not change the manuscript's three-decimal values. This distinction is documented here rather than hidden.

## Checkpoints

The original checkpoints are deliberately not committed. They can be released separately if the authors choose. A checkpoint is not needed to regenerate the paper tables from the included per-image CSVs, but it is needed for fresh inference and alternative post-processing.
