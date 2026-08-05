# Repository file map

- `configs/paper.yaml`: exact study settings.
- `data/splits/bbbc038_fixed_split.csv`: fixed 470/100/100 split.
- `results/per_seed/`: archived per-image evaluation CSVs and training histories.
- `results/tables/table5_proxy_threshold_sensitivity.csv`: published threshold-sensitivity values.
- `src/nuclei_counting/models/unet.py`: reported U-Net.
- `src/nuclei_counting/prepare_bbbc038.py`: dataset conversion.
- `src/nuclei_counting/train_unet.py`: deterministic training.
- `src/nuclei_counting/evaluate_unet.py`: inference and audit artifact generation.
- `src/nuclei_counting/metrics.py`: all metric and proxy definitions.
- `src/nuclei_counting/analyze_reliability.py`: tables and figures.
- `src/nuclei_counting/proxy_sensitivity.py`: alternative proxy thresholds.
- `tests/`: synthetic metric tests and published-number checks.
