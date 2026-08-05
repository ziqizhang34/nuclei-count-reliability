param(
  [string]$DataDir = "data/converted/bbbc038",
  [string]$RunRoot = "runs/unet",
  [string]$OutRoot = "results/reproduced"
)

$ErrorActionPreference = "Stop"
foreach ($Seed in 42, 123, 2024) {
  nuclei-evaluate `
    --data-dir $DataDir `
    --weights "$RunRoot/seed_$Seed/best_unet_seg.pt" `
    --out-dir "$OutRoot/seed_$Seed" `
    --split test `
    --threshold 0.50 `
    --min-area 8 `
    --connectivity 2 `
    --iou-match-threshold 0.50 `
    --proxy-coverage-threshold 0.10
}
