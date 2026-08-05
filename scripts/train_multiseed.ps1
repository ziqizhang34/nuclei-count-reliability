param(
  [string]$DataDir = "data/converted/bbbc038",
  [string]$OutRoot = "runs/unet"
)

$ErrorActionPreference = "Stop"
foreach ($Seed in 42, 123, 2024) {
  nuclei-train `
    --data-dir $DataDir `
    --out-dir "$OutRoot/seed_$Seed" `
    --epochs 100 `
    --batch-size 4 `
    --learning-rate 0.0001 `
    --weight-decay 0.01 `
    --seed $Seed
}
