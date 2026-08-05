param(
  [string]$ResultsRoot = "results/per_seed",
  [string]$OutDir = "results/generated"
)

$ErrorActionPreference = "Stop"
nuclei-analyze `
  --results-root $ResultsRoot `
  --out-dir $OutDir `
  --bootstrap-resamples 10000 `
  --bootstrap-seed 42
