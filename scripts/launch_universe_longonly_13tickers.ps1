$ErrorActionPreference = "Stop"

# --- CPU/threading guard ---
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# --- Recommended workers for i7-14700K (28 threads) ---
# Target ~90% load -> 25 workers.
if (-not $env:WORKERS) { $env:WORKERS = "25" }
if (-not $env:EXPERIMENT) { $env:EXPERIMENT = "universe_longonly_13tickers_v1" }
if (-not $env:CSV) { $env:CSV = "data/krx100_adj_5000.csv" }

python scripts/run_universe_longonly_sweep.py `
  --csv "$env:CSV" `
  --experiment "$env:EXPERIMENT" `
  --workers "$env:WORKERS" `
  --resume `
  --write-abc-topk-per-ticker 3

Write-Host "Done. See outputs/$($env:EXPERIMENT)/report_universe_longonly.html"
