#!/usr/bin/env bash
set -euo pipefail

# --- CPU/threading guard ---
# We run many Python processes (ProcessPool). Prevent BLAS/OMP from spawning
# additional threads per process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Recommended workers for i7-14700K (28 threads) ---
# Target ~90% load -> 25 workers.
WORKERS=${WORKERS:-25}

EXPERIMENT=${EXPERIMENT:-universe_longonly_13tickers_v1}
CSV=${CSV:-data/krx100_adj_5000.csv}

python scripts/run_universe_longonly_sweep.py \
  --csv "$CSV" \
  --experiment "$EXPERIMENT" \
  --workers "$WORKERS" \
  --resume \
  --write-abc-topk-per-ticker 3

echo "Done. See outputs/${EXPERIMENT}/report_universe_longonly.html"
