#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

MODELS = ["CNN-LSTM", "Encoder", "Transformer"]
RUNS = 5
MAX_WORKERS = 3

SYMBOLS = (
    "1028,005930,000660,005380,373220,207940,402340,000270,105560,034020,329180,012450,"
    "068270,028260,055550,032830,042660,035420,012330,015760,086790,006800,267260,005490,"
    "006400,316140,009540,000810,035720,034730,010140,138040,009150,051910,298040,064350,"
    "024110,011200,267250,003670,010120,272210,042700,096770,086280,066570,017670,047810"
)


def run_isolated_process(task_info: tuple[str, int]) -> None:
    model, run_id = task_info

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env["TF_NUM_INTRAOP_THREADS"] = "8"
    env["TF_NUM_INTEROP_THREADS"] = "8"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        "scripts/AI_pivot_point_stage5.py",
        "--target-model",
        model,
        "--run-id",
        str(run_id),
        "--panel-csv",
        "data/krx100_adj_5000.csv",
        "--index-csv",
        "data/kospi200_20260212.csv",
        "--symbols",
        SYMBOLS,
        "--index-symbol",
        "1028",
        "--index-weight",
        "0.7",
        "--seq-length",
        "60",
        "--label-order",
        "10",
        "--label-expand",
        "1",
        "--epochs",
        "150",
        "--batch-size",
        "256",
        "--runs",
        str(RUNS),
        "--target-precision",
        "0.30",
    ]

    print(f"[START] {model} - Run {run_id}")
    p = subprocess.run(cmd, env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Failed task: {model} run={run_id}, code={p.returncode}")
    print(f"[DONE]  {model} - Run {run_id}")


def main() -> None:
    temp_dir = Path("outputs/pivotal point/stage5_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(temp_dir / "*.csv")):
        os.remove(f)

    tasks = [(m, r) for m in MODELS for r in range(1, RUNS + 1)]
    print(f"Stage5 parallel run start (workers={MAX_WORKERS}, tasks={len(tasks)})")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_isolated_process, task) for task in tasks]
        for fut in futures:
            fut.result()

    print("\nAll shard runs completed. Merging results...")
    shard_files = glob.glob(str(temp_dir / "*.csv"))
    if not shard_files:
        raise SystemExit("No shard result files found.")

    df = pd.concat((pd.read_csv(f) for f in shard_files), ignore_index=True)
    summary = (
        df.groupby("model", as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            th_mean=("optimal_threshold", "mean"),
            th_std=("optimal_threshold", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
        )
        .round(4)
    )

    out_dir = Path("outputs/pivotal point")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ai_pivot_stage5_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "ai_pivot_stage5_summary.csv", index=False, encoding="utf-8-sig")

    print("\nStage5 Final Summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
