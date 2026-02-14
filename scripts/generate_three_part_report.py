#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invest_v2.reporting.three_part_report import write_three_part_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 3-part report from existing combo result CSV files.")
    p.add_argument("--out-dir", required=True, help="Output experiment directory, e.g. outputs/combo_005930_fullgrid_v1")
    p.add_argument("--results-long-csv", required=True, help="Path to results_long_only.csv")
    p.add_argument("--results-short-csv", required=True, help="Path to results_short_only.csv")
    p.add_argument("--top-k", type=int, default=10, help="Top K for overlap table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = write_three_part_report(
        out_dir=args.out_dir,
        results_long_csv=args.results_long_csv,
        results_short_csv=args.results_short_csv,
        top_k=int(args.top_k),
    )
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
