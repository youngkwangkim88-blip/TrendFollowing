#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from invest_v2.core.types import EntryRuleType
from invest_v2.data_loader import load_ohlc_csv
from invest_v2.prep import add_indicators
from invest_v2.backtest.engine import BacktestConfig, SingleSymbolBacktester
from invest_v2.reporting.plots import plot_equity_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy backtest for a single KRX symbol (default: 005930).")
    p.add_argument("--csv", type=str, required=True, help="CSV with columns: date,open,high,low,close,(volume)")
    p.add_argument("--symbol", type=str, default="005930")
    p.add_argument("--entry-rule", type=str, default="A_20_PL", choices=[e.value for e in EntryRuleType])
    p.add_argument("--initial-capital", type=float, default=700_000_000.0)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_ohlc_csv(args.csv)
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    df = add_indicators(df)

    cfg = BacktestConfig(
        symbol=args.symbol,
        initial_capital=float(args.initial_capital),
        entry_rule=EntryRuleType(args.entry_rule),
    )
    bt = SingleSymbolBacktester(df=df, cfg=cfg)
    res = bt.run()

    (outdir / "equity_curve.csv").write_text(res.equity_curve.to_csv(index=True), encoding="utf-8")
    (outdir / "trades.csv").write_text(res.trades.to_csv(index=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(res.summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== SUMMARY ===")
    print(json.dumps(res.summary, ensure_ascii=False, indent=2))

    if args.plot:
        plot_equity_curve(res.equity_curve, str(outdir / "equity_curve.png"))
        print(f"Saved plot: {outdir / 'equity_curve.png'}")

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
