#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from invest_v2.core.types import EntryRuleType
from invest_v2.prep import add_indicators
from invest_v2.backtest.engine import BacktestConfig, SingleSymbolBacktester
from invest_v2.reporting.plots import plot_equity_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy backtest for a single KRX symbol (default: 005930). "
        "Supports both (old) single-symbol CSV and (new) panel CSV format."
    )
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help=(
            "CSV path.\n"
            " - Old format: date,open,high,low,close,(volume)\n"
            " - New panel format: 이름,날짜,ticker,O,H,L,C,V"
        ),
    )
    p.add_argument("--symbol", type=str, default="005930", help="Ticker (6-digit).")
    p.add_argument("--entry-rule", type=str, default="A_20_PL", choices=[e.value for e in EntryRuleType])
    p.add_argument("--initial-capital", type=float, default=700_000_000.0)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    return p.parse_args()


def load_ohlc_auto(csv_path: str, symbol: str) -> pd.DataFrame:
    """
    Auto-detect CSV schema and return OHLC(V) DataFrame indexed by datetime.
    Output columns: open, high, low, close, (volume)
    """
    p = Path(csv_path)
    df = pd.read_csv(p, dtype={"ticker": str}, low_memory=False)

    # Case A) New panel format: 이름, 날짜, ticker, O,H,L,C,V
    panel_cols = {"날짜", "ticker", "O", "H", "L", "C"}
    if panel_cols.issubset(df.columns):
        sym = str(symbol).zfill(6)
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df = df[df["ticker"] == sym].copy()
        if df.empty:
            raise ValueError(f"No rows found for symbol={sym} in panel CSV: {csv_path}")

        df["date"] = pd.to_datetime(df["날짜"])
        rename_map = {"O": "open", "H": "high", "L": "low", "C": "close", "V": "volume"}
        df = df.rename(columns=rename_map)

        needed = ["date", "open", "high", "low", "close"]
        if not set(needed).issubset(df.columns):
            raise ValueError(f"Panel CSV missing required columns after rename. Got: {list(df.columns)}")

        keep = ["date", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
        df = df[keep].copy()
        df = df.set_index("date").sort_index()

        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        return df

    # Case B) Old single-symbol format: date,open,high,low,close,(volume)
    old_cols = {"date", "open", "high", "low", "close"}
    if old_cols.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df

    raise ValueError(
        "Unrecognized CSV schema. "
        "Expected either old format (date,open,high,low,close,...) or new panel format (이름,날짜,ticker,O,H,L,C,V). "
        f"Got columns: {list(df.columns)}"
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_ohlc_auto(args.csv, args.symbol)

    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    df = add_indicators(df)

    cfg = BacktestConfig(
        symbol=str(args.symbol).zfill(6),
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
