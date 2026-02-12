#!/usr/bin/env python3
"""Paper runner (daily, single-symbol) — scaffold.

This runner demonstrates the wiring from:
  data (EOD) -> signal -> unit sizing -> broker market order

It intentionally keeps portfolio/accounting reconciliation minimal.
"""

from __future__ import annotations

import argparse

import pandas as pd

from invest_v2.core.types import EntryRuleType, Side
from invest_v2.prep import add_indicators
from invest_v2.strategy.entry_rules import build_entry_rule, EntryContext
from invest_v2.backtest.engine import BacktestConfig
from invest_v2.live.kis.settings import KISSettings
from invest_v2.live.kis.paper import KISPaperBroker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/krx100_adj_5000.csv")
    p.add_argument("--symbol", type=str, default="005930")
    p.add_argument("--entry-rule", type=str, default=EntryRuleType.A_TURTLE.value, choices=[e.value for e in EntryRuleType])
    p.add_argument("--initial-capital", type=float, default=700_000_000.0)
    p.add_argument("--dry", action="store_true", help="Only print intended order")
    return p.parse_args()


def load_panel(csv_path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"ticker": str}, low_memory=False)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    sym = str(symbol).zfill(6)
    df = df[df["ticker"] == sym].copy()
    df["date"] = pd.to_datetime(df["날짜"])
    df = df.rename(columns={"O": "open", "H": "high", "L": "low", "C": "close"})
    df = df[["date", "open", "high", "low", "close"]].set_index("date").sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


def main() -> None:
    args = parse_args()

    df = load_panel(args.csv, args.symbol)
    df = add_indicators(df)

    cfg = BacktestConfig(symbol=str(args.symbol).zfill(6), initial_capital=float(args.initial_capital), entry_rule=EntryRuleType(args.entry_rule))
    entry_rule = build_entry_rule(cfg.entry_rule)
    ctx = EntryContext()

    i = len(df) - 1
    sig = entry_rule.evaluate(df, i, ctx)
    if sig == Side.FLAT:
        print("No entry signal today.")
        return

    atr_ref = float(df["atr10"].iloc[i])
    unit_shares = int((cfg.one_trading_risk * cfg.initial_capital) // atr_ref) if atr_ref > 0 else 0
    if unit_shares <= 0:
        print("ATR not available; skip.")
        return

    side = "buy" if sig == Side.LONG else "sell"
    print(f"Signal={sig.name}, unit_shares={unit_shares}, side={side}")

    if args.dry:
        return

    settings = KISSettings.from_env()
    broker = KISPaperBroker(settings)
    oid = broker.place_market_order(args.symbol, side, unit_shares)
    print(f"placed order_id={oid}")


if __name__ == "__main__":
    main()
