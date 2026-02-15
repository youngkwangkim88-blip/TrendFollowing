#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from invest_v2.core.types import EntryRuleType, TrailingStopType, PyramidingType, TradeMode
from invest_v2.data_loader import load_ohlc_auto, load_market_csv
from invest_v2.prep import add_indicators
from invest_v2.backtest.engine import BacktestConfig, SingleSymbolBacktester
from invest_v2.reporting.plots import plot_equity_curve


SYMBOL = "005930"  # Toy scope: fixed to Samsung Electronics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Toy backtest (fixed symbol=005930). Supports both old single-symbol CSV and new panel CSV.\n"
            "- Old: date,open,high,low,close,(volume)\n"
            "- Panel: 이름,날짜,ticker,O,H,L,C,V"
        )
    )

    # Data / IO
    p.add_argument(
        "--csv",
        type=str,
        default="data/krx100_adj_5000.csv",
        help="Input CSV path. Default: data/krx100_adj_5000.csv",
    )
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--plot", action="store_true", help="Save equity curve plot.")
    p.add_argument("--market-csv", type=str, default=None, help="Optional KOSPI200 futures/index CSV for Filter C.")
    p.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    # Entry rule
    p.add_argument(
        "--entry-rule",
        type=str,
        default=EntryRuleType.A_TURTLE.value,
        choices=[e.value for e in EntryRuleType],
        help=(
            "Entry rule type. Default A_TURTLE implements Strategy A = A.1(20)+A.2(PL filter)+A.3(55 override). "
            "Other options are kept for debugging." 
        ),
    )

    # Capital / sizing
    p.add_argument("--initial-capital", type=float, default=700_000_000.0)
    p.add_argument(
        "--one-trading-risk",
        type=float,
        default=0.01,
        help="Risk fraction used for unit sizing. Default 0.01 (=1%).",
    )
    p.add_argument("--max-units", type=int, default=4, help="Max units per symbol. Default 4.")
    p.add_argument("--max-units-total", type=int, default=10, help="Portfolio max abs units. Default 10 (toy uses single symbol).")
    p.add_argument(
        "--sell-cost-rate",
        type=float,
        default=0.003,
        help="Sell-side cost rate. Default 0.003 (=0.3%).",
    )

    # Short constraints (toy keeps the knobs; you can disable short entirely)
    # Default follows spec (short allowed subject to constraints).
    # Use --disable-short to enforce long-only.
    p.add_argument("--disable-short", action="store_true", help="Disable short trades (long-only).")
    p.add_argument(
        "--short-notional-limit",
        type=float,
        default=570_000_000.0,
        help="Short notional limit (basis). Default 570,000,000 KRW.",
    )
    p.add_argument(
        "--annual-short-interest-rate",
        type=float,
        default=0.045,
        help="Annual short interest rate. Default 0.045 (=4.5%).",
    )
    p.add_argument(
        "--short-max-hold-days",
        type=int,
        default=90,
        help="Max short hold days before forced cover. Default 90.",
    )

    # Stops / trailing stop (hyperparameters)
    p.add_argument(
        "--stop-atr-mult",
        type=float,
        default=2.0,
        help="Initial stop distance in ATR10 multiples. Default 2.0.",
    )
    p.add_argument(
        "--ts-activate-gain",
        type=float,
        default=0.20,
        help="TS activation threshold (gain). Default 0.20 (=20%).",
    )
    p.add_argument(
        "--ts-floor-gain",
        type=float,
        default=0.10,
        help="TS floor relative to entry price. Default 0.10 (=10%).",
    )
    p.add_argument(
        "--ts-trail-frac",
        type=float,
        default=0.10,
        help="TS trailing fraction from Hmax/Lmin. Default 0.10 (=10%).",
    )

    # Filters
    p.add_argument("--filter-pl", action="store_true", help="Filter A: PL filter (block opposite entry after a winning trade).")
    p.add_argument("--filter-cycle", action="store_true", help="Filter B: EMA(5/20/40) cycle filter on the ticker.")
    p.add_argument("--filter-market-cycle", action="store_true", help="Filter C: EMA(5/20/40) cycle filter on KOSPI200 futures/index.")

    # TS type
    p.add_argument(
        "--ts-type",
        type=str,
        default=TrailingStopType.A_PCT.value,
        choices=[e.value for e in TrailingStopType],
        help="Trailing stop type. Default TS.A (% trailing).",
    )

    # Pyramiding type
    p.add_argument(
        "--pyramiding-type",
        type=str,
        default=PyramidingType.A_PCT.value,
        choices=[e.value for e in PyramidingType],
        help="Pyramiding type. Default PRMD.A (% pyramiding).",
    )
    p.add_argument("--disable-pyramiding", action="store_true", help="Disable pyramiding (forces pyramiding-type=OFF).")
    p.add_argument("--pyramid-trigger", type=float, default=0.15, help="PRMD.A trigger. Default 0.15 (=15%).")
    p.add_argument("--pyramid-cooldown-days", type=int, default=5, help="PRMD.B cooldown days. Default 5.")

    return p.parse_args()
 


def main() -> None:
    args = parse_args()

    trade_mode = TradeMode.LONG_ONLY if bool(args.disable_short) else TradeMode.LONG_SHORT

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    market_df = None
    if args.market_csv:
        market_df = load_market_csv(args.market_csv, symbol=None)

    df = load_ohlc_auto(args.csv, symbol=SYMBOL)
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    df = add_indicators(df, market_df=market_df)

    # Save plot data (for plotly/matlab viewers)
    (outdir / "plot_data.csv").write_text(df.to_csv(index=True, index_label="date"), encoding="utf-8")

    cfg = BacktestConfig(
        symbol=SYMBOL,
        initial_capital=float(args.initial_capital),
        one_trading_risk=float(args.one_trading_risk),
        max_units_per_symbol=int(args.max_units),
        max_units_total=int(args.max_units_total),
        short_notional_limit=float(args.short_notional_limit),
        entry_rule=EntryRuleType(args.entry_rule),
        trade_mode=trade_mode,

        filter_pl=bool(args.filter_pl),
        filter_cycle=bool(args.filter_cycle),
        filter_market_cycle=bool(args.filter_market_cycle),
        stop_atr_mult=float(args.stop_atr_mult),
        ts_activate_gain=float(args.ts_activate_gain),
        ts_floor_gain=float(args.ts_floor_gain),
        ts_trail_frac=float(args.ts_trail_frac),
        ts_type=TrailingStopType(args.ts_type),

        pyramiding_type=(PyramidingType.OFF if bool(args.disable_pyramiding) else PyramidingType(args.pyramiding_type)),
        pyramid_trigger=float(args.pyramid_trigger),
        pyramid_cooldown_days=int(args.pyramid_cooldown_days),
        sell_cost_rate=float(args.sell_cost_rate),
        annual_short_interest_rate=float(args.annual_short_interest_rate),
        short_max_hold_days=int(args.short_max_hold_days),
    )

    bt = SingleSymbolBacktester(df=df, cfg=cfg)
    res = bt.run()

    (outdir / "equity_curve.csv").write_text(res.equity_curve.to_csv(index=True), encoding="utf-8")
    (outdir / "trades.csv").write_text(res.trades.to_csv(index=False), encoding="utf-8")
    (outdir / "fills.csv").write_text(res.fills.to_csv(index=False), encoding="utf-8")
    (outdir / "trader_report.txt").write_text(bt.trader.how_did_you_trade(max_lines=5000), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(res.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=== SUMMARY ===")
    print(json.dumps(res.summary, ensure_ascii=False, indent=2))

    if args.plot:
        plot_equity_curve(res.equity_curve, str(outdir / "equity_curve.png"))
        print(f"Saved plot: {outdir / 'equity_curve.png'}")

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
