#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invest_v2.core.types import EntryRuleType, TrailingStopType, PyramidingType
from invest_v2.data_loader import load_ohlc_auto, load_market_ohlc_auto
from invest_v2.prep import add_indicators
from invest_v2.backtest.engine import BacktestConfig, SingleSymbolBacktester
from invest_v2.reporting.plots import plot_bundle_abc_year_chunks


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
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save bundled Plot A/B/C as PNGs, chunked by N years.",
    )
    p.add_argument(
        "--plot-years",
        type=int,
        default=4,
        help="If --plot, split plots into N-year chunks. Default 4.",
    )
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

    # Market input (Strategy C / cycle filter)
    p.add_argument(
        "--market-csv",
        type=str,
        default=None,
        help="(Optional) Market index/futures CSV used for global regime filter (e.g., KOSPI200 futures).",
    )
    p.add_argument(
        "--market-ticker",
        type=str,
        default=None,
        help="(Optional) If market CSV is panel format with 'ticker', select this ticker.",
    )
    p.add_argument(
        "--c-no-market-filter",
        action="store_true",
        help="(Strategy C) Disable market phase filter (stock phase filter still applies).",
    )

    # Strategy C knobs
    p.add_argument("--c-mom-window", type=int, default=63, help="(Strategy C) Momentum lookback. Default 63.")
    p.add_argument("--c-enter-thr", type=float, default=0.05, help="(Strategy C) Entry threshold. Default 0.05 (=+5%%).")
    p.add_argument("--c-exit-thr", type=float, default=0.0, help="(Strategy C) Exit threshold. Default 0.0.")
    p.add_argument("--c-ema-short", type=int, default=5, help="(Strategy C) Short EMA for cycle phase. Default 5.")
    p.add_argument("--c-ema-mid", type=int, default=20, help="(Strategy C) Mid EMA for cycle phase. Default 20.")
    p.add_argument("--c-ema-long", type=int, default=40, help="(Strategy C) Long EMA for cycle phase. Default 40.")
    p.add_argument(
        "--c-allowed-phases-long",
        type=str,
        default="6,1,2",
        help="(Strategy C) Allowed phases for LONG. Default '6,1,2'.",
    )

    # Capital / sizing
    p.add_argument("--initial-capital", type=float, default=700_000_000.0)
    p.add_argument(
        "--one-trading-risk",
        type=float,
        default=0.01,
        help="Risk fraction used for unit sizing. Default 0.01 (=1%%).",
    )
    p.add_argument("--max-units", type=int, default=4, help="Max units per symbol. Default 4.")
    p.add_argument("--max-units-total", type=int, default=10, help="Portfolio max abs units. Default 10 (toy uses single symbol).")
    p.add_argument(
        "--sell-cost-rate",
        type=float,
        default=0.003,
        help="Sell-side cost rate. Default 0.003 (=0.3%%).",
    )

    # Short constraints
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
        help="Annual short interest rate. Default 0.045 (=4.5%%).",
    )
    p.add_argument(
        "--short-max-hold-days",
        type=int,
        default=90,
        help="Max short hold days before forced cover. Default 90.",
    )

    # Stops / trailing stop
    p.add_argument(
        "--stop-atr-mult",
        type=float,
        default=2.0,
        help="Initial stop distance in ATR10 multiples. Default 2.0.",
    )
    p.add_argument(
        "--ts-type",
        type=str,
        default=TrailingStopType.TS_A.value,
        choices=[e.value for e in TrailingStopType],
        help="Trailing stop type: TS.A (pct), TS.B (EMA cross), TS.C (Darvas box).",
    )
    # TS.A params
    p.add_argument("--ts-activate-gain", type=float, default=0.20, help="(TS.A) activation gain. Default 0.20 (=20%%).")
    p.add_argument("--ts-floor-gain", type=float, default=0.10, help="(TS.A) floor gain. Default 0.10 (=10%%).")
    p.add_argument("--ts-trail-frac", type=float, default=0.10, help="(TS.A) trailing fraction. Default 0.10 (=10%%).")

    # TS.C / Darvas box window (also used by PRMD.B)
    p.add_argument(
        "--box-window",
        type=int,
        default=20,
        help="(TS.C / PRMD.B) Darvas/box lookback window. Default 20.",
    )

    # Pyramiding
    p.add_argument("--disable-pyramiding", action="store_true", help="Disable pyramiding.")
    p.add_argument(
        "--pyramiding-type",
        type=str,
        default=PyramidingType.PRMD_A.value,
        choices=[e.value for e in PyramidingType],
        help="Pyramiding type: PRMD.A (pct), PRMD.B (Darvas box + cooldown).",
    )
    # PRMD.A param
    p.add_argument("--pyramid-trigger", type=float, default=0.15, help="(PRMD.A) pyramiding trigger gain. Default 0.15 (=15%%).")
    # PRMD.B param
    p.add_argument(
        "--prmd-b-cooldown-days",
        type=int,
        default=5,
        help="(PRMD.B) cooldown days after a pyramid add. Default 5.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    enable_short = not bool(args.disable_short)
    enable_pyramiding = not bool(args.disable_pyramiding)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load symbol data (sanitized: 0-valued O/H/L are treated as missing and filled with close)
    df = load_ohlc_auto(args.csv, SYMBOL)

    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    market_df = None
    if args.market_csv:
        market_df = load_market_ohlc_auto(args.market_csv, ticker=args.market_ticker)
        if args.start:
            market_df = market_df[market_df.index >= pd.to_datetime(args.start)]
        if args.end:
            market_df = market_df[market_df.index <= pd.to_datetime(args.end)]

    entry_rule = EntryRuleType(args.entry_rule)
    ts_type = TrailingStopType(args.ts_type)
    prmd_type = PyramidingType(args.pyramiding_type)

    c_use_market_filter = not bool(args.c_no_market_filter)
    if entry_rule == EntryRuleType.C_TSMOM_CYCLE and c_use_market_filter and market_df is None:
        raise ValueError("Strategy C (C_TSMOM_CYCLE) requires --market-csv unless --c-no-market-filter is set.")

    c_allowed_phases_long = tuple(int(x.strip()) for x in str(args.c_allowed_phases_long).split(",") if x.strip())
    cycle_ema_windows = (int(args.c_ema_short), int(args.c_ema_mid), int(args.c_ema_long))

    df = add_indicators(
        df,
        mom_windows=(int(args.c_mom_window),),
        cycle_ema_windows=cycle_ema_windows,
        market_df=market_df,
        market_prefix="mkt",
    )

    cfg = BacktestConfig(
        symbol=SYMBOL,
        initial_capital=float(args.initial_capital),
        one_trading_risk=float(args.one_trading_risk),
        max_units_per_symbol=int(args.max_units),
        max_units_total=int(args.max_units_total),
        short_notional_limit=float(args.short_notional_limit),
        entry_rule=entry_rule,
        enable_short=enable_short,
        # Strategy C params
        c_mom_window=int(args.c_mom_window),
        c_enter_thr=float(args.c_enter_thr),
        c_exit_thr=float(args.c_exit_thr),
        c_cycle_ema_short=int(args.c_ema_short),
        c_cycle_ema_mid=int(args.c_ema_mid),
        c_cycle_ema_long=int(args.c_ema_long),
        c_allowed_phases_long=c_allowed_phases_long,
        c_use_market_filter=c_use_market_filter,
        c_market_prefix="mkt",
        # Stops / TS
        stop_atr_mult=float(args.stop_atr_mult),
        ts_type=ts_type,
        ts_activate_gain=float(args.ts_activate_gain),
        ts_floor_gain=float(args.ts_floor_gain),
        ts_trail_frac=float(args.ts_trail_frac),
        ts_b_fast=5,
        ts_b_slow=20,
        ts_c_box_window=int(args.box_window),
        # Pyramiding
        enable_pyramiding=enable_pyramiding,
        pyramiding_type=prmd_type,
        pyramid_trigger=float(args.pyramid_trigger),
        prmd_b_box_window=int(args.box_window),
        prmd_b_cooldown_days=int(args.prmd_b_cooldown_days),
        # Costs / short
        sell_cost_rate=float(args.sell_cost_rate),
        annual_short_interest_rate=float(args.annual_short_interest_rate),
        short_max_hold_days=int(args.short_max_hold_days),
    )

    bt = SingleSymbolBacktester(df=df, cfg=cfg)
    res = bt.run()

    (outdir / "equity_curve.csv").write_text(res.equity_curve.to_csv(index=True), encoding="utf-8")
    (outdir / "trades.csv").write_text(res.trades.to_csv(index=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(res.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== SUMMARY ===")
    print(json.dumps(res.summary, ensure_ascii=False, indent=2))

    if args.plot:
        saved = plot_bundle_abc_year_chunks(
            df,
            res.equity_curve,
            res.trades,
            symbol=SYMBOL,
            entry_rule=cfg.entry_rule.value,
            cfg=cfg,
            market_prefix=str(getattr(cfg, "c_market_prefix", "mkt")),
            outdir=str(outdir),
            years_per_chunk=int(args.plot_years),
            filename_prefix="plot_ABC",
        )
        for pth in saved:
            print(f"Saved plot: {pth}")

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
