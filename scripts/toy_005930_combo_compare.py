#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

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
from invest_v2.reporting.plotly_abc import write_interactive_abc
from invest_v2.reporting.three_part_report import write_three_part_report

SYMBOL = "005930"


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def _decode_filter_code_bc(code: str) -> Tuple[bool, bool]:
    """
    Filter set codes for combo testing.
      - Filter A (PL) is ALWAYS ON (global default). Not part of the combo dimension.
      - Here we only control:
          B = cycle filter
          C = market cycle filter
    Examples: NONE, B, C, BC
    """
    # v2 default: Filter B(ticker cycle) is treated as always ON.
    # We keep the legacy code string only for backward compatibility of run names.
    c = code.upper()
    cyc = True
    mkt = "C" in c
    return cyc, mkt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combinatorial backtest runner (005930 toy) — 3-part report.")
    p.add_argument("--csv", type=str, default="data/krx100_adj_5000.csv")
    p.add_argument("--market-csv", type=str, default=None, help="Market CSV (needed for Filter C combos).")

    # outputs: outputs/<experiment>/...
    p.add_argument("--out-root", type=str, default="outputs", help="Root output directory (default: outputs).")
    p.add_argument("--experiment", type=str, default="combo_005930_v4", help="Experiment name (subfolder under outputs/).")

    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)

    # dimensions (LONG_SHORT excluded by design)
    p.add_argument("--entry-rules", type=str, default=",".join([
        EntryRuleType.A_TURTLE.value,
        EntryRuleType.B_EMA_CROSS_DC10.value,
    ]))
    # v2 default: Filter B is always ON; only keep "B" / "BC" as meaningful options.
    p.add_argument("--filter-sets", type=str, default="B", help="Filter sets for market-cycle only. Filter B(ticker)=ON is fixed.")
    p.add_argument("--ts-types", type=str, default=",".join([e.value for e in TrailingStopType]))
    p.add_argument("--pyramiding-types", type=str, default=",".join([
        PyramidingType.OFF.value,
        PyramidingType.A_PCT.value,
        PyramidingType.B_DARVAS_BOX.value,
    ]))
    p.add_argument("--minpos-modes", type=str, default="OFF,ON")

    # common knobs
    p.add_argument("--initial-capital", type=float, default=700_000_000.0)
    p.add_argument("--one-trading-risk", type=float, default=0.01)
    p.add_argument("--max-units", type=int, default=4)
    p.add_argument("--max-units-total", type=int, default=10)
    p.add_argument("--sell-cost-rate", type=float, default=0.003)
    p.add_argument("--short-notional-limit", type=float, default=570_000_000.0)
    p.add_argument("--annual-short-interest-rate", type=float, default=0.045)
    p.add_argument("--short-max-hold-days", type=int, default=90)

    p.add_argument("--stop-atr-mult", type=float, default=2.0)
    p.add_argument("--ts-activate-gain", type=float, default=0.20)
    p.add_argument("--ts-floor-gain", type=float, default=0.10)
    p.add_argument("--ts-trail-frac", type=float, default=0.10)
    p.add_argument("--ts-box-window", type=int, default=20)

    p.add_argument("--pyramid-trigger", type=float, default=0.15)
    p.add_argument("--pyramid-cooldown-days", type=int, default=5)

    # report controls
    p.add_argument("--top-k-overlap", type=int, default=10, help="Top-K from each leaderboard for overlap analysis.")

    # progress / safety
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--max-runs", type=int, default=None, help="Optional cap for total runs per mode (debug).")
    p.add_argument("--resume", action="store_true", help="Skip runs that already have summary.json.")

    return p.parse_args()


def _run_sweep(
    *,
    mode: TradeMode,
    df_base: pd.DataFrame,
    market_df: pd.DataFrame | None,
    out_dir: Path,
    entry_rules: List[EntryRuleType],
    filter_codes: List[str],
    ts_types: List[TrailingStopType],
    prmd_types: List[PyramidingType],
    minpos_modes: List[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    runs_dir = out_dir / ("runs_long" if mode == TradeMode.LONG_ONLY else "runs_short")
    runs_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(entry_rules, filter_codes, ts_types, prmd_types, minpos_modes))
    if args.max_runs is not None:
        combos = combos[: max(0, int(args.max_runs))]

    total = len(combos)
    print(f"\n=== Sweep: {mode.value} ===")
    print(f"runs_dir={runs_dir.resolve()}")
    print(f"total_runs={total} (entry={len(entry_rules)} * filters={len(filter_codes)} * ts={len(ts_types)} * prmd={len(prmd_types)} * minpos={len(minpos_modes)})")

    rows: List[Dict] = []
    completed = 0
    skipped = 0

    for idx, (entry_rule, filter_code, ts_type, prmd_type, minpos) in enumerate(combos, start=1):
        # Remove unnecessary LONG_ONLY variants: we only keep the combined Turtle rule (A_TURTLE).
        if mode == TradeMode.LONG_ONLY and entry_rule in (EntryRuleType.A_20_PL, EntryRuleType.A_55):
            continue

        f_cyc, f_mkt = _decode_filter_code_bc(filter_code)

        # Filter C requires market data
        if f_mkt and market_df is None:
            rows.append({
                "mode": mode.value,
                "entry_rule": entry_rule.value,
                "filter_code": filter_code,
                "filter_pl": 1,  # fixed ON
                "filter_cycle": int(f_cyc),
                "filter_market_cycle": int(f_mkt),
                "ts_type": ts_type.value,
                "pyramiding_type": prmd_type.value,
                "minpos": minpos,
                "skipped": 1,
                "skip_reason": "FILTER_C_REQUIRES_MARKET_CSV",
            })
            continue

        run_name = _safe_name(f"{mode.value}__{entry_rule.value}__F{filter_code}__{ts_type.value}__{prmd_type.value}__MIN{minpos}")
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and (run_dir / "summary.json").exists():
            skipped += 1
            if args.print_every and (idx % args.print_every == 0):
                print(f"[{idx}/{total}] RESUME-SKIP | {run_name}")
            continue

        if args.print_every and (idx % args.print_every == 0):
            print(f"[{idx}/{total}] RUN | {run_name}")

        df = df_base.copy()

        cfg = BacktestConfig(
            symbol=SYMBOL,
            initial_capital=float(args.initial_capital),
            one_trading_risk=float(args.one_trading_risk),
            max_units_per_symbol=int(args.max_units),
            max_units_total=int(args.max_units_total),
            sell_cost_rate=float(args.sell_cost_rate),
            short_notional_limit=float(args.short_notional_limit),
            annual_short_interest_rate=float(args.annual_short_interest_rate),
            short_max_hold_days=int(args.short_max_hold_days),
            stop_atr_mult=float(args.stop_atr_mult),
            ts_type=ts_type,
            ts_activate_gain=float(args.ts_activate_gain),
            ts_floor_gain=float(args.ts_floor_gain),
            ts_trail_frac=float(args.ts_trail_frac),
            ts_box_window=int(args.ts_box_window),
            pyramiding_type=prmd_type,
            pyramid_trigger=float(args.pyramid_trigger),
            pyramid_cooldown_days=int(args.pyramid_cooldown_days),
            entry_rule=entry_rule,
            trade_mode=mode,
            # Filters: A fixed ON, B/C are the combo dimensions
            filter_pl=True,
            filter_cycle=bool(f_cyc),
            filter_market_cycle=bool(f_mkt),
            min_position_mode=(minpos.upper() == "ON"),
        )

        bt = SingleSymbolBacktester(df=df, cfg=cfg)
        res = bt.run()

        # save outputs
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
        res.equity_curve.reset_index().rename(columns={"index": "date"}).to_csv(run_dir / "equity_curve.csv", index=False)
        res.trades.to_csv(run_dir / "trades.csv", index=False)
        res.fills.to_csv(run_dir / "fills.csv", index=False)
        (run_dir / "summary.json").write_text(json.dumps(res.summary, indent=2, default=str), encoding="utf-8")
        (run_dir / "trader_report.txt").write_text(bt.trader.how_did_you_trade(max_lines=5000), encoding="utf-8")

        plot_df = df.copy().reset_index().rename(columns={"index": "date"})
        plot_df.to_csv(run_dir / "plot_data.csv", index=False)

        html_path = write_interactive_abc(
            plot_data_csv=run_dir / "plot_data.csv",
            equity_curve_csv=run_dir / "equity_curve.csv",
            trades_csv=run_dir / "trades.csv",
            out_html=run_dir / "abc.html",
            title=run_name,
        )

        row = {
            "mode": mode.value,
            "entry_rule": entry_rule.value,
            "filter_code": filter_code,
            "filter_pl": 1,
            "filter_cycle": int(f_cyc),
            "filter_market_cycle": int(f_mkt),
            "ts_type": ts_type.value,
            "pyramiding_type": prmd_type.value,
            "minpos": minpos.upper(),
            "skipped": 0,
            "cagr": float(res.summary.get("cagr", 0.0)),
            "mdd": float(res.summary.get("mdd", 0.0)),
            "win_rate": float(res.summary.get("win_rate", 0.0)),
            "payoff_ratio": float(res.summary.get("payoff_ratio", 0.0)),
            "num_trades": int(res.summary.get("num_trades", 0)),
            "run_dir": str(run_dir),
            "abc_rel": str(Path(html_path).relative_to(out_dir)),
        }
        rows.append(row)
        completed += 1

        if args.checkpoint_every and (completed % int(args.checkpoint_every) == 0):
            pd.DataFrame(rows).to_csv(out_dir / f"results_{mode.value.lower()}_checkpoint.csv", index=False)

    print(f"done: completed={completed}, skipped={skipped}, total={total}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_root) / _safe_name(args.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    base_df = load_ohlc_auto(args.csv, symbol=SYMBOL)
    if args.start:
        base_df = base_df[base_df.index >= pd.to_datetime(args.start)]
    if args.end:
        base_df = base_df[base_df.index <= pd.to_datetime(args.end)]

    market_df = None
    if args.market_csv:
        market_df = load_market_csv(args.market_csv, symbol=None)

    # indicators
    base_df = add_indicators(base_df, market_df=market_df)

    entry_rules = [EntryRuleType(x) for x in _parse_csv_list(args.entry_rules)]
    filter_codes = _parse_csv_list(args.filter_sets)
    ts_types = [TrailingStopType(x) for x in _parse_csv_list(args.ts_types)]
    prmd_types = [PyramidingType(x) for x in _parse_csv_list(args.pyramiding_types)]
    minpos_modes = [m.strip().upper() for m in _parse_csv_list(args.minpos_modes)]

    # LONG_ONLY
    df_long = _run_sweep(
        mode=TradeMode.LONG_ONLY,
        df_base=base_df,
        market_df=market_df,
        out_dir=out_dir,
        entry_rules=entry_rules,
        filter_codes=filter_codes,
        ts_types=ts_types,
        prmd_types=prmd_types,
        minpos_modes=minpos_modes,
        args=args,
    )
    long_csv = out_dir / "results_long_only.csv"
    df_long.to_csv(long_csv, index=False)

    # SHORT_ONLY
    df_short = _run_sweep(
        mode=TradeMode.SHORT_ONLY,
        df_base=base_df,
        market_df=market_df,
        out_dir=out_dir,
        entry_rules=entry_rules,
        filter_codes=filter_codes,
        ts_types=ts_types,
        prmd_types=prmd_types,
        minpos_modes=minpos_modes,
        args=args,
    )
    short_csv = out_dir / "results_short_only.csv"
    df_short.to_csv(short_csv, index=False)

    report_path = write_three_part_report(
        out_dir=out_dir,
        results_long_csv=long_csv,
        results_short_csv=short_csv,
        top_k=int(args.top_k_overlap),
    )

    print("\n=== Outputs ===")
    print(f"- {long_csv}")
    print(f"- {short_csv}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
