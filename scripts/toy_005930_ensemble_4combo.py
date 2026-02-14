#!/usr/bin/env python3
"""Run a fixed 4-combo ensemble for 005930 and compute merged CAGR/MDD.

Why this script?
----------------
The previous full-grid runner explores many unnecessary combinations.
For day-to-day iteration we often want to:

1) Run only a small set of pre-selected combos (2 long + 2 short)
2) Combine them under a single-position constraint (no overlap)
3) Report merged CAGR and merged MDD

Notes
-----
- Filter A(PL) is ON.
- Filter B(ticker cycle) is treated as ON in standard runs (see docs/filters.md).
- This script normalizes any legacy combo id tokens to reflect Filter B=ON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import sys

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd

from invest_v2.core.types import EntryRuleType, TrailingStopType, PyramidingType, TradeMode
from invest_v2.data_loader import load_ohlc_auto, load_market_csv
from invest_v2.prep import add_indicators
from invest_v2.backtest.engine import BacktestConfig, SingleSymbolBacktester
from invest_v2.reporting.plotly_abc import write_interactive_abc

SYMBOL = "005930"


# ---- helpers ----
def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def _cagr_from_nav(nav: pd.Series) -> float:
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        return 0.0
    start = float(nav.iloc[0])
    end = float(nav.iloc[-1])
    years = (nav.index[-1] - nav.index[0]).days / 365.0
    if years <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def _mdd_from_nav(nav: pd.Series) -> float:
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        return 0.0
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _load_equity(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "equity_curve.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def _load_trades(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "trades.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _active_mask(eq: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    pos = eq.get("pos_side", pd.Series(index=eq.index, data=0)).fillna(0).astype(int)
    active = (pos != 0) | (pos.shift(1).fillna(0).astype(int) != 0)

    if trades is not None and len(trades):
        dates = set()
        if "entry_date" in trades.columns:
            ed = pd.to_datetime(trades["entry_date"], errors="coerce")
            dates |= set(ed.dropna().dt.normalize().tolist())
        if "exit_date" in trades.columns:
            xd = pd.to_datetime(trades["exit_date"], errors="coerce")
            dates |= set(xd.dropna().dt.normalize().tolist())
        if dates:
            active = active | eq.index.normalize().isin(list(dates))
    return active.astype(bool)


def _merged_no_overlap(
    equity_list: List[pd.DataFrame],
    trades_list: List[pd.DataFrame],
    initial_capital: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Merge multiple strategy equity curves under a single-position constraint.

    Rule (conservative):
      - If exactly one strategy is active on day t, apply that strategy's daily return.
      - If 0 active or >=2 active, apply 0 return.
    """
    if not equity_list:
        raise ValueError("equity_list is empty")
    if len(equity_list) != len(trades_list):
        raise ValueError("equity_list and trades_list must have same length")

    # union calendar
    idx = equity_list[0].index
    for eq in equity_list[1:]:
        idx = idx.union(eq.index)
    idx = idx.sort_values()

    rets: List[pd.Series] = []
    actives: List[pd.Series] = []

    for eq, tr in zip(equity_list, trades_list):
        nav = eq["nav"].astype(float).reindex(idx).ffill()
        r = nav.pct_change().fillna(0.0).astype(float)
        rets.append(r)

        eq2 = eq[["nav", "pos_side"]].copy().reindex(idx).ffill()
        a = _active_mask(eq2, tr).reindex(idx).fillna(False)
        actives.append(a)

    active_count = pd.concat(actives, axis=1).sum(axis=1).astype(int)
    merged_r = pd.Series(0.0, index=idx)
    for k, (r, a) in enumerate(zip(rets, actives)):
        merged_r.loc[(active_count == 1) & a] = r.loc[(active_count == 1) & a]

    merged_nav = float(initial_capital) * (1.0 + merged_r).cumprod()
    merged_mdd = _mdd_from_nav(merged_nav)
    merged_cagr = _cagr_from_nav(merged_nav)

    overlap_days = int((active_count >= 2).sum())
    active_days = int((active_count >= 1).sum())
    total_days = int(len(active_count))
    overlap_pct_total = float(overlap_days / max(1, total_days))
    overlap_pct_active = float(overlap_days / max(1, active_days))

    out = pd.DataFrame({
        "nav": merged_nav,
        "daily_return": merged_r,
        "active_count": active_count,
    }, index=idx)

    stats = {
        "merged_cagr": float(merged_cagr),
        "merged_mdd": float(merged_mdd),
        "overlap_pct_total": float(overlap_pct_total),
        "overlap_pct_active": float(overlap_pct_active),
    }
    return out, stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 4 fixed combos for 005930 and compute merged CAGR/MDD.")
    p.add_argument("--csv", type=str, default="data/krx100_adj_5000.csv")
    p.add_argument("--market-csv", type=str, default=None, help="Optional market CSV for Filter C")

    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--experiment", type=str, default="ensemble_005930_4combo_v1")

    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)

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

    p.add_argument("--show-donchian", action="store_true", help="Overlay Donchian bands in abc.html")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_root) / _safe_name(args.experiment)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ---- load base data ----
    base_df = load_ohlc_auto(args.csv, symbol=SYMBOL)
    if args.start:
        base_df = base_df[base_df.index >= pd.to_datetime(args.start)]
    if args.end:
        base_df = base_df[base_df.index <= pd.to_datetime(args.end)]

    market_df = None
    if args.market_csv:
        market_df = load_market_csv(args.market_csv, symbol=None)

    base_df = add_indicators(base_df, market_df=market_df)

    # ---- define the 4 combos ----
    # NOTE: Filter B(ticker cycle) is treated as ON. We normalize any FNONE -> FB in names.
    requested_ids = [
        "LONG_ONLY__B_EMA_CROSS_DC10__FB__TS_A__PRMD_B__MINOFF",
        "LONG_ONLY__A_TURTLE__FB__TS_A__PRMD_B__MINOFF",
        # user-provided best short combo sometimes appears as FNONE; we normalize it to FB.
        "SHORT_ONLY__B_EMA_CROSS_DC10__FNONE__TS_A__PRMD_A__MINON",
        "SHORT_ONLY__A_55__FB__TS_C__PRMD_B__MINON",
    ]

    # Manual config mapping (explicit, no ambiguity)
    combo_cfgs: List[Tuple[str, BacktestConfig]] = []

    def norm_filter_token(combo_id: str) -> str:
        # enforce __FB__ in the output id
        return combo_id.replace("__FNONE__", "__FB__")

    for cid in requested_ids:
        cid_norm = norm_filter_token(cid)
        if cid != cid_norm:
            print(f"[normalize] {cid} -> {cid_norm} (Filter B forced ON)")

        if cid_norm.startswith("LONG_ONLY__B_EMA_CROSS_DC10"):
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
                ts_type=TrailingStopType.A_PCT,
                ts_activate_gain=float(args.ts_activate_gain),
                ts_floor_gain=float(args.ts_floor_gain),
                ts_trail_frac=float(args.ts_trail_frac),
                ts_box_window=int(args.ts_box_window),
                pyramiding_type=PyramidingType.B_DARVAS_BOX,
                pyramid_trigger=float(args.pyramid_trigger),
                pyramid_cooldown_days=int(args.pyramid_cooldown_days),
                entry_rule=EntryRuleType.B_EMA_CROSS_DC10,
                trade_mode=TradeMode.LONG_ONLY,
                filter_pl=True,
                filter_cycle=True,  # forced ON
                filter_market_cycle=False,
                min_position_mode=False,
            )

        elif cid_norm.startswith("LONG_ONLY__A_TURTLE"):
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
                ts_type=TrailingStopType.A_PCT,
                ts_activate_gain=float(args.ts_activate_gain),
                ts_floor_gain=float(args.ts_floor_gain),
                ts_trail_frac=float(args.ts_trail_frac),
                ts_box_window=int(args.ts_box_window),
                pyramiding_type=PyramidingType.B_DARVAS_BOX,
                pyramid_trigger=float(args.pyramid_trigger),
                pyramid_cooldown_days=int(args.pyramid_cooldown_days),
                entry_rule=EntryRuleType.A_TURTLE,
                trade_mode=TradeMode.LONG_ONLY,
                filter_pl=True,
                filter_cycle=True,  # forced ON
                filter_market_cycle=False,
                min_position_mode=False,
            )

        elif cid_norm.startswith("SHORT_ONLY__B_EMA_CROSS_DC10"):
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
                ts_type=TrailingStopType.A_PCT,
                ts_activate_gain=float(args.ts_activate_gain),
                ts_floor_gain=float(args.ts_floor_gain),
                ts_trail_frac=float(args.ts_trail_frac),
                ts_box_window=int(args.ts_box_window),
                pyramiding_type=PyramidingType.A_PCT,
                pyramid_trigger=float(args.pyramid_trigger),
                pyramid_cooldown_days=int(args.pyramid_cooldown_days),
                entry_rule=EntryRuleType.B_EMA_CROSS_DC10,
                trade_mode=TradeMode.SHORT_ONLY,
                filter_pl=True,
                filter_cycle=True,  # forced ON
                filter_market_cycle=False,
                min_position_mode=True,
            )

        elif cid_norm.startswith("SHORT_ONLY__A_55"):
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
                ts_type=TrailingStopType.C_DARVAS_BOX,
                ts_activate_gain=float(args.ts_activate_gain),
                ts_floor_gain=float(args.ts_floor_gain),
                ts_trail_frac=float(args.ts_trail_frac),
                ts_box_window=int(args.ts_box_window),
                pyramiding_type=PyramidingType.B_DARVAS_BOX,
                pyramid_trigger=float(args.pyramid_trigger),
                pyramid_cooldown_days=int(args.pyramid_cooldown_days),
                entry_rule=EntryRuleType.A_55,
                trade_mode=TradeMode.SHORT_ONLY,
                filter_pl=True,
                filter_cycle=True,  # forced ON
                filter_market_cycle=False,
                min_position_mode=True,
            )
        else:
            raise ValueError(f"Unsupported combo id: {cid_norm}")

        combo_cfgs.append((cid_norm, cfg))

    # ---- run backtests ----
    per_rows: List[Dict[str, object]] = []
    equities: List[pd.DataFrame] = []
    trades_list: List[pd.DataFrame] = []

    for combo_id, cfg in combo_cfgs:
        run_dir = runs_dir / _safe_name(combo_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        bt = SingleSymbolBacktester(df=base_df.copy(), cfg=cfg)
        res = bt.run()

        # save outputs
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
        res.equity_curve.reset_index().rename(columns={"index": "date"}).to_csv(run_dir / "equity_curve.csv", index=False)
        res.trades.to_csv(run_dir / "trades.csv", index=False)
        res.fills.to_csv(run_dir / "fills.csv", index=False)
        (run_dir / "summary.json").write_text(json.dumps(res.summary, indent=2, default=str), encoding="utf-8")
        (run_dir / "trader_report.txt").write_text(bt.trader.how_did_you_trade(max_trades=200), encoding="utf-8")

        plot_df = base_df.copy().reset_index().rename(columns={"index": "date"})
        plot_df.to_csv(run_dir / "plot_data.csv", index=False)

        write_interactive_abc(
            plot_data_csv=run_dir / "plot_data.csv",
            equity_curve_csv=run_dir / "equity_curve.csv",
            trades_csv=run_dir / "trades.csv",
            out_html=run_dir / "abc.html",
            title=combo_id,
            show_donchian=bool(args.show_donchian),
        )

        per_rows.append({
            "combo_id": combo_id,
            "mode": cfg.trade_mode.value,
            "entry_rule": cfg.entry_rule.value,
            "ts_type": cfg.ts_type.value,
            "pyramiding_type": cfg.pyramiding_type.value,
            "minpos": int(bool(cfg.min_position_mode)),
            "filter_cycle": int(bool(cfg.filter_cycle)),
            "cagr": float(res.summary.get("cagr", 0.0)),
            "mdd": float(res.summary.get("mdd", 0.0)),
            "num_trades": int(res.summary.get("num_trades", 0)),
            "run_dir": str(run_dir),
            "abc_html": str((run_dir / "abc.html").relative_to(out_dir)).replace("\\", "/"),
        })

        eq = res.equity_curve.copy()
        eq.index = pd.to_datetime(eq.index)
        equities.append(eq)
        trades_list.append(res.trades.copy())

    per_df = pd.DataFrame(per_rows).sort_values(["mode", "combo_id"]).reset_index(drop=True)
    per_df.to_csv(out_dir / "results_4combo.csv", index=False)

    # ---- merged metrics ----
    merged_eq, merged_stats = _merged_no_overlap(equities, trades_list, initial_capital=float(args.initial_capital))
    merged_eq.reset_index().rename(columns={"index": "date"}).to_csv(out_dir / "merged_equity_curve.csv", index=False)
    (out_dir / "merged_summary.json").write_text(json.dumps(merged_stats, indent=2), encoding="utf-8")

    # print summary
    def fmt_pct(x: float) -> str:
        return f"{100.0 * float(x):.2f}%"

    print("\n=== 4-combo results ===")
    for _, r in per_df.iterrows():
        print(
            f"- {r['combo_id']}: CAGR={fmt_pct(r['cagr'])}, MDD={fmt_pct(r['mdd'])}, trades={int(r['num_trades'])}"
        )

    print("\n=== Merged (no-overlap, conservative) ===")
    print(f"- merged_cagr: {fmt_pct(merged_stats['merged_cagr'])}")
    print(f"- merged_mdd : {fmt_pct(merged_stats['merged_mdd'])}")
    print(f"- overlap_pct_total : {fmt_pct(merged_stats['overlap_pct_total'])}")
    print(f"- overlap_pct_active: {fmt_pct(merged_stats['overlap_pct_active'])}")

    print("\n=== Outputs ===")
    print(f"- {out_dir / 'results_4combo.csv'}")
    print(f"- {out_dir / 'merged_equity_curve.csv'}")
    print(f"- {out_dir / 'merged_summary.json'}")
    print(f"- runs: {runs_dir}")


if __name__ == "__main__":
    main()
