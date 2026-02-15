#!/usr/bin/env python3
"""Smoke-test runner for trendfollowing_latest v4 (+extensions).

Capabilities
------------
- Loads synthetic OHLCV cases from data/smoke_test/manifest.json
- Adds indicators
- Runs TraderMasterBacktester
- Asserts expected fill events and rule-level assertions

This runner supports:
- Single-trader cases (legacy): case has fields {csv, symbol, config_overrides}
- Multi-trader cases: case has field {traders:[{trader_id,symbol,csv,config_overrides,...}, ...]}

Design goals
------------
- Deterministic, small, dependency-light (pandas + project code only)
- Focused on *behavioral invariants* (GAP/TOUCH, priorities, filters, constraints)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- project imports ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invest_v2.backtest.master_engine import TraderMasterBacktester, PortfolioBacktestResult
from invest_v2.core.types import EntryRuleType, PyramidingType, TradeMode, TrailingStopType
from invest_v2.data_loader import sanitize_ohlc
from invest_v2.indicators.ma_cycle import cycle_allows
from invest_v2.prep import add_indicators
from invest_v2.trading.trader import TraderConfig
from invest_v2.trading.trader_master import TraderMasterConfig

# We intentionally import an internal helper to validate a past regression:
# merged report must count "intraday stop-out" days as active.
from invest_v2.reporting.three_part_report import _active_mask


MANIFEST_PATH = ROOT / "data" / "smoke_test" / "manifest.json"
CASE_DIR = ROOT / "data" / "smoke_test"

FILL_COLS = [
    "trader_id",
    "symbol",
    "fill_type",
    "fill_date",
    "fill_price",
    "fill_shares",
    "reason",
    "side",
    "pos_units_after",
    "pos_shares_after",
    "avg_price_after",
]

TRADE_COLS = [
    "trader_id",
    "symbol",
    "side",
    "entry_date",
    "entry_price",
    "avg_entry_price",
    "num_entries",
    "shares",
    "exit_date",
    "exit_price",
    "realized_pnl",
    "exit_reason",
]


# -----------------------
# helpers
# -----------------------

def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _normalize_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df is None or (df.empty and len(df.columns) == 0):
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _load_case_df(case_dir: Path, csv_rel: str) -> pd.DataFrame:
    path = case_dir / csv_rel
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"CSV missing 'date' column: {path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = sanitize_ohlc(df)
    return df


def _coerce_enum(field: str, value: Any) -> Any:
    if field in ("entry_rule", "entry_rule_long", "entry_rule_short"):
        return EntryRuleType(value)
    if field == "trade_mode":
        return TradeMode(value)
    if field == "ts_type":
        return TrailingStopType(value)
    if field == "pyramiding_type":
        return PyramidingType(value)
    return value


def _assert_fill_prices_within_ohlc(symbol_dfs: Dict[str, pd.DataFrame], fills: pd.DataFrame) -> None:
    if fills.empty:
        return

    for _, fill in fills.iterrows():
        sym = str(fill.get("symbol"))
        if sym not in symbol_dfs:
            raise AssertionError(f"Fill symbol {sym} not in symbol_dfs")

        df = symbol_dfs[sym]
        d = _parse_date(str(fill.get("fill_date")))
        if d not in df.index:
            raise AssertionError(f"Fill date {d.date()} not in OHLC index for symbol={sym}")

        h = float(df.loc[d, "high"])
        l = float(df.loc[d, "low"])
        px = float(fill.get("fill_price"))
        if not (l - 1e-9 <= px <= h + 1e-9):
            o = float(df.loc[d, "open"])
            c = float(df.loc[d, "close"])
            raise AssertionError(
                f"Fill price {px} out of OHLC range [{l},{h}] on {d.date()} symbol={sym} (O={o},C={c})"
            )


def _assert_expected_fills(fills: pd.DataFrame, expected: List[Dict[str, Any]]) -> None:
    """Ensure each expected fill exists exactly once.

    Expected fields
    ---------------
    - fill_type (required)
    - date      (required)
    - reason    (optional)
    - side      (optional)
    - price     (optional)
    - trader_id (optional)
    - symbol    (optional)
    """
    if not expected:
        return
    if fills.empty:
        raise AssertionError(f"Expected {len(expected)} fill(s) but fills are empty")

    for exp in expected:
        ftype = exp["fill_type"]
        date = exp["date"]
        reason = exp.get("reason")
        side = exp.get("side")
        price = exp.get("price")
        trader_id = exp.get("trader_id")
        symbol = exp.get("symbol")

        cond = fills["fill_type"].astype(str) == str(ftype)
        cond &= fills["fill_date"].astype(str) == str(date)
        if reason is not None:
            cond &= fills["reason"].astype(str) == str(reason)
        if side is not None:
            cond &= fills["side"].astype(int) == int(side)
        if trader_id is not None:
            cond &= fills["trader_id"].astype(str) == str(trader_id)
        if symbol is not None:
            cond &= fills["symbol"].astype(str) == str(symbol)

        matches = fills[cond]
        if len(matches) != 1:
            preview = fills[["trader_id", "symbol", "fill_type", "fill_date", "reason", "side", "fill_price"]].to_dict(
                orient="records"
            )
            raise AssertionError(
                f"Expected exactly one fill match: {exp}, found {len(matches)}.\nFills: {preview}"
            )

        if price is not None:
            got = float(matches.iloc[0]["fill_price"])
            if abs(got - float(price)) > 1e-6:
                raise AssertionError(f"Expected fill price {price}, got {got} for {exp}")


def _count_matching_fills(fills: pd.DataFrame, crit: Dict[str, Any]) -> int:
    if fills.empty:
        return 0

    mask = pd.Series(True, index=fills.index)
    if "trader_id" in crit:
        mask &= fills["trader_id"].astype(str) == str(crit["trader_id"])
    if "symbol" in crit:
        mask &= fills["symbol"].astype(str) == str(crit["symbol"])
    if "fill_type" in crit:
        mask &= fills["fill_type"].astype(str) == str(crit["fill_type"])
    if "date" in crit:
        mask &= fills["fill_date"].astype(str) == str(crit["date"])
    if "reason" in crit:
        mask &= fills["reason"].astype(str) == str(crit["reason"])
    if "side" in crit:
        mask &= fills["side"].astype(int) == int(crit["side"])

    return int(mask.sum())


def _assert_fill_count(fills: pd.DataFrame, expected_counts: Dict[str, int]) -> None:
    for ftype, exp_n in expected_counts.items():
        n = 0
        if not fills.empty:
            n = int((fills["fill_type"].astype(str) == str(ftype)).sum())
        if n != int(exp_n):
            raise AssertionError(f"fill_count[{ftype}] expected {exp_n}, got {n}")


def _assert_no_fills_matching(fills: pd.DataFrame, forbidden: List[Dict[str, Any]]) -> None:
    for crit in forbidden:
        n = _count_matching_fills(fills, crit)
        if n != 0:
            raise AssertionError(f"Expected 0 fills matching {crit}, got {n}")


def _assert_earliest_fill_date(fills: pd.DataFrame, expected_date: str) -> None:
    if fills.empty:
        raise AssertionError(
            f"Expected at least 1 fill but got none (earliest_fill_date={expected_date})"
        )
    min_date = pd.to_datetime(fills["fill_date"]).min().normalize()
    exp = _parse_date(expected_date)
    if min_date != exp:
        raise AssertionError(f"earliest_fill_date expected {exp.date()}, got {min_date.date()}")


def _assert_max_pos_units_after(fills: pd.DataFrame, expected_max: int, trader_id: Optional[str] = None) -> None:
    if fills.empty:
        got = 0
    else:
        df = fills
        if trader_id is not None:
            df = df[df["trader_id"].astype(str) == str(trader_id)]
        got = int(pd.to_numeric(df["pos_units_after"], errors="coerce").max()) if len(df) else 0

    if got != int(expected_max):
        raise AssertionError(f"max_pos_units_after expected {expected_max}, got {got} (trader_id={trader_id})")


def _assert_pyramid_min_gap_days(df: pd.DataFrame, fills: pd.DataFrame, min_gap: int, trader_id: Optional[str] = None) -> None:
    if fills.empty:
        raise AssertionError("No fills present; cannot check pyramid_min_gap_days")

    f = fills
    if trader_id is not None:
        f = f[f["trader_id"].astype(str) == str(trader_id)]

    pyr = f[f["fill_type"].astype(str) == "PYRAMID"].copy()
    if len(pyr) < 2:
        return  # trivially satisfied

    # Map fill_date -> df index position (trading-day distance)
    idxs: List[int] = []
    for d in pd.to_datetime(pyr["fill_date"]).dt.normalize().tolist():
        if d not in df.index:
            raise AssertionError(f"Pyramid fill date {d.date()} not found in OHLC index")
        idxs.append(int(df.index.get_loc(d)))

    idxs = sorted(idxs)
    gaps = [idxs[i] - idxs[i - 1] for i in range(1, len(idxs))]
    if min(gaps) < int(min_gap):
        raise AssertionError(f"pyramid_min_gap_days expected >= {min_gap}, got gaps={gaps}")


def _raw_check_eval(df: pd.DataFrame, chk: Dict[str, Any]) -> bool:
    typ = chk["type"]
    d = _parse_date(chk["date"])
    if d not in df.index:
        raise AssertionError(f"raw_check date {d.date()} not found in OHLC index")
    i = int(df.index.get_loc(d))

    def _get(col: str) -> float:
        return float(df.iloc[i][col])

    if typ == "DONCHIAN20_LONG":
        return _get("close") >= _get("donchian_high_20")
    if typ == "DONCHIAN20_SHORT":
        return _get("close") <= _get("donchian_low_20")
    if typ == "DONCHIAN55_LONG":
        return _get("close") >= _get("donchian_high_55")
    if typ == "DONCHIAN55_SHORT":
        return _get("close") <= _get("donchian_low_55")
    if typ == "DONCHIAN10_LONG":
        return _get("close") >= _get("donchian_high_10")
    if typ == "DONCHIAN10_SHORT":
        return _get("close") <= _get("donchian_low_10")

    if typ == "CYCLE_ALLOWS_LONG":
        phase = df.iloc[i].get("cycle_phase")
        return bool(cycle_allows(1, phase))
    if typ == "CYCLE_ALLOWS_SHORT":
        phase = df.iloc[i].get("cycle_phase")
        return bool(cycle_allows(-1, phase))

    if typ == "EMA_GOLDEN_CROSS":
        if i == 0:
            return False
        e5_prev = float(df.iloc[i - 1]["ema5"])
        e20_prev = float(df.iloc[i - 1]["ema20"])
        e5 = float(df.iloc[i]["ema5"])
        e20 = float(df.iloc[i]["ema20"])
        return (e5_prev <= e20_prev) and (e5 > e20)

    if typ == "EMA_DEAD_CROSS":
        if i == 0:
            return False
        e5_prev = float(df.iloc[i - 1]["ema5"])
        e20_prev = float(df.iloc[i - 1]["ema20"])
        e5 = float(df.iloc[i]["ema5"])
        e20 = float(df.iloc[i]["ema20"])
        return (e5_prev >= e20_prev) and (e5 < e20)

    if typ == "EMA20_40_DEAD_CROSS":
        if i == 0:
            return False
        e20_prev = float(df.iloc[i - 1]["ema20"])
        e40_prev = float(df.iloc[i - 1]["ema40"])
        e20 = float(df.iloc[i]["ema20"])
        e40 = float(df.iloc[i]["ema40"])
        return (e20_prev >= e40_prev) and (e20 < e40)

    if typ == "ES3_C2C_LONG":
        p = float(chk.get("p", 0.05))
        if i == 0:
            return False
        c_prev = float(df.iloc[i - 1]["close"])
        c = float(df.iloc[i]["close"])
        return (c / c_prev - 1.0) <= -p

    if typ == "ES3_C2C_SHORT":
        p = float(chk.get("p", 0.05))
        if i == 0:
            return False
        c_prev = float(df.iloc[i - 1]["close"])
        c = float(df.iloc[i]["close"])
        return (c / c_prev - 1.0) >= p

    raise AssertionError(f"Unknown raw_check type: {typ}")


def _assert_raw_checks(symbol_dfs: Dict[str, pd.DataFrame], checks: List[Dict[str, Any]]) -> None:
    for chk in checks:
        expected = bool(chk.get("expected", True))
        sym = str(chk.get("symbol") or chk.get("sym") or "")
        if not sym:
            if len(symbol_dfs) != 1:
                raise AssertionError(f"raw_check requires 'symbol' when multiple symbols exist: {chk}")
            sym = next(iter(symbol_dfs.keys()))
        if sym not in symbol_dfs:
            raise AssertionError(f"raw_check symbol not found: {sym}")

        got = bool(_raw_check_eval(symbol_dfs[sym], chk))
        if got != expected:
            raise AssertionError(f"raw_check failed: {chk} -> got={got}")


def _portfolio_max_units(res: PortfolioBacktestResult) -> int:
    if res is None or not getattr(res, "trader_equity_curves", None):
        return 0
    series_list: List[pd.Series] = []
    for tid, eq in res.trader_equity_curves.items():
        if eq is None or eq.empty:
            continue
        if "pos_units" not in eq.columns:
            continue
        s = pd.to_numeric(eq["pos_units"], errors="coerce").fillna(0).astype(int)
        s.name = str(tid)
        series_list.append(s)

    if not series_list:
        return 0

    mat = pd.concat(series_list, axis=1).fillna(0).astype(int)
    return int(mat.sum(axis=1).max())


def _sell_cost_rate_by_trader(case: Dict[str, Any]) -> Dict[str, float]:
    """Resolve sell_cost_rate per trader for assertions.

    Priority
    --------
    - Case-level config_overrides
    - Trader-level overrides (when case.traders exists)

    Defaults to 0.003 if not specified.
    """

    default_rate = float((case.get("config_overrides") or {}).get("sell_cost_rate", 0.003))
    if not case.get("traders"):
        return {"SMOKE": default_rate}

    out: Dict[str, float] = {}
    case_defaults = dict(case.get("config_overrides") or {})
    for tspec in list(case.get("traders") or []):
        tid = str(tspec.get("trader_id"))
        merged = dict(case_defaults)
        merged.update(tspec.get("config_overrides") or {})
        out[tid] = float(merged.get("sell_cost_rate", default_rate))
    return out


def _get_trader_equity_dt(res: PortfolioBacktestResult, trader_id: str) -> pd.DataFrame:
    if res is None or not getattr(res, "trader_equity_curves", None):
        raise AssertionError("Result is missing trader_equity_curves")
    if trader_id not in res.trader_equity_curves:
        raise AssertionError(f"trader_equity_curves missing trader_id={trader_id}")
    eq = res.trader_equity_curves[trader_id].copy()
    if eq is None or eq.empty:
        raise AssertionError(f"Empty equity curve for trader_id={trader_id}")
    # normalize index to datetime for reporting helpers
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index, errors="coerce")
    return eq


def _assert_active_mask_should_capture_intraday_stop(
    res: PortfolioBacktestResult,
    trades: pd.DataFrame,
    spec: Dict[str, Any],
    trader_id: str = "SMOKE",
) -> None:
    """Ensure report-layer active mask marks an intraday stop-out day as active."""
    if res is None:
        raise AssertionError("active_mask assertion requires portfolio result")

    date = str(spec.get("date"))
    if not date:
        raise AssertionError("active_mask_should_capture_intraday_stop requires 'date'")

    eq = _get_trader_equity_dt(res, trader_id)
    tr = trades.copy()
    for c in ("entry_date", "exit_date"):
        if c in tr.columns:
            tr[c] = pd.to_datetime(tr[c], errors="coerce")

    mask = _active_mask(eq[["nav", "pos_side"]], tr)
    d = pd.to_datetime(date)
    # normalize dates to match _active_mask normalization
    d = d.normalize()
    if d not in mask.index.normalize().unique():
        raise AssertionError(f"active_mask date not found in equity index: {date}")

    got = bool(mask.loc[mask.index.normalize() == d].iloc[0])
    if not got:
        raise AssertionError(f"active_mask should be True on intraday stop-out date={date}")


def _assert_monthly_interest_deduction(
    res: PortfolioBacktestResult,
    spec: Dict[str, Any],
    trader_id: str = "SMOKE",
    tol: float = 1e-6,
) -> None:
    """Validate monthly short-interest cash deduction is applied on first trading day of month.

    We use NAV deltas because accrued_interest is not included in NAV until it is deducted.
    """
    if res is None:
        raise AssertionError("monthly_interest_deduction assertion requires portfolio result")

    jan_last = str(spec.get("jan_last"))
    feb_first = str(spec.get("feb_first"))
    if not jan_last or not feb_first:
        raise AssertionError("monthly_interest_deduction requires 'jan_last' and 'feb_first'")

    eq = _get_trader_equity_dt(res, trader_id)
    if "nav" not in eq.columns or "accrued_interest" not in eq.columns:
        raise AssertionError("Equity curve missing required columns: nav/accrued_interest")

    d_jan = pd.to_datetime(jan_last)
    d_feb = pd.to_datetime(feb_first)

    if d_jan not in eq.index:
        raise AssertionError(f"jan_last date not in equity curve index: {jan_last}")
    if d_feb not in eq.index:
        raise AssertionError(f"feb_first date not in equity curve index: {feb_first}")

    nav_jan = float(eq.loc[d_jan, "nav"])
    nav_feb = float(eq.loc[d_feb, "nav"])
    accrued_jan = float(eq.loc[d_jan, "accrued_interest"])

    if accrued_jan <= 0:
        raise AssertionError(f"Expected positive accrued_interest at jan_last={jan_last}, got {accrued_jan}")

    # On first trading day of month, the system deducts the prior month's accrued interest from cash.
    # NAV should drop by ~accrued_jan (price is constant in the smoke case).
    drop = nav_jan - nav_feb
    if abs(drop - accrued_jan) > max(tol, 1e-3):
        raise AssertionError(
            f"Monthly interest deduction mismatch: NAV drop {drop:.6f} vs accrued_jan {accrued_jan:.6f}"
        )


def _assert_trade_pnl_matches_fee_model(
    case: Dict[str, Any],
    fills: pd.DataFrame,
    trades: pd.DataFrame,
    tol: float = 1e-6,
) -> None:
    """Recompute realized PnL from fills using the project's fee model and compare.

    Fee model (project spec)
    ------------------------
    - Long: buy fee 0%, sell fee = sell_cost_rate
        realized = exit_price*shares*(1 - fee) - sum(entry_price_i*entry_shares_i)
    - Short: short-sell fee = sell_cost_rate, cover fee 0%
        realized = sum(entry_price_i*entry_shares_i)*(1 - fee) - exit_price*shares
    """

    if trades is None or trades.empty:
        raise AssertionError("trade_pnl_matches_fee_model requires at least 1 trade")
    if fills is None or fills.empty:
        raise AssertionError("trade_pnl_matches_fee_model requires fills")

    fee_by_tid = _sell_cost_rate_by_trader(case)

    tr = trades.copy()
    for c in ("entry_date", "exit_date"):
        if c in tr.columns:
            tr[c] = pd.to_datetime(tr[c], errors="coerce")

    fl = fills.copy()
    if "fill_date" in fl.columns:
        fl["fill_date"] = pd.to_datetime(fl["fill_date"], errors="coerce")

    closed = tr.dropna(subset=["exit_date", "realized_pnl"]).copy()
    if closed.empty:
        raise AssertionError("trade_pnl_matches_fee_model requires at least 1 closed trade")

    for _, t in closed.iterrows():
        tid = str(t.get("trader_id"))
        sym = str(t.get("symbol"))
        side = int(t.get("side"))
        entry_dt = pd.to_datetime(t.get("entry_date"))
        exit_dt = pd.to_datetime(t.get("exit_date"))
        exit_price = float(t.get("exit_price"))
        realized = float(t.get("realized_pnl"))
        fee = float(fee_by_tid.get(tid, 0.003))

        fsub = fl[(fl["trader_id"] == tid) & (fl["symbol"] == sym)].copy()
        if fsub.empty:
            raise AssertionError(f"No fills for trade tid={tid} sym={sym}")

        fwin = fsub[(fsub["fill_date"] >= entry_dt) & (fsub["fill_date"] <= exit_dt)].copy()
        if fwin.empty:
            raise AssertionError(f"No fills in trade window tid={tid} sym={sym} {entry_dt}..{exit_dt}")

        entry_like = fwin[fwin["fill_type"].isin(["ENTRY", "PYRAMID"])].copy()
        exit_like = fwin[fwin["fill_type"] == "EXIT"].copy()

        if entry_like.empty:
            raise AssertionError(f"Trade window missing ENTRY/PYRAMID fills: tid={tid} sym={sym}")
        if exit_like.empty:
            raise AssertionError(f"Trade window missing EXIT fill: tid={tid} sym={sym}")

        entry_notional = float((entry_like["fill_price"].astype(float) * entry_like["fill_shares"].astype(float)).sum())
        shares = int(entry_like["fill_shares"].astype(int).sum())

        # Sanity: ensure exit fill price matches trade exit price.
        exit_fill_price = float(exit_like.sort_values("fill_date").iloc[-1]["fill_price"])
        if abs(exit_fill_price - exit_price) > 1e-6:
            raise AssertionError(
                f"Exit price mismatch tid={tid} sym={sym}: trade={exit_price} fill={exit_fill_price}"
            )

        if side == 1:
            expected = exit_price * shares * (1.0 - fee) - entry_notional
        elif side == -1:
            expected = entry_notional * (1.0 - fee) - exit_price * shares
        else:
            raise AssertionError(f"Unexpected trade side: {side}")

        if abs(expected - realized) > max(tol, 1e-3):
            raise AssertionError(
                f"Fee PnL mismatch tid={tid} sym={sym} side={side}: expected={expected:.6f}, got={realized:.6f}"
            )


def _assert_nav_matches_realized_pnl(
    res: PortfolioBacktestResult,
    trades: pd.DataFrame,
    tol: float = 1e-6,
    trader_id: str = "SMOKE",
) -> None:
    """When a case ends FLAT with no external cashflows, NAV delta must equal sum(realized_pnl)."""
    if res is None:
        raise AssertionError("nav_matches_realized_pnl requires portfolio result")

    eq = _get_trader_equity_dt(res, trader_id)
    if "nav" not in eq.columns or "pos_side" not in eq.columns:
        raise AssertionError("Equity curve missing required columns: nav/pos_side")
    if int(pd.to_numeric(eq["pos_side"], errors="coerce").fillna(0).astype(int).iloc[-1]) != 0:
        raise AssertionError("nav_matches_realized_pnl requires ending FLAT (pos_side=0)")

    tsub = trades[trades["trader_id"] == trader_id].copy() if (trades is not None and not trades.empty) else pd.DataFrame()
    realized_sum = 0.0
    if not tsub.empty and "realized_pnl" in tsub.columns:
        realized_sum = float(pd.to_numeric(tsub["realized_pnl"], errors="coerce").fillna(0.0).sum())

    nav0 = float(eq["nav"].astype(float).iloc[0])
    navT = float(eq["nav"].astype(float).iloc[-1])
    got = navT - nav0
    if abs(got - realized_sum) > max(tol, 1e-3):
        raise AssertionError(
            f"NAV delta mismatch: navT-nav0={got:.6f} vs sum(realized_pnl)={realized_sum:.6f}"
        )


def _assert_nav_is_constant(res: PortfolioBacktestResult, tol: float = 1e-9, trader_id: str = "SMOKE") -> None:
    if res is None:
        raise AssertionError("nav_is_constant requires portfolio result")
    eq = _get_trader_equity_dt(res, trader_id)
    nav = pd.to_numeric(eq.get("nav"), errors="coerce").dropna().astype(float)
    if nav.empty:
        raise AssertionError("nav_is_constant: NAV series is empty")
    if float(nav.max() - nav.min()) > max(tol, 1e-9):
        raise AssertionError(f"NAV is not constant: min={nav.min():.6f}, max={nav.max():.6f}")


def _assert_pos_side_always_flat(res: PortfolioBacktestResult, trader_id: str = "SMOKE") -> None:
    if res is None:
        raise AssertionError("pos_side_always_flat requires portfolio result")
    eq = _get_trader_equity_dt(res, trader_id)
    pos = pd.to_numeric(eq.get("pos_side"), errors="coerce").fillna(0).astype(int)
    if int(pos.abs().max()) != 0:
        raise AssertionError("Expected pos_side always FLAT (0), but found non-zero exposure")


def _assert_no_duplicate_exits_per_day(fills: pd.DataFrame) -> None:
    if fills is None or fills.empty:
        return
    f = fills.copy()
    f = f[f["fill_type"] == "EXIT"].copy()
    if f.empty:
        return
    grp = f.groupby(["trader_id", "symbol", "fill_date"]).size()
    mx = int(grp.max()) if len(grp) else 0
    if mx > 1:
        bad = grp[grp > 1].reset_index(name="n").to_dict(orient="records")
        raise AssertionError(f"Duplicate EXIT fills detected (per trader/symbol/day): {bad}")


def _apply_assertions(
    case: Dict[str, Any],
    symbol_dfs: Dict[str, pd.DataFrame],
    fills: pd.DataFrame,
    trades: pd.DataFrame,
    res: Optional[PortfolioBacktestResult],
) -> None:
    a = case.get("assertions") or {}

    if a.get("expect_no_fills"):
        if not fills.empty:
            preview = fills[["trader_id", "symbol", "fill_type", "fill_date", "reason", "side", "fill_price"]].to_dict(
                orient="records"
            )
            raise AssertionError(f"Expected no fills but got {len(fills)}: {preview}")

    if "fill_count" in a:
        _assert_fill_count(fills, a["fill_count"])

    if "no_fills_matching" in a:
        _assert_no_fills_matching(fills, a["no_fills_matching"])

    if "earliest_fill_date" in a:
        _assert_earliest_fill_date(fills, a["earliest_fill_date"])

    if "max_pos_units_after" in a:
        # legacy: applies to whole fills
        _assert_max_pos_units_after(fills, int(a["max_pos_units_after"]))

    if "max_pos_units_after_by_trader" in a:
        for tid, v in a["max_pos_units_after_by_trader"].items():
            _assert_max_pos_units_after(fills, int(v), trader_id=str(tid))

    if "pyramid_min_gap_days" in a:
        if len(symbol_dfs) != 1:
            raise AssertionError("pyramid_min_gap_days requires single-symbol case")
        sym = next(iter(symbol_dfs.keys()))
        _assert_pyramid_min_gap_days(symbol_dfs[sym], fills, int(a["pyramid_min_gap_days"]))

    if "raw_checks" in a:
        _assert_raw_checks(symbol_dfs, a["raw_checks"])

    if "portfolio_max_units" in a:
        if res is None:
            raise AssertionError("portfolio_max_units assertion requires portfolio result")
        got = _portfolio_max_units(res)
        exp = int(a["portfolio_max_units"])
        if got != exp:
            raise AssertionError(f"portfolio_max_units expected {exp}, got {got}")

    if "portfolio_max_units_leq" in a:
        if res is None:
            raise AssertionError("portfolio_max_units_leq assertion requires portfolio result")
        got = _portfolio_max_units(res)
        exp = int(a["portfolio_max_units_leq"])
        if got > exp:
            raise AssertionError(f"portfolio_max_units_leq expected <= {exp}, got {got}")

    # --- NEW / previously-unwired assertions (integrity regressions) ---

    if "active_mask_should_capture_intraday_stop" in a:
        if res is None:
            raise AssertionError("active_mask_should_capture_intraday_stop requires portfolio result")
        _assert_active_mask_should_capture_intraday_stop(
            res=res,
            trades=trades,
            spec=dict(a["active_mask_should_capture_intraday_stop"] or {}),
            trader_id="SMOKE",
        )

    if "monthly_interest_deduction" in a:
        if res is None:
            raise AssertionError("monthly_interest_deduction requires portfolio result")
        _assert_monthly_interest_deduction(
            res=res,
            spec=dict(a["monthly_interest_deduction"] or {}),
            trader_id="SMOKE",
        )

    if "trade_pnl_matches_fee_model" in a:
        spec = dict(a["trade_pnl_matches_fee_model"] or {})
        tol = float(spec.get("tolerance", 1e-6))
        _assert_trade_pnl_matches_fee_model(case=case, fills=fills, trades=trades, tol=tol)

    if "nav_matches_realized_pnl" in a:
        if res is None:
            raise AssertionError("nav_matches_realized_pnl requires portfolio result")
        spec = dict(a["nav_matches_realized_pnl"] or {})
        tol = float(spec.get("tolerance", 1e-6))
        _assert_nav_matches_realized_pnl(res=res, trades=trades, tol=tol, trader_id="SMOKE")

    if "nav_is_constant" in a:
        if res is None:
            raise AssertionError("nav_is_constant requires portfolio result")
        spec = dict(a["nav_is_constant"] or {})
        tol = float(spec.get("tolerance", 1e-9))
        _assert_nav_is_constant(res=res, tol=tol, trader_id="SMOKE")

    if a.get("pos_side_always_flat"):
        if res is None:
            raise AssertionError("pos_side_always_flat requires portfolio result")
        _assert_pos_side_always_flat(res=res, trader_id="SMOKE")

    if a.get("no_duplicate_exits_per_day"):
        _assert_no_duplicate_exits_per_day(fills)


def _run_case(case: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, PortfolioBacktestResult]:
    # --- build trader configs ---
    trader_cfgs: List[TraderConfig] = []
    symbol_dfs: Dict[str, pd.DataFrame] = {}

    # Case-level defaults
    case_defaults: Dict[str, Any] = dict(case.get("config_overrides") or {})

    if case.get("traders"):
        trader_specs = list(case.get("traders") or [])
        if not trader_specs:
            raise ValueError("case.traders is present but empty")

        for tspec in trader_specs:
            tid = str(tspec.get("trader_id"))
            sym = str(tspec.get("symbol"))
            csv_rel = str(tspec.get("csv"))
            if not tid or not sym or not csv_rel:
                raise ValueError(f"Each trader spec must include trader_id, symbol, csv: {tspec}")

            if sym not in symbol_dfs:
                df_raw = _load_case_df(CASE_DIR, csv_rel)
                df = add_indicators(df_raw)
                symbol_dfs[sym] = df

            cfg_kwargs = asdict(TraderConfig(trader_id=tid, symbol=sym))

            merged_overrides = dict(case_defaults)
            merged_overrides.update(tspec.get("config_overrides") or {})
            for k, v in merged_overrides.items():
                cfg_kwargs[k] = _coerce_enum(k, v)

            trader_cfgs.append(TraderConfig(**cfg_kwargs))

        master_over = dict(case.get("master_config_overrides") or {})
        master_cfg = TraderMasterConfig(**master_over) if master_over else TraderMasterConfig(max_units_total=10)

    else:
        # legacy single-trader case
        df_raw = _load_case_df(CASE_DIR, case["csv"])
        df = add_indicators(df_raw)

        cfg_kwargs = asdict(TraderConfig(trader_id="SMOKE", symbol=case.get("symbol", "999999")))
        for k, v in case_defaults.items():
            cfg_kwargs[k] = _coerce_enum(k, v)

        cfg = TraderConfig(**cfg_kwargs)
        trader_cfgs = [cfg]
        symbol_dfs = {cfg.symbol: df}
        master_cfg = TraderMasterConfig(max_units_total=10)

    # --- run ---
    bt = TraderMasterBacktester(symbol_dfs=symbol_dfs, trader_cfgs=trader_cfgs, master_cfg=master_cfg)
    res = bt.run()

    fills = _normalize_df(res.fills, FILL_COLS)
    trades = _normalize_df(res.trades, TRADE_COLS)

    return symbol_dfs, fills, trades, res


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"Missing manifest: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text())
    cases = manifest.get("cases", [])

    selected_id = sys.argv[1] if len(sys.argv) > 1 else None
    if selected_id:
        cases = [c for c in cases if c.get("case_id") == selected_id]
        if not cases:
            print(f"No case_id matched: {selected_id}", file=sys.stderr)
            return 2

    ok = 0
    fail = 0
    for case in cases:
        cid = case.get("case_id", "(no id)")
        try:
            symbol_dfs, fills, trades, res = _run_case(case)
            _assert_fill_prices_within_ohlc(symbol_dfs, fills)
            _assert_expected_fills(fills, list(case.get("expected_fills", [])))
            _apply_assertions(case, symbol_dfs, fills, trades, res)
            print(f"[OK] {cid} - {case.get('description','')}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {cid} - {case.get('description','')}")
            print(f"  {type(e).__name__}: {e}")
            fail += 1

    print(f"\nSummary: {ok} passed, {fail} failed (total={ok+fail})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
