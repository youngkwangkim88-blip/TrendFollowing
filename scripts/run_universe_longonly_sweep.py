#!/usr/bin/env python3
"""Universe (multi-ticker) principle test runner — LONG_ONLY fixed.

Goal
----
Run the fixed long-only strategy grid across a user-defined universe of tickers
and produce an aggregate report suitable for "원리 시험" (principle verification)
and regression monitoring.

Fixed assumptions (per user request)
-----------------------------------
- Trade mode: LONG_ONLY
- Filter B (ticker cycle): ALWAYS ON (engine-enforced)
- Min-position mode: ALWAYS ON
- Market cycle filter: ALWAYS OFF

Grid
----
- Entry rules: A_TURTLE, B_EMA_CROSS_DC10
- Trailing stops: TS.A / TS.B / TS.C
- Pyramiding: OFF / PRMD.A / PRMD.B

Total = 2 * 3 * 3 = 18 runs per ticker.

Performance
-----------
Uses ProcessPoolExecutor for CPU parallelism. Default workers ~ 90% of logical
CPU count (user machine: i7-14700k, 28 threads -> 25 workers).

Outputs
-------
outputs/<experiment>/
  prepared/                     # pickled, indicator-enriched OHLC per ticker
  runs/<symbol>/<combo>/         # per-run artifacts (summary/trades/fills/equity)
  results_all.csv                # one row per run
  results_combo_aggregate.csv    # aggregated by combo across tickers
  report_universe_longonly.html  # single HTML report

Optional
--------
Generate interactive abc.html for "top K per ticker" after sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from invest_v2.core.types import EntryRuleType, PyramidingType, TradeMode, TrailingStopType
from invest_v2.data_loader import load_ohlc_panel_symbols
from invest_v2.prep import add_indicators
from invest_v2.backtest.engine import BacktestConfig, SingleSymbolBacktester
from invest_v2.reporting.plotly_abc import write_interactive_abc


# -------------------------
# Universe mapping (KRX)
# -------------------------

DEFAULT_UNIVERSE_NAME_TO_CODE: Dict[str, str] = {
    "삼성전자": "005930",
    "현대차": "005380",
    "LG에너지솔루션": "373220",
    "삼성바이오로직스": "207940",
    "KB금융": "105560",
    "두산에너빌러티": "034020",
    "HD현대중공업": "329180",
    "한화에어로스페이스": "012450",
    "삼성물산": "028260",
    "NAVER": "035420",
    "naver": "035420",
    "SK": "034730",
    "HMM": "011200",
    "SK텔레콤": "017670",
}

DEFAULT_UNIVERSE_CODES: List[str] = [
    "005930",
    "005380",
    "373220",
    "207940",
    "105560",
    "034020",
    "329180",
    "012450",
    "028260",
    "035420",
    "034730",
    "011200",
    "017670",
]


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


def _fmt_pct(x: float) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return ""


def _fmt_float(x: float, nd: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def _recommend_workers(target_util: float = 0.90) -> int:
    n = os.cpu_count() or 1
    util = max(0.10, min(1.00, float(target_util)))
    # conservative: never exceed n-1 to keep OS responsive
    w = int(math.floor(n * util))
    w = max(1, min(w, max(1, n - 1)))
    return w


def _set_single_thread_env() -> None:
    """Avoid hidden multi-threading inside each worker process."""
    for k in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(k, "1")


@dataclass(frozen=True)
class RunTask:
    symbol: str
    prepared_path: str
    out_dir: str
    entry_rule: str
    ts_type: str
    pyramiding_type: str
    cfg_kwargs: Dict[str, Any]
    resume: bool
    write_abc: bool


def _run_one_task(task: RunTask) -> Dict[str, Any]:
    """Worker entrypoint (must be top-level for pickling)."""
    _set_single_thread_env()
    t0 = time.perf_counter()

    symbol = str(task.symbol).zfill(6)
    entry_rule = EntryRuleType(task.entry_rule)
    ts_type = TrailingStopType(task.ts_type)
    prmd_type = PyramidingType(task.pyramiding_type)

    combo_id = _safe_name(f"{entry_rule.value}__{ts_type.value}__{prmd_type.value}")
    out_dir = Path(task.out_dir)
    run_dir = out_dir / "runs" / symbol / combo_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    if task.resume and summary_path.exists():
        # read minimal summary for reporting
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            s = {}
        return {
            "symbol": symbol,
            "combo_id": combo_id,
            "entry_rule": entry_rule.value,
            "ts_type": ts_type.value,
            "pyramiding_type": prmd_type.value,
            "status": "SKIP(resume)",
            "cagr": float(s.get("cagr", 0.0)) if isinstance(s, dict) else 0.0,
            "mdd": float(s.get("mdd", 0.0)) if isinstance(s, dict) else 0.0,
            "win_rate": float(s.get("win_rate", 0.0)) if isinstance(s, dict) else 0.0,
            "payoff_ratio": float(s.get("payoff_ratio", 0.0)) if isinstance(s, dict) else 0.0,
            "num_trades": int(s.get("num_trades", 0)) if isinstance(s, dict) else 0,
            "run_dir": str(run_dir),
            "runtime_sec": float(time.perf_counter() - t0),
            "error": "",
        }

    try:
        df = pd.read_pickle(Path(task.prepared_path))

        cfg = BacktestConfig(
            symbol=symbol,
            initial_capital=float(task.cfg_kwargs["initial_capital"]),
            one_trading_risk=float(task.cfg_kwargs["one_trading_risk"]),
            max_units_per_symbol=int(task.cfg_kwargs["max_units_per_symbol"]),
            max_units_total=int(task.cfg_kwargs["max_units_total"]),
            sell_cost_rate=float(task.cfg_kwargs["sell_cost_rate"]),
            short_notional_limit=float(task.cfg_kwargs["short_notional_limit"]),
            annual_short_interest_rate=float(task.cfg_kwargs["annual_short_interest_rate"]),
            short_max_hold_days=int(task.cfg_kwargs["short_max_hold_days"]),
            stop_atr_mult=float(task.cfg_kwargs["stop_atr_mult"]),
            ts_type=ts_type,
            ts_activate_gain=float(task.cfg_kwargs["ts_activate_gain"]),
            ts_floor_gain=float(task.cfg_kwargs["ts_floor_gain"]),
            ts_trail_frac=float(task.cfg_kwargs["ts_trail_frac"]),
            ts_box_window=int(task.cfg_kwargs["ts_box_window"]),
            even_stop_gain=float(task.cfg_kwargs["even_stop_gain"]),
            emergency_stop_pct=float(task.cfg_kwargs["emergency_stop_pct"]),
            pyramiding_type=prmd_type,
            pyramid_trigger=float(task.cfg_kwargs["pyramid_trigger"]),
            pyramid_box_window=int(task.cfg_kwargs["pyramid_box_window"]),
            pyramid_cooldown_days=int(task.cfg_kwargs["pyramid_cooldown_days"]),
            entry_rule=entry_rule,
            trade_mode=TradeMode.LONG_ONLY,
            # Fixed per-session assumptions:
            filter_pl=True,
            filter_cycle=True,
            filter_market_cycle=False,
            min_position_mode=True,
            minpos_trigger_consecutive_losses=int(task.cfg_kwargs["minpos_trigger_consecutive_losses"]),
            minpos_entry_factor=float(task.cfg_kwargs["minpos_entry_factor"]),
            minpos_first_pyramid_factor=float(task.cfg_kwargs["minpos_first_pyramid_factor"]),
        )

        bt = SingleSymbolBacktester(df=df, cfg=cfg)
        res = bt.run()

        # save outputs
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
        res.equity_curve.reset_index().rename(columns={"index": "date"}).to_csv(run_dir / "equity_curve.csv", index=False)
        res.trades.to_csv(run_dir / "trades.csv", index=False)
        res.fills.to_csv(run_dir / "fills.csv", index=False)
        summary_path.write_text(json.dumps(res.summary, indent=2, default=str), encoding="utf-8")
        (run_dir / "trader_report.txt").write_text(bt.trader.how_did_you_trade(max_lines=5000), encoding="utf-8")

        abc_rel = ""
        if task.write_abc:
            # For abc.html, we need plot_data.csv with indicator columns.
            plot_df = df.copy().reset_index().rename(columns={"index": "date"})
            plot_df.to_csv(run_dir / "plot_data.csv", index=False)
            html_path = write_interactive_abc(
                plot_data_csv=run_dir / "plot_data.csv",
                equity_curve_csv=run_dir / "equity_curve.csv",
                trades_csv=run_dir / "trades.csv",
                fills_csv=run_dir / "fills.csv",
                out_html=run_dir / "abc.html",
                title=f"{symbol} | {combo_id}",
            )
            abc_rel = str(Path(html_path).relative_to(out_dir)).replace("\\", "/")

        row = {
            "symbol": symbol,
            "combo_id": combo_id,
            "entry_rule": entry_rule.value,
            "ts_type": ts_type.value,
            "pyramiding_type": prmd_type.value,
            "status": "OK",
            "cagr": float(res.summary.get("cagr", 0.0)),
            "mdd": float(res.summary.get("mdd", 0.0)),
            "win_rate": float(res.summary.get("win_rate", 0.0)),
            "payoff_ratio": float(res.summary.get("payoff_ratio", 0.0)),
            "num_trades": int(res.summary.get("num_trades", 0)),
            "run_dir": str(run_dir),
            "abc_rel": abc_rel,
            "runtime_sec": float(time.perf_counter() - t0),
            "error": "",
        }
        return row

    except Exception as e:
        tb = traceback.format_exc(limit=50)
        (run_dir / "error.txt").write_text(tb, encoding="utf-8")
        return {
            "symbol": symbol,
            "combo_id": combo_id,
            "entry_rule": entry_rule.value,
            "ts_type": ts_type.value,
            "pyramiding_type": prmd_type.value,
            "status": "FAIL",
            "cagr": 0.0,
            "mdd": 0.0,
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "num_trades": 0,
            "run_dir": str(run_dir),
            "abc_rel": "",
            "runtime_sec": float(time.perf_counter() - t0),
            "error": f"{type(e).__name__}: {e}",
        }


def _write_report(out_dir: Path, results_all: pd.DataFrame, universe_meta: Dict[str, Any]) -> Path:
    """Generate a single self-contained HTML report (tables + links)."""

    df = results_all.copy()
    # ensure stable types
    for c in ["cagr", "mdd", "win_rate", "payoff_ratio", "runtime_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "num_trades" in df.columns:
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").fillna(0).astype(int)

    # link columns
    def _link_run_dir(p: str) -> str:
        try:
            rel = str(Path(p).relative_to(out_dir)).replace("\\", "/")
            return f"<a href='{rel}/summary.json'>run</a>"
        except Exception:
            return ""

    def _link_abc(p: str) -> str:
        if not isinstance(p, str) or not p:
            return ""
        pp = p.replace("\\", "/")
        return f"<a href='{pp}'>abc</a>"

    df["run_link"] = df["run_dir"].astype(str).map(_link_run_dir) if "run_dir" in df.columns else ""
    if "abc_rel" in df.columns:
        df["abc_link"] = df["abc_rel"].astype(str).map(_link_abc)
    else:
        df["abc_link"] = ""

    # coverage matrix (symbol x combo)
    symbols = sorted(df["symbol"].astype(str).unique().tolist()) if "symbol" in df.columns else []
    combos = sorted(df["combo_id"].astype(str).unique().tolist()) if "combo_id" in df.columns else []

    pivot = df.pivot_table(index="symbol", columns="combo_id", values="status", aggfunc="first", fill_value="MISSING")
    pivot = pivot.reindex(index=symbols, columns=combos)

    # Treat resume-skips as successful runs (they have a valid summary.json).
    success = df[
        df["status"].astype(str).str.startswith("OK")
        | df["status"].astype(str).str.startswith("SKIP")
    ].copy()
    agg_rows = []
    if len(success):
        g = success.groupby("combo_id", dropna=False)
        combo_agg = g.agg(
            n=("symbol", "count"),
            mean_cagr=("cagr", "mean"),
            median_cagr=("cagr", "median"),
            min_cagr=("cagr", "min"),
            max_cagr=("cagr", "max"),
            mean_mdd=("mdd", "mean"),
            median_mdd=("mdd", "median"),
            worst_mdd=("mdd", "min"),
            pos_cagr_rate=("cagr", lambda s: float((s > 0).mean())),
            mean_trades=("num_trades", "mean"),
        ).reset_index()
        combo_agg = combo_agg.sort_values(["median_cagr", "pos_cagr_rate"], ascending=[False, False])
        combo_agg.to_csv(out_dir / "results_combo_aggregate.csv", index=False)
    else:
        combo_agg = pd.DataFrame()

    # per-ticker best table
    best_rows = []
    if len(success):
        for sym, sub in success.groupby("symbol"):
            sub2 = sub.sort_values(["cagr", "mdd"], ascending=[False, False])
            top = sub2.iloc[0]
            best_rows.append(
                {
                    "symbol": sym,
                    "best_combo": top["combo_id"],
                    "best_cagr": top["cagr"],
                    "best_mdd": top["mdd"],
                    "best_trades": int(top["num_trades"]),
                    "best_run": top.get("run_link", ""),
                    "best_abc": top.get("abc_link", ""),
                }
            )
    best_df = pd.DataFrame(best_rows)
    if len(best_df):
        best_df = best_df.sort_values("best_cagr", ascending=False)

    # format tables
    show_all = df[[
        "symbol",
        "combo_id",
        "entry_rule",
        "ts_type",
        "pyramiding_type",
        "status",
        "cagr",
        "mdd",
        "num_trades",
        "win_rate",
        "payoff_ratio",
        "runtime_sec",
        "run_link",
        "abc_link",
        "error",
    ]].copy()
    show_all["cagr"] = show_all["cagr"].map(_fmt_pct)
    show_all["mdd"] = show_all["mdd"].map(_fmt_pct)
    show_all["win_rate"] = show_all["win_rate"].map(_fmt_pct)
    show_all["payoff_ratio"] = show_all["payoff_ratio"].map(lambda x: _fmt_float(x, 3))
    show_all["runtime_sec"] = show_all["runtime_sec"].map(lambda x: _fmt_float(x, 2))

    if len(combo_agg):
        combo_show = combo_agg.copy()
        combo_show["mean_cagr"] = combo_show["mean_cagr"].map(_fmt_pct)
        combo_show["median_cagr"] = combo_show["median_cagr"].map(_fmt_pct)
        combo_show["min_cagr"] = combo_show["min_cagr"].map(_fmt_pct)
        combo_show["max_cagr"] = combo_show["max_cagr"].map(_fmt_pct)
        combo_show["mean_mdd"] = combo_show["mean_mdd"].map(_fmt_pct)
        combo_show["median_mdd"] = combo_show["median_mdd"].map(_fmt_pct)
        combo_show["worst_mdd"] = combo_show["worst_mdd"].map(_fmt_pct)
        combo_show["pos_cagr_rate"] = combo_show["pos_cagr_rate"].map(_fmt_pct)
        combo_show["mean_trades"] = combo_show["mean_trades"].map(lambda x: _fmt_float(x, 1))
        combo_html = combo_show.to_html(index=False, escape=False)
    else:
        combo_html = "<p>(No successful runs)</p>"

    if len(best_df):
        best_show = best_df.copy()
        best_show["best_cagr"] = best_show["best_cagr"].map(_fmt_pct)
        best_show["best_mdd"] = best_show["best_mdd"].map(_fmt_pct)
        best_html = best_show.to_html(index=False, escape=False)
    else:
        best_html = "<p>(No successful runs)</p>"

    # coverage matrix as HTML (compact)
    cov_html = pivot.to_html(escape=False)

    meta_pretty = json.dumps(universe_meta, indent=2, ensure_ascii=False)

    total = len(df)
    ok_n = int(df["status"].astype(str).str.startswith("OK").sum())
    fail_n = int((df["status"] == "FAIL").sum())
    skip_n = int(df["status"].astype(str).str.startswith("SKIP").sum())

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Universe Long-only Principle Test Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }}
    h1,h2,h3 {{ margin: 0.35em 0 0.5em; }}
    .note {{ color: #222; background: #f6f6f6; padding: 12px 14px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
    th {{ background: #fafafa; text-align: left; }}
    code, pre {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
    pre {{ padding: 10px 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Universe Long-only Principle Test</h1>
  <div class='note'>
    <div><b>Scope</b>: LONG_ONLY 고정, Filter B=ON, MinPos=ON, Market Cycle=OFF</div>
    <div><b>Grid</b>: entry(2) × ts(3) × prmd(3) = 18 combos per ticker</div>
    <div><b>Runs</b>: total={total}, ok={ok_n}, fail={fail_n}, skip(resume)={skip_n}</div>
  </div>

  <h2>Experiment metadata</h2>
  <pre>{meta_pretty}</pre>

  <h2>Part 1 — Combo robustness leaderboard (aggregated across tickers)</h2>
  {combo_html}

  <h2>Part 2 — Best combo per ticker (by CAGR)</h2>
  {best_html}

  <h2>Part 3 — Coverage matrix (symbol × combo_id)</h2>
  <p>셀 값은 각 (symbol, combo) 실행 상태입니다. 원칙적으로 모두 <b>OK</b>여야 합니다.</p>
  {cov_html}

  <h2>Part 4 — Full run table (all rows)</h2>
  {show_all.to_html(index=False, escape=False)}

</body>
</html>
"""

    out = out_dir / "report_universe_longonly.html"
    out.write_text(html, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universe long-only sweep runner (multi-core) + HTML report.")

    p.add_argument("--csv", type=str, default="data/krx100_adj_5000.csv", help="KRX panel CSV path")
    p.add_argument("--out-root", type=str, default="outputs", help="Root output directory")
    p.add_argument("--experiment", type=str, default="universe_longonly_13tickers_v1", help="Experiment name")

    p.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_UNIVERSE_CODES),
        help="Comma-separated: 6-digit tickers or Korean names (mapped via built-in dict).",
    )

    p.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    # grid
    p.add_argument(
        "--entry-rules",
        type=str,
        default=",".join([EntryRuleType.A_TURTLE.value, EntryRuleType.B_EMA_CROSS_DC10.value]),
    )
    p.add_argument("--ts-types", type=str, default=",".join([e.value for e in TrailingStopType]))
    p.add_argument(
        "--pyramiding-types",
        type=str,
        default=",".join([PyramidingType.OFF.value, PyramidingType.A_PCT.value, PyramidingType.B_DARVAS_BOX.value]),
    )

    # backtest knobs (keep aligned with toy script defaults)
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
    p.add_argument("--even-stop-gain", type=float, default=0.10)
    p.add_argument("--emergency-stop-pct", type=float, default=0.05)
    p.add_argument("--pyramid-trigger", type=float, default=0.15)
    p.add_argument("--pyramid-box-window", type=int, default=20)
    p.add_argument("--pyramid-cooldown-days", type=int, default=5)
    p.add_argument("--minpos-trigger-consecutive-losses", type=int, default=2)
    p.add_argument("--minpos-entry-factor", type=float, default=0.5)
    p.add_argument("--minpos-first-pyramid-factor", type=float, default=0.5)

    # parallel
    p.add_argument("--workers", type=int, default=None, help="Process workers. Default: ~90% of logical CPUs")
    p.add_argument("--target-cpu-util", type=float, default=0.90, help="Used when --workers is not provided")

    # reporting / i/o
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--resume", action="store_true", help="Skip runs with existing summary.json")

    # abc generation control
    p.add_argument("--write-abc-all", action="store_true", help="Generate abc.html for every run (large/slow)")
    p.add_argument(
        "--write-abc-topk-per-ticker",
        type=int,
        default=0,
        help="After sweep, generate abc.html for top-K combos per ticker (by CAGR).",
    )

    return p.parse_args()


def _resolve_symbols(raw: str) -> List[str]:
    items = _parse_csv_list(raw)
    out: List[str] = []
    for x in items:
        if x in DEFAULT_UNIVERSE_NAME_TO_CODE:
            out.append(DEFAULT_UNIVERSE_NAME_TO_CODE[x])
            continue
        # numeric code
        xx = str(x).strip()
        if xx.isdigit():
            out.append(xx.zfill(6))
            continue
        # allow '005930.KS' style
        if "." in xx and xx.split(".")[0].isdigit():
            out.append(xx.split(".")[0].zfill(6))
            continue
        raise ValueError(f"Unrecognized symbol/name: {x}. Provide 6-digit ticker or known name.")

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _prepare_pickles(
    *,
    panel_csv: str,
    symbols: List[str],
    start: Optional[str],
    end: Optional[str],
    prepared_dir: Path,
    resume: bool,
) -> Dict[str, Path]:
    """Load panel once, compute indicators, and persist per-symbol pickles."""
    prepared_dir.mkdir(parents=True, exist_ok=True)

    # If resume and all pickles exist, skip loading entirely
    if resume:
        all_exist = True
        for sym in symbols:
            if not (prepared_dir / f"{sym}.pkl").exists():
                all_exist = False
                break
        if all_exist:
            return {sym: (prepared_dir / f"{sym}.pkl") for sym in symbols}

    # Load in one pass
    panel = load_ohlc_panel_symbols(panel_csv, symbols)
    out: Dict[str, Path] = {}
    for sym in symbols:
        df = panel[str(sym).zfill(6)].copy()
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]

        # indicators (market cycle not used for this experiment)
        df = add_indicators(df, market_df=None)

        p = prepared_dir / f"{sym}.pkl"
        df.to_pickle(p)
        out[sym] = p
    return out


def _generate_abc_for_topk(
    *,
    out_dir: Path,
    prepared_paths: Dict[str, Path],
    results_all_csv: Path,
    topk: int,
) -> None:
    if topk <= 0:
        return
    df = pd.read_csv(results_all_csv)
    if "status" not in df.columns:
        return
    df_ok = df[
        df["status"].astype(str).str.startswith("OK")
        | df["status"].astype(str).str.startswith("SKIP")
    ].copy()
    if df_ok.empty:
        return

    # for each ticker, select top-k by CAGR
    targets: List[Tuple[str, str, str]] = []  # (symbol, combo_id, run_dir)
    for sym, sub in df_ok.groupby("symbol"):
        ss = sub.sort_values("cagr", ascending=False).head(int(topk))
        for _, r in ss.iterrows():
            targets.append((str(sym).zfill(6), str(r["combo_id"]), str(r["run_dir"])))

    # generate
    for sym, combo_id, run_dir_s in targets:
        run_dir = Path(run_dir_s)
        html = run_dir / "abc.html"
        if html.exists():
            continue
        pkl = prepared_paths.get(sym)
        if pkl is None or not Path(pkl).exists():
            continue
        dfp = pd.read_pickle(pkl)
        plot_df = dfp.copy().reset_index().rename(columns={"index": "date"})
        plot_df.to_csv(run_dir / "plot_data.csv", index=False)
        write_interactive_abc(
            plot_data_csv=run_dir / "plot_data.csv",
            equity_curve_csv=run_dir / "equity_curve.csv",
            trades_csv=run_dir / "trades.csv",
            fills_csv=run_dir / "fills.csv",
            out_html=run_dir / "abc.html",
            title=f"{sym} | {combo_id}",
        )

    # update results_all.csv with abc_rel links
    df2 = pd.read_csv(results_all_csv)
    abc_rels = []
    for _, r in df2.iterrows():
        run_dir = Path(str(r.get("run_dir", "")))
        html = run_dir / "abc.html"
        if html.exists():
            try:
                abc_rels.append(str(html.relative_to(out_dir)).replace("\\", "/"))
            except Exception:
                abc_rels.append("")
        else:
            abc_rels.append(str(r.get("abc_rel", "")) if "abc_rel" in df2.columns else "")
    df2["abc_rel"] = abc_rels
    df2.to_csv(results_all_csv, index=False)


def main() -> None:
    args = parse_args()
    _set_single_thread_env()

    # resolve universe
    symbols = _resolve_symbols(args.symbols)

    # resolve grid
    entry_rules = [EntryRuleType(x) for x in _parse_csv_list(args.entry_rules)]
    ts_types = [TrailingStopType(x) for x in _parse_csv_list(args.ts_types)]
    prmd_types = [PyramidingType(x) for x in _parse_csv_list(args.pyramiding_types)]
    combos = list(product(entry_rules, ts_types, prmd_types))

    # output dirs
    out_dir = Path(args.out_root) / _safe_name(args.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = out_dir / "prepared"
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    # meta
    universe_meta: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "panel_csv": str(args.csv),
        "symbols": symbols,
        "fixed": {
            "trade_mode": "LONG_ONLY",
            "filter_pl": True,
            "filter_cycle": True,
            "filter_market_cycle": False,
            "min_position_mode": True,
        },
        "grid": {
            "entry_rules": [e.value for e in entry_rules],
            "ts_types": [t.value for t in ts_types],
            "pyramiding_types": [p.value for p in prmd_types],
            "runs_per_symbol": int(len(combos)),
            "expected_total_runs": int(len(combos) * len(symbols)),
        },
        "date_range": {"start": args.start, "end": args.end},
        "parallel": {
            "workers": int(args.workers) if args.workers is not None else _recommend_workers(args.target_cpu_util),
            "target_cpu_util": float(args.target_cpu_util),
        },
        "params": {
            "initial_capital": float(args.initial_capital),
            "one_trading_risk": float(args.one_trading_risk),
            "max_units_per_symbol": int(args.max_units),
            "max_units_total": int(args.max_units_total),
            "sell_cost_rate": float(args.sell_cost_rate),
            "stop_atr_mult": float(args.stop_atr_mult),
            "ts_activate_gain": float(args.ts_activate_gain),
            "ts_floor_gain": float(args.ts_floor_gain),
            "ts_trail_frac": float(args.ts_trail_frac),
            "ts_box_window": int(args.ts_box_window),
            "even_stop_gain": float(args.even_stop_gain),
            "emergency_stop_pct": float(args.emergency_stop_pct),
            "pyramid_trigger": float(args.pyramid_trigger),
            "pyramid_box_window": int(args.pyramid_box_window),
            "pyramid_cooldown_days": int(args.pyramid_cooldown_days),
            "minpos_trigger_consecutive_losses": int(args.minpos_trigger_consecutive_losses),
            "minpos_entry_factor": float(args.minpos_entry_factor),
            "minpos_first_pyramid_factor": float(args.minpos_first_pyramid_factor),
        },
        "notes": [
            "This is a principle-test sweep. Do not over-interpret absolute CAGR from a single ticker.",
            "Use combo aggregate (median CAGR / positive rate) for robustness ranking.",
        ],
    }
    (out_dir / "universe_meta.json").write_text(json.dumps(universe_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # prepare data (single pass load + indicator enrichment)
    prepared_paths = _prepare_pickles(
        panel_csv=args.csv,
        symbols=symbols,
        start=args.start,
        end=args.end,
        prepared_dir=prepared_dir,
        resume=bool(args.resume),
    )

    # tasks
    cfg_kwargs = {
        "initial_capital": float(args.initial_capital),
        "one_trading_risk": float(args.one_trading_risk),
        "max_units_per_symbol": int(args.max_units),
        "max_units_total": int(args.max_units_total),
        "sell_cost_rate": float(args.sell_cost_rate),
        "short_notional_limit": float(args.short_notional_limit),
        "annual_short_interest_rate": float(args.annual_short_interest_rate),
        "short_max_hold_days": int(args.short_max_hold_days),
        "stop_atr_mult": float(args.stop_atr_mult),
        "ts_activate_gain": float(args.ts_activate_gain),
        "ts_floor_gain": float(args.ts_floor_gain),
        "ts_trail_frac": float(args.ts_trail_frac),
        "ts_box_window": int(args.ts_box_window),
        "even_stop_gain": float(args.even_stop_gain),
        "emergency_stop_pct": float(args.emergency_stop_pct),
        "pyramid_trigger": float(args.pyramid_trigger),
        "pyramid_box_window": int(args.pyramid_box_window),
        "pyramid_cooldown_days": int(args.pyramid_cooldown_days),
        "minpos_trigger_consecutive_losses": int(args.minpos_trigger_consecutive_losses),
        "minpos_entry_factor": float(args.minpos_entry_factor),
        "minpos_first_pyramid_factor": float(args.minpos_first_pyramid_factor),
    }

    tasks: List[RunTask] = []
    for sym in symbols:
        pkl = prepared_paths[str(sym).zfill(6)]
        for er, ts, pr in combos:
            tasks.append(
                RunTask(
                    symbol=str(sym).zfill(6),
                    prepared_path=str(pkl),
                    out_dir=str(out_dir),
                    entry_rule=er.value,
                    ts_type=ts.value,
                    pyramiding_type=pr.value,
                    cfg_kwargs=cfg_kwargs,
                    resume=bool(args.resume),
                    write_abc=bool(args.write_abc_all),
                )
            )

    expected_total = int(len(symbols) * len(combos))
    print("=== Universe long-only sweep ===")
    print(f"out_dir={out_dir.resolve()}")
    print(f"symbols={len(symbols)} | combos={len(combos)} | expected_total_runs={expected_total}")

    workers = int(args.workers) if args.workers is not None else _recommend_workers(args.target_cpu_util)
    print(f"workers={workers} (cpu_count={os.cpu_count()})")
    print(f"write_abc_all={bool(args.write_abc_all)}, write_abc_topk_per_ticker={int(args.write_abc_topk_per_ticker)}")

    # run parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    results_csv = out_dir / "results_all.csv"
    checkpoint_every = max(1, int(args.checkpoint_every))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one_task, t) for t in tasks]
        total = len(futs)
        done = 0
        fail = 0
        for fut in as_completed(futs):
            row = fut.result()
            results.append(row)
            done += 1
            if row.get("status") == "FAIL":
                fail += 1
            if done % 10 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                rps = done / max(1e-9, elapsed)
                print(f"[{done}/{total}] done | fails={fail} | {rps:.2f} runs/sec")
            if done % checkpoint_every == 0:
                pd.DataFrame(results).to_csv(results_csv.with_name("results_all_checkpoint.csv"), index=False)

    df_all = pd.DataFrame(results)
    # normalize run_dir to relative paths (for stable report links)
    if "run_dir" in df_all.columns:
        df_all["run_dir"] = df_all["run_dir"].astype(str)
    if "abc_rel" not in df_all.columns:
        df_all["abc_rel"] = ""
    df_all.to_csv(results_csv, index=False)

    # optional: generate abc.html for top-k per ticker (after sweep)
    if int(args.write_abc_topk_per_ticker) > 0:
        _generate_abc_for_topk(
            out_dir=out_dir,
            prepared_paths=prepared_paths,
            results_all_csv=results_csv,
            topk=int(args.write_abc_topk_per_ticker),
        )
        # reload to include abc_rel
        df_all = pd.read_csv(results_csv)

    report_path = _write_report(out_dir=out_dir, results_all=df_all, universe_meta=universe_meta)

    # hard coverage check
    success_df = df_all[
        df_all["status"].astype(str).str.startswith("OK")
        | df_all["status"].astype(str).str.startswith("SKIP")
    ].copy()
    ok_count = len(success_df)
    fail_count = int((df_all["status"] == "FAIL").sum())
    expected = expected_total
    if ok_count < expected:
        missing = expected - ok_count
        print(f"ERROR: coverage incomplete: ok={ok_count} < expected={expected} (missing={missing}).")
        print(f"report={report_path}")
        raise SystemExit(2)
    if fail_count > 0:
        print(f"ERROR: {fail_count} runs failed. See results_all.csv and per-run error.txt")
        print(f"report={report_path}")
        raise SystemExit(3)

    elapsed = time.perf_counter() - t0
    print("\n=== Outputs ===")
    print(f"- results: {results_csv}")
    print(f"- combo aggregate: {out_dir / 'results_combo_aggregate.csv'}")
    print(f"- report: {report_path}")
    print(f"- elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    # For Windows compatibility when frozen; harmless otherwise.
    try:
        import multiprocessing as mp

        mp.freeze_support()
    except Exception:
        pass
    main()
