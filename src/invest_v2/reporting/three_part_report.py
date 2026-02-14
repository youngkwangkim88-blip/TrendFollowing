from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd


def _read_results(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    # Backward compatibility: older result files may not have combo_id/abc_html.
    if "combo_id" not in df.columns:
        if "run_dir" in df.columns:
            df["combo_id"] = df["run_dir"].astype(str).map(lambda x: Path(x).name)
        else:
            df["combo_id"] = [f"combo_{i}" for i in range(len(df))]
    if "abc_html" not in df.columns and "abc_rel" in df.columns:
        df["abc_html"] = df["abc_rel"].astype(str).str.replace("\\", "/", regex=False)
    # normalize booleans
    for c in ("filter_cycle", "filter_market_cycle"):
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def _format_percent(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.2f}%"


def _format_float(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def _load_equity(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir) / "equity_curve.csv"
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError(f"equity_curve.csv missing date: {p}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def _cagr_from_nav(nav: pd.Series) -> float:
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        return 0.0
    start = float(nav.iloc[0])
    end = float(nav.iloc[-1])
    years = (nav.index[-1] - nav.index[0]).days / 365.0
    if years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def _mdd_from_nav(nav: pd.Series) -> float:
    """Max drawdown (as a negative number).

    Parameters
    ----------
    nav:
        NAV series indexed by datetime.
    """
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        return 0.0
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _load_trades(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir) / "trades.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _active_mask(eq: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    """Conservative 'active' mask for merge.

    We treat a day as active if:
      - Position exists at EOD, OR
      - Position existed at prior EOD (exposure at today's open), OR
      - Any trade entry/exit occurred on that date (captures intraday stop-outs).

    This avoids the common bug where intraday stop losses are missed because
    `pos_side` ends the day as FLAT.
    """
    pos = eq.get("pos_side", pd.Series(index=eq.index, data=0)).fillna(0).astype(int)
    active = (pos != 0) | (pos.shift(1).fillna(0).astype(int) != 0)

    if trades is not None and len(trades):
        dates = set()
        if "entry_date" in trades.columns:
            dates |= set(trades["entry_date"].dropna().dt.normalize().tolist())
        if "exit_date" in trades.columns:
            dates |= set(trades["exit_date"].dropna().dt.normalize().tolist())
        if dates:
            idx_norm = eq.index.normalize()
            active = active | idx_norm.isin(list(dates))
    return active.astype(bool)


def _union_no_overlap_metrics(
    eq_long: pd.DataFrame,
    tr_long: pd.DataFrame,
    eq_short: pd.DataFrame,
    tr_short: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """Compute overlap stats and a conservative merged CAGR/MDD.

    Rule (conservative, single-position constraint):
      - If exactly one strategy is active that day, apply that strategy's daily return.
      - If both are active that day (overlap), apply 0 return (flat).
      - If neither is active, apply 0 return.

    Notes
    -----
    We use a conservative `active` definition (see `_active_mask`) to avoid
    missing intraday stop-outs.

    Returns
    -------
    overlap_pct_total, overlap_pct_active, merged_cagr, merged_mdd
    """
    l = eq_long[["nav", "pos_side"]].copy()
    s = eq_short[["nav", "pos_side"]].copy()

    # align calendars
    idx = l.index.union(s.index)
    l = l.reindex(idx).ffill()
    s = s.reindex(idx).ffill()

    lr = l["nav"].pct_change().fillna(0.0).astype(float)
    sr = s["nav"].pct_change().fillna(0.0).astype(float)

    l_active = _active_mask(l, tr_long).reindex(idx).fillna(False)
    s_active = _active_mask(s, tr_short).reindex(idx).fillna(False)

    overlap = l_active & s_active
    active_union = l_active | s_active

    merged_r = pd.Series(0.0, index=idx)
    merged_r.loc[l_active & ~s_active] = lr.loc[l_active & ~s_active]
    merged_r.loc[~l_active & s_active] = sr.loc[~l_active & s_active]
    merged_r.loc[overlap] = 0.0

    merged_nav = (1.0 + merged_r).cumprod()
    merged_cagr = _cagr_from_nav(merged_nav)
    merged_mdd = _mdd_from_nav(merged_nav)

    overlap_pct_total = float(overlap.mean())
    overlap_pct_active = float(overlap.sum() / max(1, int(active_union.sum())))
    return overlap_pct_total, overlap_pct_active, float(merged_cagr), float(merged_mdd)


def _table_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    show = df.head(max_rows).copy()

    # compact column subset
    cols = []
    for c in [
        "combo_id",
        "entry_rule",
        "filter_cycle",
        "filter_market_cycle",
        "pyramiding_type",
        "ts_type",
        "cagr",
        "mdd",
        "num_trades",
        "win_rate",
        "payoff_ratio",
        "abc_html",
    ]:
        if c in show.columns:
            cols.append(c)
    show = show[cols]

    if "cagr" in show.columns:
        show["cagr"] = show["cagr"].map(_format_percent)
    if "mdd" in show.columns:
        show["mdd"] = show["mdd"].map(_format_percent)
    if "win_rate" in show.columns:
        show["win_rate"] = show["win_rate"].map(_format_percent)
    if "payoff_ratio" in show.columns:
        show["payoff_ratio"] = show["payoff_ratio"].map(_format_float)

    # linkify
    if "abc_html" in show.columns:
        show["abc_html"] = show["abc_html"].map(lambda p: f"<a href='{p}'>abc</a>" if isinstance(p, str) else "")

    return show.to_html(index=False, escape=False)


def write_three_part_report(
    out_dir: str | Path,
    results_long_csv: str | Path,
    results_short_csv: str | Path,
    top_k: int = 10,
) -> Path:
    out_dir = Path(out_dir)
    long_df = _read_results(results_long_csv)
    # Remove legacy single-window Turtle variants from LONG_ONLY report.
    # (We keep them in code for compatibility/debugging, but they should not
    # appear in the primary report/leaderboard.)
    if "entry_rule" in long_df.columns:
        long_df = long_df[~long_df["entry_rule"].isin(["A_20_PL", "A_55"])].copy()
    long_df = long_df.sort_values("cagr", ascending=False)
    short_df = _read_results(results_short_csv).sort_values("cagr", ascending=False)

    best_long = long_df.iloc[0] if len(long_df) else None
    best_short = short_df.iloc[0] if len(short_df) else None

    # pairing table (Top-K x Top-K)
    pairs: List[Dict[str, object]] = []
    if best_long is not None and best_short is not None:
        topL = long_df.head(int(top_k)).copy()
        topS = short_df.head(int(top_k)).copy()
        for _, L in topL.iterrows():
            eqL = _load_equity(L["run_dir"])
            trL = _load_trades(L["run_dir"])
            for _, S in topS.iterrows():
                eqS = _load_equity(S["run_dir"])
                trS = _load_trades(S["run_dir"])
                ov_total, ov_active, merged_cagr, merged_mdd = _union_no_overlap_metrics(eqL, trL, eqS, trS)
                pairs.append(
                    {
                        "long_combo": L.get("combo_id", Path(str(L.get("run_dir", ""))).name),
                        "short_combo": S.get("combo_id", Path(str(S.get("run_dir", ""))).name),
                        "overlap_pct_total": ov_total,
                        "overlap_pct_active": ov_active,
                        "merged_cagr": merged_cagr,
                        "merged_mdd": merged_mdd,
                        "long_abc": L.get("abc_html", ""),
                        "short_abc": S.get("abc_html", ""),
                    }
                )

    pair_df = pd.DataFrame(pairs)
    if len(pair_df):
        pair_df = pair_df.sort_values(["merged_cagr", "overlap_pct_active"], ascending=[False, True])
        pair_show = pair_df.head(50).copy()
        pair_show["merged_cagr"] = pair_show["merged_cagr"].map(_format_percent)
        pair_show["merged_mdd"] = pair_show["merged_mdd"].map(_format_percent)
        pair_show["overlap_pct_total"] = pair_show["overlap_pct_total"].map(_format_percent)
        pair_show["overlap_pct_active"] = pair_show["overlap_pct_active"].map(_format_percent)
        pair_show["long_abc"] = pair_show["long_abc"].map(lambda p: f"<a href='{p}'>long abc</a>" if isinstance(p, str) and p else "")
        pair_show["short_abc"] = pair_show["short_abc"].map(lambda p: f"<a href='{p}'>short abc</a>" if isinstance(p, str) and p else "")
        pair_html = pair_show.to_html(index=False, escape=False)
    else:
        pair_html = "<p>(No pairing table generated)</p>"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>005930 Combo Report (Long-only / Short-only / Overlap)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; }}
    h1,h2 {{ margin: 0.2em 0 0.4em; }}
    .note {{ color: #333; background: #f6f6f6; padding: 12px 14px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
    th {{ background: #fafafa; text-align: left; }}
    code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>005930 Combo Report</h1>
  <div class='note'>
    <div><b>Part 1</b>: LONG_ONLY 결과 (Filter A=ON 기본, Filter B/C 조합, TS/PRMD 조합)</div>
    <div><b>Part 2</b>: SHORT_ONLY 결과</div>
    <div><b>Part 3</b>: 중첩(동시 보유) 가능성 분석 — 법적으로 long/short 동시 보유 불가 전제</div>
  </div>

  <h2>Part 1 — Long-only leaderboard</h2>
  {_table_html(long_df, max_rows=40)}

  <h2>Part 2 — Short-only leaderboard</h2>
  {_table_html(short_df, max_rows=40)}

  <h2>Part 3 — Overlap / compatibility (Top-{int(top_k)} × Top-{int(top_k)})</h2>
  <p>
    아래 테이블은 long-only 상위 {int(top_k)}개와 short-only 상위 {int(top_k)}개의 조합쌍을 대상으로,
    <b>동시 포지션(overlap)</b> 발생 빈도와, 보수적으로 <b>겹치는 날은 거래하지 않는(수익률 0)</b> 규칙으로 합성한
    가상의 <b>merged CAGR</b> 및 <b>merged MDD</b>를 계산한 결과입니다.
  </p>
  {pair_html}

</body>
</html>
"""

    out = out_dir / "report_three_parts.html"
    out.write_text(html, encoding="utf-8")
    return out
