#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def _normalize_ticker(v: object) -> str:
    s = str(v).strip()
    if s.isdigit():
        return s.zfill(6)
    return s


def _resolve_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create pivot-point labels (20d/110d) and optional HTML chart.")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out", default="outputs/labeled_pivots.csv", help="Output CSV path")
    p.add_argument("--out-html", default=None, help="Output HTML chart path. Default: <out>.html")
    p.add_argument("--no-html", action="store_true", help="Disable HTML generation")
    p.add_argument("--ticker", default=None, help="Optional ticker filter for panel CSV")
    p.add_argument("--short-window", type=int, default=20, help="Short pivot window (default: 20)")
    p.add_argument("--mid-window", type=int, default=110, help="Mid pivot window (default: 110)")
    p.add_argument("--plot-tail", type=int, default=1000, help="Rows to display in HTML chart")
    return p.parse_args()


def _add_pivot_labels(df: pd.DataFrame, high_col: str, low_col: str, window: int, suffix: str) -> pd.DataFrame:
    order = max(1, int(window // 2))
    roll = 2 * order + 1

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    local_max = (high == high.rolling(window=roll, center=True, min_periods=roll).max()).fillna(False)
    local_min = (low == low.rolling(window=roll, center=True, min_periods=roll).min()).fillna(False)

    label_col = f"label_{suffix}"
    df[label_col] = 0
    df.loc[local_max, label_col] = 1
    df.loc[local_min, label_col] = -1
    return df


def _write_html(
    df: pd.DataFrame,
    date_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    out_html: Path,
    short_suffix: str,
    mid_suffix: str,
    plot_tail: int,
) -> None:
    plot_df = df.tail(int(plot_tail)).copy()
    x = plot_df[date_col]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=plot_df[close_col].astype(float),
            mode="lines",
            name="Close",
            line=dict(color="gray", width=1.5),
            opacity=0.7,
        )
    )

    max_20 = plot_df[plot_df[f"label_{short_suffix}"] == 1]
    min_20 = plot_df[plot_df[f"label_{short_suffix}"] == -1]
    max_110 = plot_df[plot_df[f"label_{mid_suffix}"] == 1]
    min_110 = plot_df[plot_df[f"label_{mid_suffix}"] == -1]

    fig.add_trace(
        go.Scatter(
            x=max_20[date_col],
            y=max_20[high_col].astype(float),
            mode="markers",
            name=f"Local Max ({short_suffix})",
            marker=dict(symbol="triangle-down", size=7, color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=min_20[date_col],
            y=min_20[low_col].astype(float),
            mode="markers",
            name=f"Local Min ({short_suffix})",
            marker=dict(symbol="triangle-up", size=7, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=max_110[date_col],
            y=max_110[high_col].astype(float) * 1.02,
            mode="markers",
            name=f"Local Max ({mid_suffix})",
            marker=dict(symbol="triangle-down", size=12, color="darkblue", line=dict(color="black", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=min_110[date_col],
            y=min_110[low_col].astype(float) * 0.98,
            mode="markers",
            name=f"Local Min ({mid_suffix})",
            marker=dict(symbol="triangle-up", size=12, color="darkred", line=dict(color="black", width=1)),
        )
    )

    fig.update_layout(
        title="Pivot Points Auto-Labeling",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    in_path = Path(args.csv)
    out_path = Path(args.out)
    out_html = Path(args.out_html) if args.out_html else out_path.with_suffix(".html")

    df = pd.read_csv(in_path, low_memory=False)

    ticker_col = _resolve_col(df, ["ticker", "종목코드", "code"], required=False)
    if args.ticker and ticker_col:
        target = _normalize_ticker(args.ticker)
        df = df[df[ticker_col].map(_normalize_ticker) == target].copy()
    elif args.ticker and not ticker_col:
        raise ValueError("--ticker was provided but no ticker column exists in the CSV.")

    date_col = _resolve_col(df, ["date", "Date", "날짜"])
    high_col = _resolve_col(df, ["H", "high", "High"])
    low_col = _resolve_col(df, ["L", "low", "Low"])
    close_col = _resolve_col(df, ["C", "close", "Close", "close_adj", "Close(Adj)"])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    short_suffix = f"{int(args.short_window)}d"
    mid_suffix = f"{int(args.mid_window)}d"
    df = _add_pivot_labels(df, high_col=high_col, low_col=low_col, window=int(args.short_window), suffix=short_suffix)
    df = _add_pivot_labels(df, high_col=high_col, low_col=low_col, window=int(args.mid_window), suffix=mid_suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_path, index=False)
        saved_csv = out_path
    except PermissionError:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}")
        df.to_csv(alt, index=False)
        saved_csv = alt
        print(f"[WARN] CSV is locked: {out_path}. Saved to alternate file: {alt}")

    print(f"Saved: {saved_csv}")
    print(f"Rows: {len(df)}")

    if not args.no_html:
        _write_html(
            df=df,
            date_col=date_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            out_html=out_html,
            short_suffix=short_suffix,
            mid_suffix=mid_suffix,
            plot_tail=int(args.plot_tail),
        )
        print(f"Saved: {out_html}")


if __name__ == "__main__":
    main()
