from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # date column 후보들
    for col in ["date", "Date", "dt", "datetime", "timestamp", "t"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.sort_values(col)
            df = df.set_index(col)
            return df
    raise ValueError(f"Cannot find date column in {path}. Columns={list(df.columns)}")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def create_interactive_abc_html(
    price_csv: str,
    equity_csv: str,
    out_html: str,
    trades_csv: Optional[str] = None,
    title: str = "Interactive A/B/C",
    start: Optional[str] = None,
    end: Optional[str] = None,
    width: int = 1600,
    height: int = 1000,
) -> str:
    """
    Create an interactive HTML with 3 vertically stacked plots (A/B/C) sharing the X-axis.
    Returns out_html.
    """
    price = _read_csv(price_csv)
    eq = _read_csv(equity_csv)

    if start:
        price = price[price.index >= pd.to_datetime(start)]
        eq = eq[eq.index >= pd.to_datetime(start)]
    if end:
        price = price[price.index <= pd.to_datetime(end)]
        eq = eq[eq.index <= pd.to_datetime(end)]

    close_col = _pick_col(price, ["close_adj", "Close(Adj)", "close", "C", "Close"])
    if close_col is None:
        raise ValueError(f"Close column not found in {price_csv}")

    ema5_col = _pick_col(price, ["ema5", "EMA5"])
    ema20_col = _pick_col(price, ["ema20", "EMA20"])
    ema40_col = _pick_col(price, ["ema40", "EMA40"])

    mom_col = _pick_col(price, ["mom63", "MOM63", "momentum_63", "mom_63", "mom", "momentum"])
    phase_col = _pick_col(price, ["cycle_phase", "phase", "ma_cycle_phase"])
    mkt_phase_col = _pick_col(price, ["market_cycle_phase", "mkt_phase", "market_phase"])

    nav_col = _pick_col(eq, ["nav", "equity", "NAV", "Equity", "total_value"])
    if nav_col is None:
        raise ValueError(f"NAV/equity column not found in {equity_csv}")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("Plot A: Price + EMA + Signals", "Plot B: Equity Curve", "Plot C: Momentum / Phase"),
    )

    # Plot A
    fig.add_trace(go.Scatter(x=price.index, y=price[close_col].astype(float), mode="lines", name=f"Close ({close_col})"), row=1, col=1)
    if ema5_col:
        fig.add_trace(go.Scatter(x=price.index, y=price[ema5_col].astype(float), mode="lines", name="EMA5"), row=1, col=1)
    if ema20_col:
        fig.add_trace(go.Scatter(x=price.index, y=price[ema20_col].astype(float), mode="lines", name="EMA20"), row=1, col=1)
    if ema40_col:
        fig.add_trace(go.Scatter(x=price.index, y=price[ema40_col].astype(float), mode="lines", name="EMA40"), row=1, col=1)

    # markers from trades.csv if exists
    if trades_csv and os.path.exists(trades_csv):
        tr = pd.read_csv(trades_csv)
        # normalize date columns
        for c in ["entry_date", "EntryDate", "entry_dt"]:
            if c in tr.columns:
                tr[c] = pd.to_datetime(tr[c])
                if c != "entry_date":
                    tr["entry_date"] = tr[c]
        for c in ["exit_date", "ExitDate", "exit_dt"]:
            if c in tr.columns:
                tr[c] = pd.to_datetime(tr[c])
                if c != "exit_date":
                    tr["exit_date"] = tr[c]

        side = tr.get("side")
        if side is None:
            side = tr.get("Side", pd.Series([""] * len(tr)))
        side = side.astype(str).str.upper()

        if "entry_date" in tr.columns and "entry_price" in tr.columns:
            long_entry = (side == "LONG") & tr["entry_date"].notna()
            short_entry = (side == "SHORT") & tr["entry_date"].notna()
            fig.add_trace(
                go.Scatter(
                    x=tr.loc[long_entry, "entry_date"],
                    y=tr.loc[long_entry, "entry_price"],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", size=10),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=tr.loc[short_entry, "entry_date"],
                    y=tr.loc[short_entry, "entry_price"],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", size=10),
                ),
                row=1,
                col=1,
            )

        if "exit_date" in tr.columns and "exit_price" in tr.columns:
            long_exit = (side == "LONG") & tr["exit_date"].notna()
            short_exit = (side == "SHORT") & tr["exit_date"].notna()
            fig.add_trace(
                go.Scatter(
                    x=tr.loc[long_exit, "exit_date"],
                    y=tr.loc[long_exit, "exit_price"],
                    mode="markers",
                    name="Long Exit",
                    marker=dict(symbol="circle", size=9),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=tr.loc[short_exit, "exit_date"],
                    y=tr.loc[short_exit, "exit_price"],
                    mode="markers",
                    name="Short Exit",
                    marker=dict(symbol="circle-open", size=9),
                ),
                row=1,
                col=1,
            )

        # pyramiding (optional)
        action = tr.get("action", tr.get("Action"))
        if action is not None and "entry_date" in tr.columns and "entry_price" in tr.columns:
            py = action.astype(str).str.upper().str.contains("PYRAMID")
            fig.add_trace(
                go.Scatter(
                    x=tr.loc[py, "entry_date"],
                    y=tr.loc[py, "entry_price"],
                    mode="markers",
                    name="Pyramiding",
                    marker=dict(symbol="diamond", size=9),
                ),
                row=1,
                col=1,
            )

    # Plot B
    fig.add_trace(go.Scatter(x=eq.index, y=eq[nav_col].astype(float), mode="lines", name=f"Equity ({nav_col})"), row=2, col=1)

    # Plot C
    if mom_col:
        fig.add_trace(go.Scatter(x=price.index, y=price[mom_col].astype(float), mode="lines", name=mom_col), row=3, col=1)
    if phase_col:
        fig.add_trace(go.Scatter(x=price.index, y=price[phase_col].astype(float), mode="lines", name=f"phase({phase_col})", line=dict(dash="dot")), row=3, col=1)
    if mkt_phase_col:
        fig.add_trace(go.Scatter(x=price.index, y=price[mkt_phase_col].astype(float), mode="lines", name=f"mkt_phase({mkt_phase_col})", line=dict(dash="dash")), row=3, col=1)

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    fig.update_yaxes(showline=True)

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html
