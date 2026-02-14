from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _read_csv_with_date_index(path: str | Path, date_cols: tuple[str, ...] = ("date", "Date", "날짜")) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    date_col = None
    for c in date_cols:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found in {p}. Columns={list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df


def _classify_exit_reason(reason: str) -> str:
    """Map exit_reason strings to a compact category for legend/color coding."""
    r = (reason or "").strip().upper()
    if not r:
        return "EXIT(UNKNOWN)"
    if r.startswith("STOP_LOSS"):
        return "STOP_LOSS"
    if r.startswith("TS_A"):
        return "TS_A"
    if r.startswith("TS_C"):
        return "TS_C"
    if r.startswith("TS_B"):
        return "TS_B"
    if r.startswith("EVEN_STOP"):
        return "EVEN_STOP"
    if r.startswith("EMERGENCY_OPEN"):
        return "EMERGENCY_OPEN"
    if r.startswith("EMERGENCY_PREVCLOSE"):
        return "EMERGENCY_PREVCLOSE"
    if r.startswith("EMERGENCY_STOP"):
        return "EMERGENCY_C2C"
    if r.startswith("SHORT_MAX_HOLD"):
        return "FORCED_SHORT_MAXHOLD"
    return "EXIT(OTHER)"


def write_interactive_abc(
    plot_data_csv: str | Path,
    equity_curve_csv: str | Path,
    out_html: str | Path,
    trades_csv: Optional[str | Path] = None,
    fills_csv: Optional[str | Path] = None,
    title: str = "Trend Following Analysis",
    show_donchian: bool = False,
) -> Path:
    """A/B/C 패널을 생성하며, 진입/청산 및 피라미딩 마커를 표시합니다.

    Priority of data sources
    ------------------------
    - If `fills_csv` exists: use it for *all* markers (ENTRY / PYRAMID / EXIT).
      This avoids the classic pyramiding "ghost entry" artifact.
    - Else fallback to `trades_csv` for entry/exit and `equity_curve.pos_units` diff for pyramiding.
    """

    plot_df = _read_csv_with_date_index(plot_data_csv)
    eq_df = _read_csv_with_date_index(equity_curve_csv)

    # OHLC column normalization
    required_cols = {"O": "open", "H": "high", "L": "low", "C": "close"}
    for target, fallback in required_cols.items():
        if target not in plot_df.columns:
            if fallback in plot_df.columns:
                plot_df[target] = plot_df[fallback]
            elif target.lower() in plot_df.columns:
                plot_df[target] = plot_df[target.lower()]

    # equity nav
    nav_col = next((c for c in ("nav", "equity", "NAV", "total_value") if c in eq_df.columns), None)
    if nav_col is None:
        raise ValueError("Cannot find nav/equity column in equity curve.")
    nav = eq_df[nav_col].astype(float)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("Plot A: Candlestick & Trading Signals", "Plot B: Equity Curve", "Plot C: Phases"),
    )

    # --- Plot A: Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["O"],
            high=plot_df["H"],
            low=plot_df["L"],
            close=plot_df["C"],
            name="Candlestick",
            increasing_line_color="#FF0000",
            decreasing_line_color="#0000FF",
            legendgroup="group1",
            legendgrouptitle_text="Plot A: Price",
        ),
        row=1,
        col=1,
    )

    # EMA lines
    for ema_col, name in (("ema5", "EMA5"), ("ema20", "EMA20"), ("ema40", "EMA40")):
        if ema_col in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[ema_col],
                    mode="lines",
                    name=name,
                    line=dict(width=1.2),
                    opacity=0.7,
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )

    # --- Trading markers ---
    fills_path: Optional[Path] = None
    if fills_csv is not None and Path(fills_csv).exists():
        fills_path = Path(fills_csv)
    elif trades_csv is not None and Path(trades_csv).exists():
        maybe = Path(trades_csv).with_name("fills.csv")
        if maybe.exists():
            fills_path = maybe

    if fills_path is not None:
        fdf = pd.read_csv(fills_path)
        if "fill_date" in fdf.columns:
            fdf["fill_date"] = pd.to_datetime(fdf["fill_date"])
        if "side" in fdf.columns:
            fdf["side"] = fdf["side"].astype(int)
        if "fill_price" in fdf.columns:
            fdf["fill_price"] = fdf["fill_price"].astype(float)
        if "fill_type" not in fdf.columns:
            fdf["fill_type"] = ""
        if "reason" not in fdf.columns:
            fdf["reason"] = ""

        # Entries
        ent = fdf[fdf["fill_type"].astype(str).str.upper() == "ENTRY"].copy()
        if len(ent):
            long_e = ent[ent["side"] == 1]
            short_e = ent[ent["side"] == -1]
            fig.add_trace(
                go.Scatter(
                    x=long_e["fill_date"],
                    y=long_e["fill_price"],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", size=12, color="#2ecc71", line=dict(width=1, color="black")),
                    legendgroup="group1",
                    hovertext=long_e.apply(lambda r: f"ENTRY(LONG)<br>{r.get('reason','')}<br>px={float(r['fill_price']):,.2f}", axis=1),
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=short_e["fill_date"],
                    y=short_e["fill_price"],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", size=12, color="#e67e22", line=dict(width=1, color="black")),
                    legendgroup="group1",
                    hovertext=short_e.apply(lambda r: f"ENTRY(SHORT)<br>{r.get('reason','')}<br>px={float(r['fill_price']):,.2f}", axis=1),
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

        # Pyramiding
        pyr = fdf[fdf["fill_type"].astype(str).str.upper() == "PYRAMID"].copy()
        if len(pyr):
            fig.add_trace(
                go.Scatter(
                    x=pyr["fill_date"],
                    y=pyr["fill_price"],
                    mode="markers",
                    name="Pyramiding",
                    marker=dict(symbol="diamond", size=10, color="#3498db", line=dict(width=1, color="white")),
                    legendgroup="group1",
                    hovertext=pyr.apply(lambda r: f"PYRAMID<br>{r.get('reason','')}<br>px={float(r['fill_price']):,.2f}", axis=1),
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

        # Exits (color by exit category)
        ex = fdf[fdf["fill_type"].astype(str).str.upper() == "EXIT"].copy()
        if len(ex):
            ex["exit_cat"] = ex["reason"].astype(str).map(_classify_exit_reason)

            color_map = {
                "STOP_LOSS": "#e74c3c",
                "TS_A": "#3498db",
                "TS_C": "#1abc9c",
                "TS_B": "#7f8c8d",
                "EVEN_STOP": "#f1c40f",
                "EMERGENCY_OPEN": "#9b59b6",
                "EMERGENCY_PREVCLOSE": "#e67e22",
                "EMERGENCY_C2C": "#2c3e50",
                "FORCED_SHORT_MAXHOLD": "#000000",
                "EXIT(UNKNOWN)": "#95a5a6",
                "EXIT(OTHER)": "#95a5a6",
            }

            cat_order = [
                "STOP_LOSS",
                "TS_A",
                "TS_C",
                "TS_B",
                "EVEN_STOP",
                "EMERGENCY_OPEN",
                "EMERGENCY_PREVCLOSE",
                "EMERGENCY_C2C",
                "FORCED_SHORT_MAXHOLD",
                "EXIT(OTHER)",
                "EXIT(UNKNOWN)",
            ]

            ex["hover"] = ex.apply(
                lambda r: f"EXIT<br>Reason: {str(r.get('reason',''))}<br>px={float(r['fill_price']):,.2f}", axis=1
            )

            for cat in cat_order:
                sub = ex[ex["exit_cat"] == cat]
                if len(sub) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["fill_date"],
                        y=sub["fill_price"],
                        mode="markers",
                        name=f"Exit: {cat}",
                        marker=dict(symbol="x", size=9, color=color_map.get(cat, "#95a5a6"), line=dict(width=1, color="black")),
                        hovertext=sub["hover"],
                        hoverinfo="text",
                        legendgroup="group1",
                    ),
                    row=1,
                    col=1,
                )

    elif trades_csv and Path(trades_csv).exists():
        # Fallback legacy plotting from trades.csv
        tr = pd.read_csv(trades_csv)
        for dc in ("entry_date", "exit_date"):
            if dc in tr.columns:
                tr[dc] = pd.to_datetime(tr[dc])

        if {"entry_date", "entry_price", "side"}.issubset(tr.columns):
            long_e = tr[tr["side"].astype(int) == 1]
            short_e = tr[tr["side"].astype(int) == -1]
            fig.add_trace(
                go.Scatter(
                    x=long_e["entry_date"],
                    y=long_e["entry_price"],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", size=12, color="#2ecc71", line=dict(width=1, color="black")),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=short_e["entry_date"],
                    y=short_e["entry_price"],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", size=12, color="#e67e22", line=dict(width=1, color="black")),
                    legendgroup="group1",
                ),
                row=1,
                col=1,
            )

        if {"exit_date", "exit_price"}.issubset(tr.columns):
            ex = tr.dropna(subset=["exit_date", "exit_price"]).copy()
            if "exit_reason" not in ex.columns:
                ex["exit_reason"] = ""
            ex["exit_reason"] = ex["exit_reason"].fillna("").astype(str)
            if "side" in ex.columns:
                ex["side"] = ex["side"].astype(int)
            else:
                ex["side"] = 0

            ex["exit_cat"] = ex["exit_reason"].map(_classify_exit_reason)

            color_map = {
                "STOP_LOSS": "#e74c3c",
                "TS_A": "#3498db",
                "TS_C": "#1abc9c",
                "TS_B": "#7f8c8d",
                "EVEN_STOP": "#f1c40f",
                "EMERGENCY_OPEN": "#9b59b6",
                "EMERGENCY_PREVCLOSE": "#e67e22",
                "EMERGENCY_C2C": "#2c3e50",
                "FORCED_SHORT_MAXHOLD": "#000000",
                "EXIT(UNKNOWN)": "#95a5a6",
                "EXIT(OTHER)": "#95a5a6",
            }

            cat_order = [
                "STOP_LOSS",
                "TS_A",
                "TS_C",
                "TS_B",
                "EVEN_STOP",
                "EMERGENCY_OPEN",
                "EMERGENCY_PREVCLOSE",
                "EMERGENCY_C2C",
                "FORCED_SHORT_MAXHOLD",
                "EXIT(OTHER)",
                "EXIT(UNKNOWN)",
            ]

            def _side_name(v: int) -> str:
                if int(v) == 1:
                    return "LONG"
                if int(v) == -1:
                    return "SHORT"
                return "?"

            ex["hover"] = ex.apply(
                lambda r: f"{_side_name(int(r['side']))} Exit<br>Reason: {str(r['exit_reason'])}<br>Price: {float(r['exit_price']):,.2f}",
                axis=1,
            )

            for cat in cat_order:
                sub = ex[ex["exit_cat"] == cat]
                if len(sub) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["exit_date"],
                        y=sub["exit_price"],
                        mode="markers",
                        name=f"Exit: {cat}",
                        marker=dict(symbol="x", size=9, color=color_map.get(cat, "#95a5a6"), line=dict(width=1, color="black")),
                        hovertext=sub["hover"],
                        hoverinfo="text",
                        legendgroup="group1",
                    ),
                    row=1,
                    col=1,
                )

        # Pyramiding fallback (pos_units diff)
        if "pos_units" in eq_df.columns:
            pu = eq_df["pos_units"].astype(float)
            inc = (pu.diff() > 0.0) & (pu > 1.0)
            if inc.any():
                prmd_dates = pu.index[inc.fillna(False)]
                prmd_prices = plot_df["C"].reindex(prmd_dates).astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=prmd_dates,
                        y=prmd_prices,
                        mode="markers",
                        name="Pyramiding",
                        marker=dict(symbol="diamond", size=10, color="#3498db", line=dict(width=1, color="white")),
                        legendgroup="group1",
                    ),
                    row=1,
                    col=1,
                )

    # --- Plot B: Equity ---
    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=nav,
            mode="lines",
            name="Equity NAV",
            line=dict(color="#2c3e50"),
            legendgroup="group2",
            legendgrouptitle_text="Plot B: Account",
        ),
        row=2,
        col=1,
    )

    # --- Plot C: Phases ---
    if "cycle_phase" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["cycle_phase"],
                mode="lines",
                name="Cycle Phase",
                line=dict(dash="dot", color="#8e44ad"),
                legendgroup="group3",
                legendgrouptitle_text="Plot C: Market",
            ),
            row=3,
            col=1,
        )

    # layout
    fig.update_layout(
        title=title,
        height=1000,
        width=1600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left", yanchor="top", traceorder="grouped"),
        margin=dict(l=60, r=150, t=80, b=40),
        hovermode="x unified",
    )

    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)

    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out
