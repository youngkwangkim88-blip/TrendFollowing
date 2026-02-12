from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Figure defaults
# -----------------------------
# Single panel: 16x9 @ 120dpi ~= 1920x1080
HD_FIGSIZE: Tuple[float, float] = (16.0, 9.0)
HD_DPI: int = 120

# Bundle panel (A/B/C stacked): slightly taller than HD for readability.
# 16x12 @ 120dpi ~= 1920x1440
BUNDLE_FIGSIZE: Tuple[float, float] = (16.0, 12.0)


def _to_dt_index(idx: Iterable[Any]) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Index(idx))


def _maybe_get_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg):
        return asdict(cfg)
    # fallback: best-effort
    return {k: getattr(cfg, k) for k in dir(cfg) if k and not k.startswith("_") and not callable(getattr(cfg, k))}


# ============================================================
# Legacy standalone plots (kept for compatibility)
# ============================================================


def plot_equity_curve(equity: pd.DataFrame, outpath: str) -> None:
    """Plot NAV curve."""
    nav = equity["nav"].astype(float)
    fig = plt.figure(figsize=HD_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(_to_dt_index(equity.index), nav.values, label="NAV")
    ax.set_title("Equity Curve (NAV)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (KRW)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=HD_DPI)
    plt.close(fig)


def plot_price_with_signals(
    df: pd.DataFrame,
    equity: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    *,
    symbol: str,
    entry_rule: str,
    cfg: Any | None = None,
    outpath: str,
) -> None:
    """Plot A: adjusted close + EMA(5/20/40) + trade/position markers."""
    fig = plt.figure(figsize=HD_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1)
    _plot_A(ax, df=df, equity=equity, trades=trades, symbol=symbol, entry_rule=entry_rule, cfg=cfg)
    fig.tight_layout()
    fig.savefig(outpath, dpi=HD_DPI)
    plt.close(fig)


def plot_momentum_with_phases(
    df: pd.DataFrame,
    *,
    symbol: str,
    cfg: Any | None = None,
    market_prefix: str = "mkt",
    outpath: str,
) -> None:
    """Plot C: Momentum + (optional) MA-cycle phases."""
    fig = plt.figure(figsize=HD_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1)
    _plot_C(ax, df=df, symbol=symbol, cfg=cfg, market_prefix=market_prefix)
    fig.tight_layout()
    fig.savefig(outpath, dpi=HD_DPI)
    plt.close(fig)


# ============================================================
# New: bundled A/B/C in a single PNG + year chunking
# ============================================================


def _slice_year_chunks(index: pd.DatetimeIndex, years_per_chunk: int = 4) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0:
        return []

    years = int(years_per_chunk)
    if years <= 0:
        raise ValueError("years_per_chunk must be positive")

    start = pd.Timestamp(index.min().year, 1, 1)
    end = index.max()

    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        nxt = cur + pd.DateOffset(years=years)
        # inclusive end
        chunk_end = min(nxt - pd.Timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = nxt

    # Remove empty chunks (where index has no dates)
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in chunks:
        if ((index >= a) & (index <= b)).any():
            out.append((a, b))
    return out


def plot_bundle_abc(
    df: pd.DataFrame,
    equity: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    *,
    symbol: str,
    entry_rule: str,
    cfg: Any | None = None,
    market_prefix: str = "mkt",
    outpath: str,
) -> None:
    """Save a single PNG with Plot A/B/C stacked vertically."""
    price = df.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = _to_dt_index(price.index)

    eq = equity.copy()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = _to_dt_index(eq.index)
    eq = eq.sort_index().reindex(price.index, method="ffill")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=BUNDLE_FIGSIZE)
    axA, axB, axC = axes[0], axes[1], axes[2]

    _plot_A(axA, df=price, equity=eq, trades=trades, symbol=symbol, entry_rule=entry_rule, cfg=cfg)

    # Plot B
    nav = eq["nav"].astype(float)
    axB.plot(price.index, nav.values, label="NAV")
    axB.set_title("Plot B | Equity Curve")
    axB.set_ylabel("NAV (KRW)")
    axB.grid(True)
    axB.legend(loc="best", fontsize=9)

    _plot_C(axC, df=price, symbol=symbol, cfg=cfg, market_prefix=market_prefix)

    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=HD_DPI)
    plt.close(fig)


def plot_bundle_abc_year_chunks(
    df: pd.DataFrame,
    equity: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    *,
    symbol: str,
    entry_rule: str,
    cfg: Any | None = None,
    market_prefix: str = "mkt",
    outdir: str,
    years_per_chunk: int = 4,
    filename_prefix: str = "plot_ABC",
) -> List[str]:
    """Create multiple bundled PNGs, split into N-year chunks.

    Returns
    -------
    List of saved file paths.
    """
    price = df.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = _to_dt_index(price.index)

    eq = equity.copy()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = _to_dt_index(eq.index)
    eq = eq.sort_index().reindex(price.index, method="ffill")

    chunks = _slice_year_chunks(price.index, years_per_chunk=years_per_chunk)
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for a, b in chunks:
        mask = (price.index >= a) & (price.index <= b)
        p2 = price.loc[mask].copy()
        e2 = eq.loc[mask].copy()

        # Trades slice (entries/exits within the chunk)
        t2 = None
        if trades is not None and len(trades) > 0:
            t2 = trades.copy()
            if "entry_date" in t2.columns:
                t2["entry_date"] = pd.to_datetime(t2["entry_date"])
            if "exit_date" in t2.columns:
                t2["exit_date"] = pd.to_datetime(t2["exit_date"])
            # Keep trades that intersect the chunk interval
            if "entry_date" in t2.columns and "exit_date" in t2.columns:
                t2 = t2[(t2["entry_date"] <= b) & (t2["exit_date"] >= a)].copy()

        name = f"{filename_prefix}_{int(a.year)}-{int(b.year)}.png"
        outpath = str(outdir_p / name)
        plot_bundle_abc(
            p2,
            e2,
            t2,
            symbol=symbol,
            entry_rule=entry_rule,
            cfg=cfg,
            market_prefix=market_prefix,
            outpath=outpath,
        )
        saved.append(outpath)

    return saved


# ============================================================
# Internals: axis-level plotting
# ============================================================


def _plot_A(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    equity: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    symbol: str,
    entry_rule: str,
    cfg: Any | None = None,
) -> None:
    """Axis Plot A implementation."""
    _cfg = _maybe_get_cfg(cfg)

    # EMA windows used by phase filter (fallback 5/20/40)
    ema_s = int(_cfg.get("c_cycle_ema_short", 5) or 5)
    ema_m = int(_cfg.get("c_cycle_ema_mid", 20) or 20)
    ema_l = int(_cfg.get("c_cycle_ema_long", 40) or 40)

    price = df.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = _to_dt_index(price.index)

    eq = equity.copy()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = _to_dt_index(eq.index)
    eq = eq.sort_index().reindex(price.index, method="ffill")

    # Ensure EMA columns exist
    for w in (ema_s, ema_m, ema_l):
        col = f"ema_{w}"
        if col not in price.columns:
            price[col] = price["close"].astype(float).ewm(span=int(w), adjust=False).mean()

    ax.plot(price.index, price["close"].astype(float).values, label="Close (Adj)")
    ax.plot(price.index, price[f"ema_{ema_s}"].astype(float).values, label=f"EMA{ema_s}")
    ax.plot(price.index, price[f"ema_{ema_m}"].astype(float).values, label=f"EMA{ema_m}")
    ax.plot(price.index, price[f"ema_{ema_l}"].astype(float).values, label=f"EMA{ema_l}")

    # Strategy A: Donchian support/resistance overlays
    if str(entry_rule).startswith("A_"):
        if "donchian_high_20" in price.columns and "donchian_low_20" in price.columns:
            ax.plot(price.index, price["donchian_high_20"].astype(float).values, label="Donchian20 Resistance")
            ax.plot(price.index, price["donchian_low_20"].astype(float).values, label="Donchian20 Support")
        if "donchian_high_55" in price.columns and "donchian_low_55" in price.columns and str(entry_rule) in ("A_TURTLE", "A_55"):
            ax.plot(
                price.index,
                price["donchian_high_55"].astype(float).values,
                linestyle="--",
                label="Donchian55 Resistance",
            )
            ax.plot(
                price.index,
                price["donchian_low_55"].astype(float).values,
                linestyle="--",
                label="Donchian55 Support",
            )

    # ------------------------
    # Markers: entry/exit/pyr
    # ------------------------

    # 1) pyramiding markers (from equity pos_units increments)
    if "pos_side" in eq.columns and "pos_units" in eq.columns:
        side = eq["pos_side"].astype(float)
        units = eq["pos_units"].astype(float)
        units_prev = units.shift(1)
        side_prev = side.shift(1)

        pyr_long = (side_prev == 1) & (side == 1) & (units > units_prev)
        pyr_short = (side_prev == -1) & (side == -1) & (units > units_prev)

        if pyr_long.any():
            dts = price.index[pyr_long.fillna(False)]
            ax.scatter(dts, price.loc[dts, "open"].astype(float).values, marker="D", label="Pyramiding (Long)")
        if pyr_short.any():
            dts = price.index[pyr_short.fillna(False)]
            ax.scatter(dts, price.loc[dts, "open"].astype(float).values, marker="D", label="Pyramiding (Short)")

    # 2) entry/exit markers
    used_long_entry_dates: set[pd.Timestamp] = set()
    used_short_entry_dates: set[pd.Timestamp] = set()

    if (
        trades is not None
        and len(trades) > 0
        and {"side", "entry_date", "entry_price", "exit_date", "exit_price"}.issubset(trades.columns)
    ):
        td = trades.copy()
        td["entry_date"] = pd.to_datetime(td["entry_date"], errors="coerce")
        td["exit_date"] = pd.to_datetime(td["exit_date"], errors="coerce")
        td["side"] = td["side"].astype(int)
        td["entry_price"] = pd.to_numeric(td["entry_price"], errors="coerce")
        td["exit_price"] = pd.to_numeric(td["exit_price"], errors="coerce")

        # Filter clearly invalid prices (e.g., 0 due to missing OHLC)
        td = td[(td["entry_price"] > 0) & (td["exit_price"] > 0)].copy()

        long_tr = td[td["side"] == 1]
        short_tr = td[td["side"] == -1]

        if len(long_tr) > 0:
            ax.scatter(long_tr["entry_date"], long_tr["entry_price"], marker="^", label="Long Entry")
            ax.scatter(long_tr["exit_date"], long_tr["exit_price"], marker="o", label="Long Exit")
            used_long_entry_dates = set(long_tr["entry_date"].dropna().tolist())

        if len(short_tr) > 0:
            ax.scatter(short_tr["entry_date"], short_tr["entry_price"], marker="v", label="Short Entry")
            ax.scatter(
                short_tr["exit_date"],
                short_tr["exit_price"],
                marker="o",
                facecolors="none",
                label="Short Exit",
            )
            used_short_entry_dates = set(short_tr["entry_date"].dropna().tolist())

    # Fallback/augment: detect entries not present in trades (e.g., last open trade)
    if "pos_side" in eq.columns:
        side = eq["pos_side"].astype(float)
        side_prev = side.shift(1)

        long_ent = (side_prev == 0) & (side == 1)
        short_ent = (side_prev == 0) & (side == -1)

        if long_ent.any():
            dts = price.index[long_ent.fillna(False)]
            dts = [d for d in dts if d not in used_long_entry_dates]
            if len(dts) > 0:
                ax.scatter(dts, price.loc[dts, "open"].astype(float).values, marker="^", label="Long Entry (EOD pos)")

        if short_ent.any():
            dts = price.index[short_ent.fillna(False)]
            dts = [d for d in dts if d not in used_short_entry_dates]
            if len(dts) > 0:
                ax.scatter(dts, price.loc[dts, "open"].astype(float).values, marker="v", label="Short Entry (EOD pos)")

        # Exits (equity transition). If trades were present, exits are already displayed with exact exit_price.
        if trades is None or len(trades) == 0:
            long_ex = (side_prev == 1) & (side == 0)
            short_ex = (side_prev == -1) & (side == 0)

            if long_ex.any():
                dts = price.index[long_ex.fillna(False)]
                ax.scatter(dts, price.loc[dts, "open"].astype(float).values, marker="o", label="Long Exit (EOD pos)")

            if short_ex.any():
                dts = price.index[short_ex.fillna(False)]
                ax.scatter(
                    dts,
                    price.loc[dts, "open"].astype(float).values,
                    marker="o",
                    facecolors="none",
                    label="Short Exit (EOD pos)",
                )

    ax.set_title(f"Plot A | {symbol} | {entry_rule}")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend(loc="best", fontsize=9)


def _plot_C(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    symbol: str,
    cfg: Any | None = None,
    market_prefix: str = "mkt",
) -> None:
    """Axis Plot C implementation."""
    _cfg = _maybe_get_cfg(cfg)

    mom_w = int(_cfg.get("c_mom_window", 63) or 63)
    enter_thr = _cfg.get("c_enter_thr", None)
    exit_thr = _cfg.get("c_exit_thr", None)
    ema_s = int(_cfg.get("c_cycle_ema_short", 5) or 5)
    ema_m = int(_cfg.get("c_cycle_ema_mid", 20) or 20)
    ema_l = int(_cfg.get("c_cycle_ema_long", 40) or 40)

    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = _to_dt_index(data.index)

    mom_col = f"mom_{mom_w}"
    if mom_col not in data.columns:
        data[mom_col] = data["close"].astype(float) / data["close"].astype(float).shift(mom_w) - 1.0

    phase_col = f"cycle_phase_{ema_s}_{ema_m}_{ema_l}"
    mkt_phase_col = f"{market_prefix}_cycle_phase_{ema_s}_{ema_m}_{ema_l}"

    ax.plot(data.index, data[mom_col].astype(float).values, label=f"Momentum {mom_w}d")
    ax.axhline(0.0, linestyle="--", linewidth=1)

    if isinstance(enter_thr, (int, float)):
        ax.axhline(float(enter_thr), linestyle=":", linewidth=1, label=f"Enter thr ({float(enter_thr):.2f})")
    if isinstance(exit_thr, (int, float)) and float(exit_thr) != 0.0:
        ax.axhline(float(exit_thr), linestyle=":", linewidth=1, label=f"Exit thr ({float(exit_thr):.2f})")

    ax.set_title(f"Plot C | {symbol} | Momentum + Cycle phase")
    ax.set_xlabel("Date")
    ax.set_ylabel("Momentum")
    ax.grid(True)

    has_phase = phase_col in data.columns
    has_mkt_phase = mkt_phase_col in data.columns
    if has_phase or has_mkt_phase:
        ax2 = ax.twinx()
        if has_phase:
            ax2.step(data.index, data[phase_col].astype(float).values, where="post", label="Stock phase")
        if has_mkt_phase:
            ax2.step(
                data.index,
                data[mkt_phase_col].astype(float).values,
                where="post",
                linestyle="--",
                label="Market phase",
            )
        ax2.set_ylabel("Cycle Phase (1..6)")
        ax2.set_ylim(0.5, 6.5)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    else:
        ax.legend(loc="best", fontsize=9)
