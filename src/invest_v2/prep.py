from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd

from invest_v2.indicators.atr import atr_ema
from invest_v2.indicators.donchian import donchian_high, donchian_low
from invest_v2.indicators.regression import rolling_linreg_slope_r2, normalized_slope_pct, sma
from invest_v2.indicators.moving_average_cycle import momentum_pct, ma_cycle_phase


def add_indicators(
    df: pd.DataFrame,
    *,
    mom_windows: Tuple[int, ...] = (63,),
    cycle_ema_windows: Tuple[int, int, int] = (5, 20, 40),
    market_df: Optional[pd.DataFrame] = None,
    market_prefix: str = "mkt",
) -> pd.DataFrame:
    """Add indicators used by entry rules and risk management.

    Parameters
    ----------
    df:
        OHLC(V) with datetime index and columns: open, high, low, close.
    mom_windows:
        Momentum lookbacks to compute as percentage change: `mom_{L}`.
    cycle_ema_windows:
        EMA windows (short, mid, long) for 'moving average cycle' phase:
        `cycle_phase_{s}_{m}_{l}`.
    market_df:
        Optional market index OHLC(V). If provided, market cycle phase is computed
        and joined into `df` with prefix `market_prefix` (default: 'mkt').
    market_prefix:
        Prefix for market columns.

    Notes
    -----
    - This function is backward-compatible: existing columns remain unchanged.
    - For market joins, we align by index and forward-fill from the last available market value
      (no look-ahead).
    """
    out = df.copy()

    # Risk management base indicator
    out["atr10"] = atr_ema(out, n=10)

    # Strategy A indicators
    out["donchian_high_20"] = donchian_high(out, window=20)
    out["donchian_low_20"] = donchian_low(out, window=20)
    out["donchian_high_55"] = donchian_high(out, window=55)
    out["donchian_low_55"] = donchian_low(out, window=55)

    # Strategy B indicators
    slope20, r2_20 = rolling_linreg_slope_r2(out["close"], window=20)
    out["slope_20"] = slope20
    out["r2_20"] = r2_20
    out["norm_slope_20"] = normalized_slope_pct(slope20, out["close"])
    out["ma60"] = sma(out["close"], window=60)

    # Strategy C indicators: momentum + MA cycle phase
    for w in mom_windows:
        ww = int(w)
        out[f"mom_{ww}"] = momentum_pct(out["close"], lookback=ww)

    s, m, l = (int(cycle_ema_windows[0]), int(cycle_ema_windows[1]), int(cycle_ema_windows[2]))
    cycle_df = ma_cycle_phase(out["close"], short=s, mid=m, long=l)
    out = out.join(cycle_df)

    # Market (index/futures) cycle phase for global regime filter
    if market_df is not None:
        if "close" not in market_df.columns:
            raise ValueError("market_df must include 'close' column.")
        mkt_cycle = ma_cycle_phase(market_df["close"], short=s, mid=m, long=l)
        # keep only close + phase (EMAs are optional but useful for debugging)
        mkt_cols = pd.DataFrame(
            {
                f"{market_prefix}_close": market_df["close"].astype(float),
                f"{market_prefix}_cycle_phase_{s}_{m}_{l}": mkt_cycle[f"cycle_phase_{s}_{m}_{l}"],
            },
            index=market_df.index,
        )
        # Align to out.index and forward-fill safely
        mkt_cols = mkt_cols.sort_index().reindex(out.index, method="ffill")
        out = out.join(mkt_cols)

    return out
