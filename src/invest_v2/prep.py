from __future__ import annotations

import pandas as pd

from invest_v2.indicators.atr import atr_ema
from invest_v2.indicators.donchian import donchian_high, donchian_low
from invest_v2.indicators.moving_average import ema
from invest_v2.indicators.ma_cycle import ma_cycle_phase


def add_indicators(df: pd.DataFrame, market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add all indicators required by Strategy A/B/C and optional market filter.

    Required input columns: open, high, low, close (float).

    Adds:
      - atr10
      - donchian_high/low_{10,20,55}
      - ema5/ema20/ema40
      - cycle_phase (Kojiro 6-phase)
      - mom63 (time-series momentum)
      - market_cycle_phase (optional, if market_df is provided)
    """

    out = df.copy()
    out["atr10"] = atr_ema(out, n=10)

    for w in (10, 20, 55):
        out[f"donchian_high_{w}"] = donchian_high(out, window=w)
        out[f"donchian_low_{w}"] = donchian_low(out, window=w)

    out["ema5"] = ema(out["close"], span=5)
    out["ema20"] = ema(out["close"], span=20)
    out["ema40"] = ema(out["close"], span=40)
    out["cycle_phase"] = ma_cycle_phase(out["ema5"], out["ema20"], out["ema40"])

    # 63 trading days ~ 3 months
    out["mom63"] = out["close"].astype(float) / out["close"].astype(float).shift(63) - 1.0

    if market_df is not None and len(market_df) > 0:
        m = market_df.copy()
        if "close" not in m.columns:
            raise ValueError("market_df must have a 'close' column")
        m["ema5"] = ema(m["close"], span=5)
        m["ema20"] = ema(m["close"], span=20)
        m["ema40"] = ema(m["close"], span=40)
        m["market_cycle_phase"] = ma_cycle_phase(m["ema5"], m["ema20"], m["ema40"])

        # align to symbol calendar (forward-fill is reasonable for non-trading gaps)
        out["market_cycle_phase"] = (
            m["market_cycle_phase"]
            .reindex(out.index, method="ffill")
            .astype(float)
        )
    else:
        out["market_cycle_phase"] = pd.NA

    return out
