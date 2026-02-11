from __future__ import annotations

import pandas as pd

from invest_v2.indicators.atr import atr_ema
from invest_v2.indicators.donchian import donchian_high, donchian_low
from invest_v2.indicators.regression import rolling_linreg_slope_r2, normalized_slope_pct, sma


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr10"] = atr_ema(out, n=10)

    out["donchian_high_20"] = donchian_high(out, window=20)
    out["donchian_low_20"] = donchian_low(out, window=20)
    out["donchian_high_55"] = donchian_high(out, window=55)
    out["donchian_low_55"] = donchian_low(out, window=55)

    slope20, r2_20 = rolling_linreg_slope_r2(out["close"], window=20)
    out["slope_20"] = slope20
    out["r2_20"] = r2_20
    out["norm_slope_20"] = normalized_slope_pct(slope20, out["close"])
    out["ma60"] = sma(out["close"], window=60)

    return out
