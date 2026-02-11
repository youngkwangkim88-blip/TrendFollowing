from __future__ import annotations

import pandas as pd


def donchian_high(df: pd.DataFrame, window: int) -> pd.Series:
    # lookahead-safe: use highs from t-window .. t-1
    return df["high"].shift(1).rolling(window=window, min_periods=window).max()


def donchian_low(df: pd.DataFrame, window: int) -> pd.Series:
    # lookahead-safe: use lows from t-window .. t-1
    return df["low"].shift(1).rolling(window=window, min_periods=window).min()
