from __future__ import annotations

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close_prev = df["close"].astype(float).shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr_ema(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    ATR(n) with EMA smoothing.
    """
    tr = true_range(df)
    return ema(tr, span=n)
