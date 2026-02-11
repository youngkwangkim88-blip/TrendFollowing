from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_linreg_slope_r2(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Rolling OLS on y=close, x=0..window-1.
    Returns (slope, r2). slope unit: KRW/day.
    """
    y = close.astype(float).values
    n = int(window)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_demean = x - x_mean
    sxx = np.sum(x_demean**2)

    slopes = np.full_like(y, fill_value=np.nan, dtype=float)
    r2s = np.full_like(y, fill_value=np.nan, dtype=float)

    for i in range(n - 1, len(y)):
        yi = y[i - n + 1 : i + 1]
        y_mean = yi.mean()
        y_demean = yi - y_mean
        sxy = np.sum(x_demean * y_demean)
        slope = sxy / sxx
        intercept = y_mean - slope * x_mean

        y_hat = intercept + slope * x
        ss_tot = np.sum((yi - y_mean) ** 2)
        ss_res = np.sum((yi - y_hat) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        slopes[i] = slope
        r2s[i] = r2

    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


def normalized_slope_pct(slope: pd.Series, close: pd.Series) -> pd.Series:
    return (slope / close.astype(float)) * 100.0


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=window).mean()
