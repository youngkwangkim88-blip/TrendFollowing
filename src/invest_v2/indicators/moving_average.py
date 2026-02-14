from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (EMA).

    Notes
    -----
    - Uses pandas `ewm(adjust=False)`.
    - For trend-following style signals, `adjust=False` is the common choice.
    """
    return series.astype(float).ewm(span=int(span), adjust=False).mean()
