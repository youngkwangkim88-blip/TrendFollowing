from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (EMA).

    Notes
    -----
    - Uses pandas ewm with adjust=False (standard trading convention).
    - Returns a float series indexed like input.
    """
    n = int(span)
    if n <= 0:
        raise ValueError(f"EMA span must be positive. Got {span}")
    return series.astype(float).ewm(span=n, adjust=False).mean()


def momentum_pct(series: pd.Series, lookback: int) -> pd.Series:
    """Simple momentum as percentage change over `lookback` bars.

    MOM_L(t) = Close_t / Close_{t-L} - 1
    """
    lb = int(lookback)
    if lb <= 0:
        raise ValueError(f"lookback must be positive. Got {lookback}")
    s = series.astype(float)
    return s / s.shift(lb) - 1.0


def ma_cycle_phase_from_emas(
    ema_short: pd.Series,
    ema_mid: pd.Series,
    ema_long: pd.Series,
    *,
    eps: float = 1e-12,
) -> pd.Series:
    """Kojirō(小次郎) 'Moving Average Cycle' 6-phase classification using 3 EMAs.

    Phase mapping (top -> bottom):
      1) short > mid  > long
      2) mid   > short> long
      3) mid   > long > short
      4) long  > mid  > short
      5) long  > short> mid
      6) short > long > mid

    If any pair is effectively equal (|a-b|<=eps), phase is set to NaN.
    """
    s = ema_short.astype(float).values
    m = ema_mid.astype(float).values
    l = ema_long.astype(float).values

    phase = np.full(len(s), np.nan, dtype=float)

    tie = (np.abs(s - m) <= eps) | (np.abs(s - l) <= eps) | (np.abs(m - l) <= eps)

    cond1 = (s > m) & (m > l)
    cond2 = (m > s) & (s > l)
    cond3 = (m > l) & (l > s)
    cond4 = (l > m) & (m > s)
    cond5 = (l > s) & (s > m)
    cond6 = (s > l) & (l > m)

    phase[cond1 & ~tie] = 1.0
    phase[cond2 & ~tie] = 2.0
    phase[cond3 & ~tie] = 3.0
    phase[cond4 & ~tie] = 4.0
    phase[cond5 & ~tie] = 5.0
    phase[cond6 & ~tie] = 6.0

    return pd.Series(phase, index=ema_short.index)


def ma_cycle_phase(
    close: pd.Series,
    *,
    short: int = 5,
    mid: int = 20,
    long: int = 40,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Convenience wrapper: computes EMAs and returns (ema_short, ema_mid, ema_long, phase) as a DataFrame."""
    s = int(short)
    m = int(mid)
    l = int(long)
    ema_s = ema(close, span=s)
    ema_m = ema(close, span=m)
    ema_l = ema(close, span=l)
    phase = ma_cycle_phase_from_emas(ema_s, ema_m, ema_l, eps=eps)
    return pd.DataFrame(
        {
            f"ema_{s}": ema_s,
            f"ema_{m}": ema_m,
            f"ema_{l}": ema_l,
            f"cycle_phase_{s}_{m}_{l}": phase,
        },
        index=close.index,
    )
