from __future__ import annotations

import numpy as np
import pandas as pd


def ma_cycle_phase(ema_fast: pd.Series, ema_mid: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """고지로(小次郎) 이동평균선 대순환(6단계) phase.

    Parameters
    ----------
    ema_fast : EMA(5)
    ema_mid  : EMA(20)
    ema_slow : EMA(40)

    Returns
    -------
    pd.Series
        phase in {1..6} (float dtype with NaN for warmup).

    Phase definition (top -> bottom):
      1) fast > mid > slow
      2) mid  > fast > slow
      3) mid  > slow > fast
      4) slow > mid  > fast
      5) slow > fast > mid
      6) fast > slow > mid
    """

    f = ema_fast.astype(float)
    m = ema_mid.astype(float)
    s = ema_slow.astype(float)

    out = pd.Series(np.nan, index=f.index, dtype=float)

    # 1
    mask = (f > m) & (m > s)
    out.loc[mask] = 1.0
    # 2
    mask = (m > f) & (f > s)
    out.loc[mask] = 2.0
    # 3
    mask = (m > s) & (s > f)
    out.loc[mask] = 3.0
    # 4
    mask = (s > m) & (m > f)
    out.loc[mask] = 4.0
    # 5
    mask = (s > f) & (f > m)
    out.loc[mask] = 5.0
    # 6
    mask = (f > s) & (s > m)
    out.loc[mask] = 6.0

    return out


def cycle_allows(side: int, phase: float | int | None) -> bool:
    """Cycle filter gate.

    - LONG allowed: phases {6,1,2}
    - SHORT allowed: phases {3,4,5}
    """
    if phase is None or (isinstance(phase, float) and np.isnan(phase)):
        return False
    p = int(phase)
    if side > 0:
        return p in (6, 1, 2)
    if side < 0:
        return p in (3, 4, 5)
    return False
