from __future__ import annotations

import math


def tick_size_krx(price: float) -> int:
    """
    KRX tick size (approximate, common schedule).

    NOTE:
    - Tick rules can change; treat this as a replaceable policy.
    - For the toy system (005930 price band), this is sufficient.

    Typical schedule (KRW):
      < 1,000        : 1
      1,000-5,000    : 5
      5,000-10,000   : 10
      10,000-50,000  : 50
      50,000-100,000 : 100
      100,000-500,000: 500
      >= 500,000     : 1,000
    """
    p = float(price)
    if p < 1_000:
        return 1
    if p < 5_000:
        return 5
    if p < 10_000:
        return 10
    if p < 50_000:
        return 50
    if p < 100_000:
        return 100
    if p < 500_000:
        return 500
    return 1_000


def tick_down(price: float) -> float:
    t = tick_size_krx(price)
    return math.floor(price / t) * t


def tick_up(price: float) -> float:
    t = tick_size_krx(price)
    return math.ceil(price / t) * t
