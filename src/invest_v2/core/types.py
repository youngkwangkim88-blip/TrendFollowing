from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class EntryRuleType(str, Enum):
    # -----------------
    # Type A (Turtle)
    # -----------------
    # NOTE: In this project spec, "Strategy A" is the *combined* rule:
    #   - A.1 Donchian(20)
    #   - A.2 PL filter applied to A.1
    #   - A.3 Donchian(55) which ignores PL filter (override)
    # Prefer A_TURTLE unless you intentionally want a legacy single-window variant.
    A_TURTLE = "A_TURTLE"

    # Legacy single-window variants (kept for compatibility / debugging)
    A_20_PL = "A_20_PL"  # Donchian(20) with PL filter
    A_55 = "A_55"        # Donchian(55) without PL filter

    # ---------------------
    # Type B (Regression)
    # ---------------------
    B_SLOPE20 = "B_SLOPE20"  # legacy strict: (0-cross) & (r2>0.6) & (MA60 filter)

    # Type B variants (added after observing 0-signal issue on 005930)
    BA_REGIME20 = "BA_REGIME20"            # Regime entry: ns>0 & r2>thr & MA60 (enter when regime turns ON)
    BB_CROSS20_R2PREV = "BB_CROSS20_R2PREV"  # 0-cross + r2(t-1) filter

    # ---------------------
    # Strategy C
    # ---------------------
    C_TSMOM_CYCLE = "C_TSMOM_CYCLE"  # Time-series momentum + Moving Average Cycle(6-phase) filter


class TrailingStopType(str, Enum):
    """Trailing stop / exit policy selector.

    Naming follows user-facing convention:
      - TS.A: percentage-based trailing stop (current default)
      - TS.B: EMA5/20 cross exit (dead-cross for long, golden-cross for short)
      - TS.C: Darvas/box-style S/R breakout trailing stop (min 20 days)
    """

    TS_A = "TS.A"
    TS_B = "TS.B"
    TS_C = "TS.C"


class PyramidingType(str, Enum):
    """Pyramiding policy selector.

    - PRMD.A: percentage trigger (legacy)
    - PRMD.B: Darvas/box breakout pyramiding (min 20 days) + 5-day cooldown
    """

    PRMD_A = "PRMD.A"
    PRMD_B = "PRMD.B"


@dataclass(frozen=True)
class Order:
    # Market order executed at today's open (09:00).
    symbol: str
    side: Side            # LONG -> buy, SHORT -> short sell
    unit_shares: int
    atr_ref: float        # ATR10 computed at signal day (T)
    signal_date: str      # YYYY-MM-DD
    reason: str = ""


@dataclass
class Trade:
    symbol: str
    side: Side
    entry_date: str
    entry_price: float
    shares: int
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    exit_reason: Optional[str] = None
