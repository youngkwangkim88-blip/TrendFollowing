from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class EntryRuleType(str, Enum):
    A_20_PL = "A_20_PL"
    A_55 = "A_55"
    B_SLOPE20 = "B_SLOPE20"


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
