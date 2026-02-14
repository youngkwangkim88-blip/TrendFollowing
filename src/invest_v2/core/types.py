from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class TradeMode(str, Enum):
    """Directional mode for the single-symbol engine."""

    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    LONG_SHORT = "LONG_SHORT"


class EntryRuleType(str, Enum):
    # Type A (Turtle)
    # NOTE: In this project spec, "Strategy A" is the *combined* rule:
    #   - A.1 Donchian(20)
    #   - A.2 PL filter applied to A.1
    #   - A.3 Donchian(55) which *ignores* PL filter (override)
    # Prefer A_TURTLE unless you intentionally want a legacy single-window variant.
    A_TURTLE = "A_TURTLE"

    # Type B (NEW): EMA(5/20) cross -> then Donchian(10) breakout
    #   - After GOLDEN cross, wait for 10-day Donchian high breakout -> LONG
    #   - After DEAD cross,   wait for 10-day Donchian low  breakout -> SHORT
    B_EMA_CROSS_DC10 = "B_EMA_CROSS_DC10"

    # Legacy single-window variants (kept for compatibility / debugging)
    A_20_PL = "A_20_PL"  # Donchian(20) with PL filter
    A_55 = "A_55"        # Donchian(55) without PL filter

    # Short patch: EMA(20/40) dead cross entry (SHORT only)
    #   - At T close, if EMA20 crosses below EMA40 -> schedule SHORT entry at T+1 open.
    SHORT_EMA20_40_DEAD = "SHORT_EMA20_40_DEAD"

    # NOTE: 과거 회귀 기반 전략B는 본 세션 범위에서 폐기(deprecated)했다.
    # 필요하면 별도 브랜치/파일로 분리해서 유지한다.


class TrailingStopType(str, Enum):
    # TS.A: % 기반 트레일링 스탑(기존 TS)
    A_PCT = "TS.A"
    # TS.B: EMA5/20 cross 기반 TS
    B_EMA_CROSS = "TS.B"
    # TS.C: Darvas/Box(최소 20일) support/resistance 기반 TS
    C_DARVAS_BOX = "TS.C"


class PyramidingType(str, Enum):
    OFF = "OFF"
    # PRMD.A: % 기준 피라미딩(기존)
    A_PCT = "PRMD.A"
    # PRMD.B: Darvas/20-box breakout 피라미딩 + cooldown
    B_DARVAS_BOX = "PRMD.B"


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
