from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class TradeMode(str, Enum):
    """Directional mode for the engine."""

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

    # Type B: EMA(5/20) cross -> Donchian(10) breakout
    B_EMA_CROSS_DC10 = "B_EMA_CROSS_DC10"

    # Legacy single-window variants (kept for compatibility / debugging)
    A_20_PL = "A_20_PL"  # Donchian(20) with PL filter
    A_55 = "A_55"        # Donchian(55) without PL filter

    # Short patch: EMA(20/40) dead cross entry (SHORT only)
    SHORT_EMA20_40_DEAD = "SHORT_EMA20_40_DEAD"

    # Smoke-test helper: manual entry signals from data column `manual_entry`
    MANUAL = "MANUAL"

    # NOTE: 과거 회귀 기반 전략B는 본 세션 범위에서 폐기(deprecated)했다.


class TrailingStopType(str, Enum):
    # TS.A: % 기반 트레일링 스탑
    A_PCT = "TS.A"
    # TS.B: EMA5/20 cross 기반 청산
    B_EMA_CROSS = "TS.B"
    # TS.C: Darvas/Box support/resistance 기반 청산
    C_DARVAS_BOX = "TS.C"


class PyramidingType(str, Enum):
    OFF = "OFF"
    # PRMD.A: % 기준 피라미딩
    A_PCT = "PRMD.A"
    # PRMD.B: Darvas/20-box breakout 피라미딩 + cooldown
    B_DARVAS_BOX = "PRMD.B"


@dataclass(frozen=True)
class Order:
    """Market order executed at the next trading day's open (09:00)."""

    symbol: str
    side: Side            # LONG -> buy, SHORT -> short sell
    unit_shares: int
    atr_ref: float        # ATR10 computed at signal day (T)
    signal_date: str      # YYYY-MM-DD
    reason: str = ""


@dataclass
class Trade:
    """Trade summary record.

    Important
    ---------
    In trend-following with pyramiding, "entry price" is ambiguous:
      - the *first fill* price (for plotting / time-local debugging)
      - the *average entry* price across all pyramid fills (for PnL)

    We store both to avoid the classic "ghost entry" chart artifact.
    """

    symbol: str
    side: Side

    # First entry (immutable once created)
    entry_date: str
    entry_price: float

    # Aggregated position info (can change while the trade is open)
    shares: int
    avg_entry_price: float = 0.0
    num_entries: int = 1
    entry_notional_gross: float = 0.0  # sum(price * shares) across entry/pyramid fills

    # Exit
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    exit_reason: Optional[str] = None
