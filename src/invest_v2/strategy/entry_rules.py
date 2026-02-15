from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from invest_v2.core.types import EntryRuleType, Side


@dataclass(frozen=True)
class EntryDecision:
    """Entry decision evaluated at day T close.

    The backtest engine will convert this decision into an order executed at day T+1 open.

    Attributes
    ----------
    side:
        LONG / SHORT / FLAT
    reason:
        A short label used in order.reason / reports.
    bypass_filter_pl:
        If True, Filter A(PL filter) will not be applied to this decision.
        (Used for Strategy A's Donchian55 override.)
    """

    side: Side
    reason: str = ""
    bypass_filter_pl: bool = False


@dataclass
class EntryContext:
    # Filter A (PL) state
    last_trade_side: Optional[Side] = None
    last_trade_pnl: Optional[float] = None

    # Strategy B state: remember the latest EMA cross regime
    last_ema_cross_side: Optional[Side] = None
    last_ema_cross_index: Optional[int] = None


class EntryRule:
    def __init__(self, rule_type: EntryRuleType):
        self.rule_type = rule_type

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> EntryDecision:
        raise NotImplementedError


class TurtleRule(EntryRule):
    """Legacy single-window Donchian breakout."""

    def __init__(self, rule_type: EntryRuleType, window: int, bypass_pl: bool = False):
        super().__init__(rule_type)
        self.window = int(window)
        self.bypass_pl = bool(bypass_pl)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> EntryDecision:
        if i <= 0:
            return EntryDecision(Side.FLAT)

        c = float(df["close"].iloc[i])
        dh = df.get(f"donchian_high_{self.window}", pd.Series(index=df.index)).iloc[i]
        dl = df.get(f"donchian_low_{self.window}", pd.Series(index=df.index)).iloc[i]
        if pd.isna(dh) or pd.isna(dl):
            return EntryDecision(Side.FLAT)

        if c >= float(dh):
            return EntryDecision(Side.LONG, reason=f"ENTRY_A_{self.window}", bypass_filter_pl=self.bypass_pl)
        if c <= float(dl):
            return EntryDecision(Side.SHORT, reason=f"ENTRY_A_{self.window}", bypass_filter_pl=self.bypass_pl)
        return EntryDecision(Side.FLAT)


class TurtleComboRule(EntryRule):
    """Strategy A: A.1(20) + A.2(PL filter) + A.3(55 override).

    Priority
    --------
    1) Donchian(55) breakout: ALWAYS take (bypasses PL filter)
    2) Else Donchian(20) breakout: subject to PL filter (Filter A)
    """

    def __init__(self, donchian_window_20: int = 20, donchian_window_55: int = 55):
        super().__init__(EntryRuleType.A_TURTLE)
        self.w20 = int(donchian_window_20)
        self.w55 = int(donchian_window_55)

    def _donchian_signal(self, df: pd.DataFrame, i: int, window: int) -> Side:
        c = float(df["close"].iloc[i])
        dh = df.get(f"donchian_high_{window}", pd.Series(index=df.index)).iloc[i]
        dl = df.get(f"donchian_low_{window}", pd.Series(index=df.index)).iloc[i]
        if pd.isna(dh) or pd.isna(dl):
            return Side.FLAT
        if c >= float(dh):
            return Side.LONG
        if c <= float(dl):
            return Side.SHORT
        return Side.FLAT

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> EntryDecision:
        if i <= 0:
            return EntryDecision(Side.FLAT)

        # A.3: Donchian55 (override)
        sig55 = self._donchian_signal(df, i, self.w55)
        if sig55 != Side.FLAT:
            return EntryDecision(sig55, reason="ENTRY_A_55", bypass_filter_pl=True)

        # A.1: Donchian20 (subject to PL filter)
        sig20 = self._donchian_signal(df, i, self.w20)
        if sig20 != Side.FLAT:
            return EntryDecision(sig20, reason="ENTRY_A_20", bypass_filter_pl=False)

        return EntryDecision(Side.FLAT)


class EmaCrossDonchianRule(EntryRule):
    """Strategy B: EMA(5/20) cross -> then Donchian breakout.

    Base rule
    ---------
    - After GOLDEN cross (ema5 crosses above ema20), wait until C >= DonchianHigh(10) -> LONG.
    - After DEAD cross (ema5 crosses below ema20), wait until C <= DonchianLow(10)  -> SHORT.

    Override (Filter A bypass)
    --------------------------
    - If the breakout happens on a wider Donchian window (default=20), this entry may bypass Filter A.

    NOTE
    ----
    We treat "이후" literally: breakout entry is allowed only on days strictly after the cross day.
    """

    def __init__(
        self,
        ema_fast: str = "ema5",
        ema_mid: str = "ema20",
        donchian_window: int = 10,
        donchian_override_window: int = 20,
    ):
        super().__init__(EntryRuleType.B_EMA_CROSS_DC10)
        self.ema_fast = str(ema_fast)
        self.ema_mid = str(ema_mid)
        self.w = int(donchian_window)
        self.w_ovr = int(donchian_override_window)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> EntryDecision:
        if i <= 1:
            return EntryDecision(Side.FLAT)

        f_prev = df.get(self.ema_fast, pd.Series(index=df.index)).iloc[i - 1]
        m_prev = df.get(self.ema_mid, pd.Series(index=df.index)).iloc[i - 1]
        f = df.get(self.ema_fast, pd.Series(index=df.index)).iloc[i]
        m = df.get(self.ema_mid, pd.Series(index=df.index)).iloc[i]
        if pd.isna(f_prev) or pd.isna(m_prev) or pd.isna(f) or pd.isna(m):
            return EntryDecision(Side.FLAT)

        # Detect cross
        golden = float(f_prev) <= float(m_prev) and float(f) > float(m)
        dead = float(f_prev) >= float(m_prev) and float(f) < float(m)
        if golden:
            ctx.last_ema_cross_side = Side.LONG
            ctx.last_ema_cross_index = i
            return EntryDecision(Side.FLAT)
        if dead:
            ctx.last_ema_cross_side = Side.SHORT
            ctx.last_ema_cross_index = i
            return EntryDecision(Side.FLAT)

        if ctx.last_ema_cross_side is None or ctx.last_ema_cross_index is None:
            return EntryDecision(Side.FLAT)
        if i <= int(ctx.last_ema_cross_index):
            return EntryDecision(Side.FLAT)

        c = float(df["close"].iloc[i])

        # Evaluate override window first (bypass Filter A)
        dh_ovr = df.get(f"donchian_high_{self.w_ovr}", pd.Series(index=df.index)).iloc[i]
        dl_ovr = df.get(f"donchian_low_{self.w_ovr}", pd.Series(index=df.index)).iloc[i]

        # Then the base window (subject to Filter A)
        dh = df.get(f"donchian_high_{self.w}", pd.Series(index=df.index)).iloc[i]
        dl = df.get(f"donchian_low_{self.w}", pd.Series(index=df.index)).iloc[i]

        if pd.isna(dh) or pd.isna(dl):
            return EntryDecision(Side.FLAT)

        if ctx.last_ema_cross_side == Side.LONG:
            if not pd.isna(dh_ovr) and c >= float(dh_ovr):
                return EntryDecision(Side.LONG, reason="ENTRY_B_GOLDEN_DC20_OVR", bypass_filter_pl=True)
            if c >= float(dh):
                return EntryDecision(Side.LONG, reason="ENTRY_B_GOLDEN_DC10", bypass_filter_pl=False)
        elif ctx.last_ema_cross_side == Side.SHORT:
            if not pd.isna(dl_ovr) and c <= float(dl_ovr):
                return EntryDecision(Side.SHORT, reason="ENTRY_B_DEAD_DC20_OVR", bypass_filter_pl=True)
            if c <= float(dl):
                return EntryDecision(Side.SHORT, reason="ENTRY_B_DEAD_DC10", bypass_filter_pl=False)

        return EntryDecision(Side.FLAT)


class ShortEma2040DeadCrossRule(EntryRule):
    """SHORT entry on EMA(20/40) dead cross.

    Rule
    ----
    - At day T close:
        if EMA20 crosses below EMA40 (dead cross), emit SHORT entry decision.

    Notes
    -----
    - Exit is handled elsewhere (e.g., TS.B EMA5/20 golden-cross exit).
    - This rule is intentionally *one-sided* (SHORT-only). It never emits LONG.
    """

    def __init__(self, ema_fast: str = "ema20", ema_slow: str = "ema40"):
        super().__init__(EntryRuleType.SHORT_EMA20_40_DEAD)
        self.ema_fast = str(ema_fast)
        self.ema_slow = str(ema_slow)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> EntryDecision:
        if i <= 0:
            return EntryDecision(Side.FLAT)

        f_prev = df.get(self.ema_fast, pd.Series(index=df.index)).iloc[i - 1]
        s_prev = df.get(self.ema_slow, pd.Series(index=df.index)).iloc[i - 1]
        f = df.get(self.ema_fast, pd.Series(index=df.index)).iloc[i]
        s = df.get(self.ema_slow, pd.Series(index=df.index)).iloc[i]
        if pd.isna(f_prev) or pd.isna(s_prev) or pd.isna(f) or pd.isna(s):
            return EntryDecision(Side.FLAT)

        dead = float(f_prev) >= float(s_prev) and float(f) < float(s)
        if dead:
            return EntryDecision(Side.SHORT, reason="ENTRY_EMA20_40_DEAD", bypass_filter_pl=False)

        return EntryDecision(Side.FLAT)


class ManualEntryRule(EntryRule):
    """Manual entry rule for synthetic smoke tests.

    This rule reads a column `manual_entry` from the dataframe at day T close.

    Convention
    ----------
    - manual_entry > 0  -> LONG entry decision
    - manual_entry < 0  -> SHORT entry decision
    - else              -> FLAT

    Notes
    -----
    - Decisions are still executed at **T+1 open** by the engine.
    - We set `bypass_filter_pl=True` so the PL filter cannot suppress manual signals.
    """

    def __init__(self, col: str = "manual_entry"):
        super().__init__(EntryRuleType.MANUAL)
        self.col = str(col)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> EntryDecision:
        if self.col not in df.columns:
            return EntryDecision(Side.FLAT)

        v = df[self.col].iloc[i]
        if pd.isna(v):
            return EntryDecision(Side.FLAT)

        # accept ints/floats and some strings
        side = None
        if isinstance(v, (int, float)):
            if float(v) > 0:
                side = Side.LONG
            elif float(v) < 0:
                side = Side.SHORT
        else:
            s = str(v).strip().upper()
            if s in ("1", "+1", "LONG", "L"):
                side = Side.LONG
            elif s in ("-1", "SHORT", "S"):
                side = Side.SHORT

        if side == Side.LONG:
            return EntryDecision(Side.LONG, reason="ENTRY_MANUAL_LONG", bypass_filter_pl=True)
        if side == Side.SHORT:
            return EntryDecision(Side.SHORT, reason="ENTRY_MANUAL_SHORT", bypass_filter_pl=True)
        return EntryDecision(Side.FLAT)



def build_entry_rule(rule_type: EntryRuleType) -> EntryRule:
    if rule_type == EntryRuleType.A_TURTLE:
        return TurtleComboRule(donchian_window_20=20, donchian_window_55=55)
    if rule_type == EntryRuleType.A_20_PL:
        # legacy: same as Donchian20; PL filter is now a global filter option
        return TurtleRule(rule_type, window=20, bypass_pl=False)
    if rule_type == EntryRuleType.A_55:
        return TurtleRule(rule_type, window=55, bypass_pl=True)
    if rule_type == EntryRuleType.B_EMA_CROSS_DC10:
        # Strategy B: base Donchian(10) entry, with Donchian(20) override that bypasses Filter A.
        return EmaCrossDonchianRule(ema_fast="ema5", ema_mid="ema20", donchian_window=10, donchian_override_window=20)

    if rule_type == EntryRuleType.SHORT_EMA20_40_DEAD:
        return ShortEma2040DeadCrossRule(ema_fast="ema20", ema_slow="ema40")

    if rule_type == EntryRuleType.MANUAL:
        return ManualEntryRule(col="manual_entry")

    raise ValueError(f"Unknown entry rule type: {rule_type}")
