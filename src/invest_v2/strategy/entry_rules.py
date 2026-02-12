from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

from invest_v2.core.types import EntryRuleType, Side


@dataclass
class EntryContext:
    last_trade_side: Optional[Side] = None
    last_trade_pnl: Optional[float] = None


class EntryRule:
    def __init__(self, rule_type: EntryRuleType):
        self.rule_type = rule_type

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        """Evaluate entry signal at index i (close-based). Entry executes at next day's open."""
        raise NotImplementedError

    def should_exit(self, df: pd.DataFrame, i: int, ctx: EntryContext, pos_side: Side) -> bool:
        """Optional regime-based exit (close-based at index i, exit at next day's open).

        Default is disabled (stop-loss / trailing-stop manages exits).
        """
        return False


# ============================================================
# Strategy A (Turtle): Donchian breakout with PL filter + 55 override
# ============================================================


class TurtleRule(EntryRule):
    def __init__(self, rule_type: EntryRuleType, window: int, pl_filter: bool):
        super().__init__(rule_type)
        self.window = int(window)
        self.pl_filter = bool(pl_filter)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        if i <= 0:
            return Side.FLAT

        c = float(df["close"].iloc[i])
        dh = df[f"donchian_high_{self.window}"].iloc[i]
        dl = df[f"donchian_low_{self.window}"].iloc[i]
        if pd.isna(dh) or pd.isna(dl):
            return Side.FLAT

        sig = Side.FLAT
        if c >= float(dh):
            sig = Side.LONG
        elif c <= float(dl):
            sig = Side.SHORT

        if sig == Side.FLAT:
            return Side.FLAT

        # PL filter: if last trade pnl > 0, ignore opposite-direction signal.
        if self.pl_filter and ctx.last_trade_pnl is not None and ctx.last_trade_side is not None:
            if ctx.last_trade_pnl > 0 and sig != ctx.last_trade_side:
                return Side.FLAT

        return sig


class TurtleComboRule(EntryRule):
    """Strategy A: A.1(20) + A.2(PL filter) + A.3(55 override).

    Evaluation order (close-based at day T, fill at T+1 open):
      1) A.3 Donchian(55) breakout -> ALWAYS take (ignores PL filter)
      2) Else A.1 Donchian(20) breakout -> apply PL filter (A.2)

    If both 20 and 55 trigger in the same direction, 55 still wins but is equivalent.
    If 20 triggers one way and 55 triggers the other, 55 wins (stronger breakout).
    """

    def __init__(self, donchian_window_20: int = 20, donchian_window_55: int = 55):
        super().__init__(EntryRuleType.A_TURTLE)
        self.w20 = int(donchian_window_20)
        self.w55 = int(donchian_window_55)

    def _donchian_signal(self, df: pd.DataFrame, i: int, window: int) -> Side:
        c = float(df["close"].iloc[i])
        dh = df[f"donchian_high_{window}"].iloc[i]
        dl = df[f"donchian_low_{window}"].iloc[i]
        if pd.isna(dh) or pd.isna(dl):
            return Side.FLAT
        if c >= float(dh):
            return Side.LONG
        if c <= float(dl):
            return Side.SHORT
        return Side.FLAT

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        if i <= 0:
            return Side.FLAT

        # 1) A.3: Donchian(55) always executes
        sig55 = self._donchian_signal(df, i, self.w55)
        if sig55 != Side.FLAT:
            return sig55

        # 2) A.1: Donchian(20) with PL filter (A.2)
        sig20 = self._donchian_signal(df, i, self.w20)
        if sig20 == Side.FLAT:
            return Side.FLAT

        if ctx.last_trade_pnl is not None and ctx.last_trade_side is not None:
            if ctx.last_trade_pnl > 0 and sig20 != ctx.last_trade_side:
                return Side.FLAT

        return sig20


# ============================================================
# Strategy B (Regression): slope(20) + r^2 + MA60
# ============================================================


class RegressionRule(EntryRule):
    """Legacy strict Type B.

    Condition (same day):
      - norm_slope_20 0-cross
      - r2_20 > 0.6
      - MA60 direction filter

    Observation: on 005930 the cross days typically have very low r2,
    resulting in near-zero signals. BA/BB were introduced to relax this.
    """

    def __init__(self):
        super().__init__(EntryRuleType.B_SLOPE20)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        if i <= 1:
            return Side.FLAT

        ns_prev = df["norm_slope_20"].iloc[i - 1]
        ns = df["norm_slope_20"].iloc[i]
        r2 = df["r2_20"].iloc[i]
        ma60 = df["ma60"].iloc[i]
        c = float(df["close"].iloc[i])

        if pd.isna(ns_prev) or pd.isna(ns) or pd.isna(r2) or pd.isna(ma60):
            return Side.FLAT

        if float(r2) <= 0.6:
            return Side.FLAT

        # 60MA directional filter
        long_ok = c >= float(ma60)
        short_ok = c <= float(ma60)

        # 0-line crossing
        if float(ns_prev) <= 0.0 and float(ns) > 0.0 and long_ok:
            return Side.LONG
        if float(ns_prev) >= 0.0 and float(ns) < 0.0 and short_ok:
            return Side.SHORT

        return Side.FLAT


class BARegimeRule(EntryRule):
    """Type B variant BA: 'Regime entry' instead of 0-cross.

    Regime conditions:
      LONG  when (norm_slope_20 > 0) & (r2_20 > 0.6) & (close >= MA60)
      SHORT when (norm_slope_20 < 0) & (r2_20 > 0.6) & (close <= MA60)

    Entry occurs only when the regime turns ON (False -> True).
    """

    def __init__(self, *, r2_thr: float = 0.6):
        super().__init__(EntryRuleType.BA_REGIME20)
        self.r2_thr = float(r2_thr)

    def _cond_long(self, df: pd.DataFrame, i: int) -> Optional[bool]:
        if i < 0:
            return None
        ns = df["norm_slope_20"].iloc[i]
        r2 = df["r2_20"].iloc[i]
        ma60 = df["ma60"].iloc[i]
        c = df["close"].iloc[i]
        if pd.isna(ns) or pd.isna(r2) or pd.isna(ma60) or pd.isna(c):
            return None
        return (float(ns) > 0.0) and (float(r2) > self.r2_thr) and (float(c) >= float(ma60))

    def _cond_short(self, df: pd.DataFrame, i: int) -> Optional[bool]:
        if i < 0:
            return None
        ns = df["norm_slope_20"].iloc[i]
        r2 = df["r2_20"].iloc[i]
        ma60 = df["ma60"].iloc[i]
        c = df["close"].iloc[i]
        if pd.isna(ns) or pd.isna(r2) or pd.isna(ma60) or pd.isna(c):
            return None
        return (float(ns) < 0.0) and (float(r2) > self.r2_thr) and (float(c) <= float(ma60))

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        if i <= 0:
            return Side.FLAT

        cl = self._cond_long(df, i)
        cl_prev = self._cond_long(df, i - 1)
        cs = self._cond_short(df, i)
        cs_prev = self._cond_short(df, i - 1)

        if cl is not None and cl_prev is not None:
            if bool(cl) and (not bool(cl_prev)):
                return Side.LONG

        if cs is not None and cs_prev is not None:
            if bool(cs) and (not bool(cs_prev)):
                return Side.SHORT

        return Side.FLAT


class BBCrossR2PrevRule(EntryRule):
    """Type B variant BB: keep 0-cross, but evaluate r2 at t-1.

    Motivation: r2 is structurally low on the turning-point day.

    Condition:
      - r2_20(t-1) > 0.6
      - MA60 direction filter (at day t)
      - norm_slope_20 crosses 0 at day t
    """

    def __init__(self, *, r2_thr: float = 0.6):
        super().__init__(EntryRuleType.BB_CROSS20_R2PREV)
        self.r2_thr = float(r2_thr)

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        if i <= 1:
            return Side.FLAT

        ns_prev = df["norm_slope_20"].iloc[i - 1]
        ns = df["norm_slope_20"].iloc[i]
        r2_prev = df["r2_20"].iloc[i - 1]
        ma60 = df["ma60"].iloc[i]
        c = float(df["close"].iloc[i])

        if pd.isna(ns_prev) or pd.isna(ns) or pd.isna(r2_prev) or pd.isna(ma60):
            return Side.FLAT

        if float(r2_prev) <= self.r2_thr:
            return Side.FLAT

        long_ok = c >= float(ma60)
        short_ok = c <= float(ma60)

        if float(ns_prev) <= 0.0 and float(ns) > 0.0 and long_ok:
            return Side.LONG
        if float(ns_prev) >= 0.0 and float(ns) < 0.0 and short_ok:
            return Side.SHORT

        return Side.FLAT


# ============================================================
# Strategy C: Time-series momentum + MA-cycle phase filter
# ============================================================


class TSMomCycleRule(EntryRule):
    """Strategy C: Time-series momentum + Moving Average Cycle (6-phase) regime filter.

    - Momentum: mom_L = Close_t / Close_{t-L} - 1
    - Regime filter: allow LONG only when both:
        (a) stock phase ∈ {6, 1, 2}
        (b) market phase ∈ {6, 1, 2}  (KOSPI200 futures index, assumed provided via CSV)

    Entry style: 'regime entry' (take only the first day the full condition turns True).
    Exit style: optional regime exit (if condition turns False while holding LONG).
    """

    def __init__(
        self,
        *,
        mom_window: int = 63,
        enter_thr: float = 0.05,
        exit_thr: float = 0.0,
        cycle_windows: Tuple[int, int, int] = (5, 20, 40),
        allowed_phases_long: Sequence[int] = (6, 1, 2),
        use_market_filter: bool = True,
        market_prefix: str = "mkt",
    ):
        super().__init__(EntryRuleType.C_TSMOM_CYCLE)
        self.mom_window = int(mom_window)
        self.enter_thr = float(enter_thr)
        self.exit_thr = float(exit_thr)
        s, m, l = (int(cycle_windows[0]), int(cycle_windows[1]), int(cycle_windows[2]))
        self.cycle_windows = (s, m, l)
        self.phase_col = f"cycle_phase_{s}_{m}_{l}"
        self.mom_col = f"mom_{self.mom_window}"
        self.use_market_filter = bool(use_market_filter)
        self.market_prefix = str(market_prefix)
        self.mkt_phase_col = f"{self.market_prefix}_cycle_phase_{s}_{m}_{l}"
        self.allowed_phases_long = tuple(int(x) for x in allowed_phases_long)

    def _phase_ok(self, phase_val: Any) -> bool:
        if pd.isna(phase_val):
            return False
        try:
            p = int(float(phase_val))
        except Exception:
            return False
        return p in self.allowed_phases_long

    def _entry_condition(self, df: pd.DataFrame, i: int) -> Optional[bool]:
        # Required columns
        if self.mom_col not in df.columns or self.phase_col not in df.columns:
            return None
        if self.use_market_filter and self.mkt_phase_col not in df.columns:
            return None

        mom = df[self.mom_col].iloc[i]
        phase = df[self.phase_col].iloc[i]
        if pd.isna(mom) or pd.isna(phase):
            return None

        if float(mom) <= self.enter_thr:
            return False

        if not self._phase_ok(phase):
            return False

        if self.use_market_filter:
            mkt_phase = df[self.mkt_phase_col].iloc[i]
            if not self._phase_ok(mkt_phase):
                return False

        return True

    def evaluate(self, df: pd.DataFrame, i: int, ctx: EntryContext) -> Side:
        if i <= 0:
            return Side.FLAT

        cond = self._entry_condition(df, i)
        cond_prev = self._entry_condition(df, i - 1)

        if cond is None or cond_prev is None:
            return Side.FLAT

        if bool(cond) and (not bool(cond_prev)):
            return Side.LONG

        return Side.FLAT

    def should_exit(self, df: pd.DataFrame, i: int, ctx: EntryContext, pos_side: Side) -> bool:
        if pos_side != Side.LONG:
            return False

        # If we can't evaluate, be conservative and exit.
        if self.mom_col not in df.columns or self.phase_col not in df.columns:
            return True
        if self.use_market_filter and self.mkt_phase_col not in df.columns:
            return True

        mom = df[self.mom_col].iloc[i]
        phase = df[self.phase_col].iloc[i]
        if pd.isna(mom) or pd.isna(phase):
            return True

        hold_ok = True

        # Momentum must stay positive (or above exit_thr).
        if float(mom) <= self.exit_thr:
            hold_ok = False

        # Stock phase must remain in allowed set.
        if not self._phase_ok(phase):
            hold_ok = False

        # Market phase must remain in allowed set.
        if self.use_market_filter:
            mkt_phase = df[self.mkt_phase_col].iloc[i]
            if not self._phase_ok(mkt_phase):
                hold_ok = False

        return not hold_ok


# ============================================================
# Factory
# ============================================================


def build_entry_rule(rule_type: EntryRuleType, params: Optional[Mapping[str, Any]] = None) -> EntryRule:
    params = params or {}

    if rule_type == EntryRuleType.A_TURTLE:
        return TurtleComboRule(donchian_window_20=20, donchian_window_55=55)
    if rule_type == EntryRuleType.A_20_PL:
        return TurtleRule(rule_type, window=20, pl_filter=True)
    if rule_type == EntryRuleType.A_55:
        return TurtleRule(rule_type, window=55, pl_filter=False)

    if rule_type == EntryRuleType.B_SLOPE20:
        return RegressionRule()
    if rule_type == EntryRuleType.BA_REGIME20:
        return BARegimeRule(r2_thr=float(params.get("r2_thr", 0.6)))
    if rule_type == EntryRuleType.BB_CROSS20_R2PREV:
        return BBCrossR2PrevRule(r2_thr=float(params.get("r2_thr", 0.6)))

    if rule_type == EntryRuleType.C_TSMOM_CYCLE:
        cycle_windows = params.get("cycle_windows", (5, 20, 40))
        allowed_phases_long = params.get("allowed_phases_long", (6, 1, 2))
        return TSMomCycleRule(
            mom_window=int(params.get("mom_window", 63)),
            enter_thr=float(params.get("enter_thr", 0.05)),
            exit_thr=float(params.get("exit_thr", 0.0)),
            cycle_windows=(int(cycle_windows[0]), int(cycle_windows[1]), int(cycle_windows[2])),
            allowed_phases_long=tuple(int(x) for x in allowed_phases_long),
            use_market_filter=bool(params.get("use_market_filter", True)),
            market_prefix=str(params.get("market_prefix", "mkt")),
        )

    raise ValueError(f"Unknown entry rule type: {rule_type}")
