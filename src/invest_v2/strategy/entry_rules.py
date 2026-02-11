from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        # Evaluate signal at index i (close-based). Entry executes at next day's open.
        raise NotImplementedError


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


class RegressionRule(EntryRule):
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


def build_entry_rule(rule_type: EntryRuleType) -> EntryRule:
    if rule_type == EntryRuleType.A_20_PL:
        return TurtleRule(rule_type, window=20, pl_filter=True)
    if rule_type == EntryRuleType.A_55:
        return TurtleRule(rule_type, window=55, pl_filter=False)
    if rule_type == EntryRuleType.B_SLOPE20:
        return RegressionRule()
    raise ValueError(f"Unknown entry rule type: {rule_type}")
