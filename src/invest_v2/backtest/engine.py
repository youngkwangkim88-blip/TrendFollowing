from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from invest_v2.core.types import (
    EntryRuleType,
    Side,
    Order,
    Trade,
    TrailingStopType,
    PyramidingType,
)
from invest_v2.strategy.entry_rules import build_entry_rule, EntryContext
from invest_v2.utils.krx_ticks import tick_down, tick_up
from invest_v2.backtest.accounting import Account
from invest_v2.indicators.donchian import donchian_high, donchian_low


@dataclass
class BacktestConfig:
    symbol: str = "005930"
    initial_capital: float = 700_000_000.0
    one_trading_risk: float = 0.01
    max_units_per_symbol: int = 4
    max_units_total: int = 10
    short_notional_limit: float = 570_000_000.0  # backtest assumption

    # Entry
    entry_rule: EntryRuleType = EntryRuleType.A_TURTLE
    enable_short: bool = True

    # Strategy C (TS momentum + MA-cycle filter) knobs
    # NOTE: used only when entry_rule == C_TSMOM_CYCLE
    c_mom_window: int = 63
    c_enter_thr: float = 0.05
    c_exit_thr: float = 0.0
    c_cycle_ema_short: int = 5
    c_cycle_ema_mid: int = 20
    c_cycle_ema_long: int = 40
    c_allowed_phases_long: Tuple[int, ...] = (6, 1, 2)
    c_use_market_filter: bool = True
    c_market_prefix: str = "mkt"

    # Stop-loss (Initial Stop)
    stop_atr_mult: float = 2.0

    # Trailing stop selection
    ts_type: TrailingStopType = TrailingStopType.TS_A

    # TS.A (percentage)
    ts_activate_gain: float = 0.20
    ts_floor_gain: float = 0.10
    ts_trail_frac: float = 0.10

    # TS.B (EMA cross)
    ts_b_fast: int = 5
    ts_b_slow: int = 20

    # TS.C (Darvas/box)
    ts_c_box_window: int = 20
    ts_c_monotonic: bool = True

    # Pyramiding
    enable_pyramiding: bool = True
    pyramiding_type: PyramidingType = PyramidingType.PRMD_A

    # PRMD.A (percentage)
    pyramid_trigger: float = 0.15

    # PRMD.B (Darvas/box)
    prmd_b_box_window: int = 20
    prmd_b_cooldown_days: int = 5

    # Costs
    sell_cost_rate: float = 0.003

    # Short constraints
    annual_short_interest_rate: float = 0.045
    short_max_hold_days: int = 90


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    summary: Dict[str, Any]


class SingleSymbolBacktester:
    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        self.df = df.copy()
        self.cfg = cfg

        # Ensure some indicators exist if the caller didn't run `add_indicators`.
        self._ensure_required_columns()

        self.entry_rule = build_entry_rule(
            cfg.entry_rule,
            params={
                "mom_window": cfg.c_mom_window,
                "enter_thr": cfg.c_enter_thr,
                "exit_thr": cfg.c_exit_thr,
                "cycle_windows": (cfg.c_cycle_ema_short, cfg.c_cycle_ema_mid, cfg.c_cycle_ema_long),
                "allowed_phases_long": cfg.c_allowed_phases_long,
                "use_market_filter": cfg.c_use_market_filter,
                "market_prefix": cfg.c_market_prefix,
            },
        )
        self.ctx = EntryContext()

        self.account = Account(cash_cma=float(cfg.initial_capital))

        # Position state
        self.pos_side: Side = Side.FLAT
        self.pos_shares: int = 0
        self.pos_units: int = 0
        self.pos_avg_price: float = 0.0  # X
        self.pos_atr_ref: float = 0.0
        self.pos_h_max: float = 0.0
        self.pos_l_min: float = 0.0
        self.pos_ts_active: bool = False  # TS.A activation flag

        # TS.C (box) state
        self.ts_c_level: float = 0.0

        # Pyramiding cooldown (PRMD.B)
        self.last_pyramid_idx: Optional[int] = None

        # Short management state
        self.short_hold_days: int = 0
        self.short_basis_price: float = 0.0  # weighted avg of short sell prices
        self.short_open_date: Optional[str] = None

        self.pending_order: Optional[Order] = None
        self.pending_exit: Optional[Dict[str, Any]] = None  # next-open exit (regime or TS.B)
        self.active_trade: Optional[Trade] = None

        self.snapshots: List[Dict[str, Any]] = []
        self.trade_logs: List[Dict[str, Any]] = []

    # -------------------------
    # Data / indicator hygiene
    # -------------------------

    def _ensure_required_columns(self) -> None:
        """Make the backtester more robust when called with partially-prepared data."""
        df = self.df

        # Basic OHLC sanity
        for c in ["open", "high", "low", "close"]:
            if c not in df.columns:
                raise ValueError(f"Input df missing required OHLC column: {c}")

        # Ensure numeric
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # If there are non-positive prices, they will break the backtest.
        # We don't auto-fix here to avoid silent data corruption; loaders should sanitize.
        bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if bool(bad.any()):
            # Keep the error actionable.
            first_bad = df.index[bad].min()
            raise ValueError(
                "Found non-positive OHLC values. Please sanitize the data (0/negative treated as missing). "
                f"First bad date: {first_bad}"
            )

        # Ensure EMA columns for TS.B if requested
        if self.cfg.ts_type == TrailingStopType.TS_B:
            for w in (int(self.cfg.ts_b_fast), int(self.cfg.ts_b_slow)):
                col = f"ema_{w}"
                if col not in df.columns:
                    df[col] = df["close"].astype(float).ewm(span=w, adjust=False).mean()

        # Ensure Donchian columns for TS.C / PRMD.B if requested with non-default windows
        if self.cfg.ts_type == TrailingStopType.TS_C:
            w = int(self.cfg.ts_c_box_window)
            if f"donchian_high_{w}" not in df.columns:
                df[f"donchian_high_{w}"] = donchian_high(df, window=w)
            if f"donchian_low_{w}" not in df.columns:
                df[f"donchian_low_{w}"] = donchian_low(df, window=w)

        if self.cfg.enable_pyramiding and self.cfg.pyramiding_type == PyramidingType.PRMD_B:
            w = int(self.cfg.prmd_b_box_window)
            if f"donchian_high_{w}" not in df.columns:
                df[f"donchian_high_{w}"] = donchian_high(df, window=w)
            if f"donchian_low_{w}" not in df.columns:
                df[f"donchian_low_{w}"] = donchian_low(df, window=w)

        self.df = df

    # -------------------------
    # Valuation helpers
    # -------------------------

    def _holding_value(self, close_price: float) -> float:
        return float(self.pos_shares) * float(close_price) if self.pos_side == Side.LONG else 0.0

    def _short_liability(self, close_price: float) -> float:
        return float(self.pos_shares) * float(close_price) if self.pos_side == Side.SHORT else 0.0

    def _short_notional_basis(self) -> float:
        if self.pos_side != Side.SHORT or self.pos_shares <= 0:
            return 0.0
        return float(self.short_basis_price) * float(self.pos_shares)

    def _compute_unit_shares(self, M: float, atr10: float) -> int:
        if atr10 <= 0:
            return 0
        return int((self.cfg.one_trading_risk * M) // float(atr10))

    # -------------------------
    # Stop / TS levels
    # -------------------------

    def _update_trailing_extrema(self, high: float, low: float) -> None:
        if self.pos_side == Side.LONG:
            self.pos_h_max = max(self.pos_h_max, float(high))
        elif self.pos_side == Side.SHORT:
            if self.pos_l_min == 0.0:
                self.pos_l_min = float(low)
            else:
                self.pos_l_min = min(self.pos_l_min, float(low))

    def _update_ts_c_level(self, i: int) -> None:
        """Update TS.C (Darvas/box) support/resistance level using Donchian bands."""
        if self.cfg.ts_type != TrailingStopType.TS_C or self.pos_side == Side.FLAT:
            return

        w = int(self.cfg.ts_c_box_window)
        if self.pos_side == Side.LONG:
            col = f"donchian_low_{w}"
            val = self.df[col].iloc[i] if col in self.df.columns else pd.NA
            if pd.isna(val):
                return
            v = float(val)
            if self.ts_c_level <= 0:
                self.ts_c_level = v
            else:
                self.ts_c_level = max(self.ts_c_level, v) if bool(self.cfg.ts_c_monotonic) else v

        elif self.pos_side == Side.SHORT:
            col = f"donchian_high_{w}"
            val = self.df[col].iloc[i] if col in self.df.columns else pd.NA
            if pd.isna(val):
                return
            v = float(val)
            if self.ts_c_level <= 0:
                self.ts_c_level = v
            else:
                self.ts_c_level = min(self.ts_c_level, v) if bool(self.cfg.ts_c_monotonic) else v

    def _stop_loss_level(self) -> Optional[float]:
        if self.pos_side == Side.FLAT:
            return None
        X = float(self.pos_avg_price)
        atr = float(self.pos_atr_ref)
        stop_mult = float(self.cfg.stop_atr_mult)

        if self.pos_side == Side.LONG:
            return tick_down(X - stop_mult * atr)
        return tick_up(X + stop_mult * atr)

    def _ts_a_level(self) -> Optional[float]:
        """TS.A: percentage-based trailing stop level."""
        if self.cfg.ts_type != TrailingStopType.TS_A or self.pos_side == Side.FLAT:
            return None

        X = float(self.pos_avg_price)
        ts_act = float(self.cfg.ts_activate_gain)
        ts_floor = float(self.cfg.ts_floor_gain)
        ts_trail = float(self.cfg.ts_trail_frac)

        if self.pos_side == Side.LONG:
            if self.pos_h_max >= (1.0 + ts_act) * X:
                self.pos_ts_active = True
            if not self.pos_ts_active:
                return None
            ts_raw = max((1.0 + ts_floor) * X, (1.0 - ts_trail) * float(self.pos_h_max))
            return tick_down(ts_raw)

        # SHORT
        if self.pos_l_min > 0 and self.pos_l_min <= (1.0 - ts_act) * X:
            self.pos_ts_active = True
        if not self.pos_ts_active:
            return None
        ts_raw = min((1.0 - ts_floor) * X, (1.0 + ts_trail) * float(self.pos_l_min))
        return tick_up(ts_raw)

    def _ts_c_level_tick(self) -> Optional[float]:
        """TS.C: darvas/box trailing stop level."""
        if self.cfg.ts_type != TrailingStopType.TS_C or self.pos_side == Side.FLAT:
            return None
        if self.ts_c_level <= 0:
            return None
        return tick_down(self.ts_c_level) if self.pos_side == Side.LONG else tick_up(self.ts_c_level)

    def _effective_stop(self) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        """Return (stop_loss, ts_level, effective, effective_type).

        effective_type in {"STOP_LOSS", "TS_A", "TS_C"}
        """
        stop_loss = self._stop_loss_level()
        if stop_loss is None:
            return None, None, None, "STOP_LOSS"

        ts_level = None
        if self.cfg.ts_type == TrailingStopType.TS_A:
            ts_level = self._ts_a_level()
        elif self.cfg.ts_type == TrailingStopType.TS_C:
            ts_level = self._ts_c_level_tick()

        eff = float(stop_loss)
        eff_type = "STOP_LOSS"

        if ts_level is not None:
            tl = float(ts_level)
            if self.pos_side == Side.LONG:
                if tl > eff:
                    eff = tl
                    eff_type = "TS_A" if self.cfg.ts_type == TrailingStopType.TS_A else "TS_C"
            else:
                if tl < eff:
                    eff = tl
                    eff_type = "TS_A" if self.cfg.ts_type == TrailingStopType.TS_A else "TS_C"

        return float(stop_loss), (float(ts_level) if ts_level is not None else None), float(eff), eff_type

    def _check_stop_trigger_and_exit(self, o: float, h: float, l: float) -> Optional[Dict[str, Any]]:
        """Intraday stop check for stop-loss and TS.A/TS.C."""
        _, _, eff, eff_type = self._effective_stop()
        if eff is None:
            return None
        eff = float(eff)

        if self.pos_side == Side.LONG:
            if float(o) <= eff:
                return {"exit_price": float(o), "exit_reason": f"{eff_type}_GAP"}
            if float(o) > eff and float(l) <= eff:
                return {"exit_price": eff, "exit_reason": f"{eff_type}_TOUCH"}
            return None

        # SHORT
        if float(o) >= eff:
            return {"exit_price": float(o), "exit_reason": f"{eff_type}_GAP"}
        if float(o) < eff and float(h) >= eff:
            return {"exit_price": eff, "exit_reason": f"{eff_type}_TOUCH"}
        return None

    def _ts_b_cross_exit_reason(self, i: int) -> Optional[str]:
        """TS.B: EMA cross exit (close-based)."""
        if self.cfg.ts_type != TrailingStopType.TS_B or self.pos_side == Side.FLAT:
            return None
        if i <= 0:
            return None

        f = int(self.cfg.ts_b_fast)
        s = int(self.cfg.ts_b_slow)
        ef_col = f"ema_{f}"
        es_col = f"ema_{s}"
        if ef_col not in self.df.columns or es_col not in self.df.columns:
            return None

        ef_prev = self.df[ef_col].iloc[i - 1]
        ef = self.df[ef_col].iloc[i]
        es_prev = self.df[es_col].iloc[i - 1]
        es = self.df[es_col].iloc[i]
        if pd.isna(ef_prev) or pd.isna(ef) or pd.isna(es_prev) or pd.isna(es):
            return None

        if self.pos_side == Side.LONG:
            if float(ef_prev) >= float(es_prev) and float(ef) < float(es):
                return "TS_B_DEAD_CROSS"
        elif self.pos_side == Side.SHORT:
            if float(ef_prev) <= float(es_prev) and float(ef) > float(es):
                return "TS_B_GOLDEN_CROSS"

        return None

    # -------------------------
    # Execution / state reset
    # -------------------------

    def _exit_position_full(self, date: str, exit_price: float, reason: str) -> None:
        shares = int(self.pos_shares)
        if shares <= 0 or self.active_trade is None:
            return

        if self.pos_side == Side.LONG:
            entry_cost = self.active_trade.entry_price * shares
            exit_net = exit_price * shares * (1.0 - self.cfg.sell_cost_rate)
            realized = exit_net - entry_cost
            self.account.long_sell(exit_price, shares, sell_cost_rate=self.cfg.sell_cost_rate)
        else:
            entry_net = self.active_trade.entry_price * shares * (1.0 - self.cfg.sell_cost_rate)
            cover_cost = exit_price * shares
            realized = entry_net - cover_cost
            self.account.short_cover(exit_price, shares)

        self.active_trade.exit_date = date
        self.active_trade.exit_price = float(exit_price)
        self.active_trade.realized_pnl = float(realized)
        self.active_trade.exit_reason = reason

        self.trade_logs.append(
            {
                "symbol": self.active_trade.symbol,
                "side": int(self.active_trade.side.value),
                "entry_date": self.active_trade.entry_date,
                "entry_price": self.active_trade.entry_price,
                "shares": self.active_trade.shares,
                "exit_date": self.active_trade.exit_date,
                "exit_price": self.active_trade.exit_price,
                "realized_pnl": self.active_trade.realized_pnl,
                "exit_reason": self.active_trade.exit_reason,
            }
        )

        self.ctx.last_trade_side = self.active_trade.side
        self.ctx.last_trade_pnl = float(realized)

        if self.pos_side == Side.SHORT:
            self.account.release_locked()

        # reset position
        self.pos_side = Side.FLAT
        self.pos_shares = 0
        self.pos_units = 0
        self.pos_avg_price = 0.0
        self.pos_atr_ref = 0.0
        self.pos_h_max = 0.0
        self.pos_l_min = 0.0
        self.pos_ts_active = False
        self.ts_c_level = 0.0
        self.last_pyramid_idx = None

        self.short_hold_days = 0
        self.short_basis_price = 0.0
        self.short_open_date = None
        self.active_trade = None

    def _can_add_unit(self) -> bool:
        return self.pos_units < self.cfg.max_units_per_symbol and self.pos_units < self.cfg.max_units_total

    def _can_open_short_with_limit(self, add_shares: int, sell_price: float) -> bool:
        new_basis_total = self._short_notional_basis() + float(sell_price) * int(add_shares)
        return new_basis_total <= self.cfg.short_notional_limit

    # -------------------------
    # Main loop
    # -------------------------

    def run(self) -> BacktestResult:
        df = self.df

        for i in range(len(df)):
            ts = df.index[i]
            date = str(ts.date())
            row = df.iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # first trading day of month?
            is_first_trading_day = True if i == 0 else (df.index[i - 1].month != ts.month)
            self.account.apply_monthly_interest_if_first_trading_day(is_first_trading_day_of_month=is_first_trading_day)

            # 1) execute pending next-open exit (regime exit / TS.B)
            if self.pending_exit is not None and self.pos_side != Side.FLAT:
                self.account.ensure_trading_cash()
                reason = str(self.pending_exit.get("reason", "EXIT"))
                self.pending_exit = None
                # Exit has priority over any pending entry/pyramiding order.
                self.pending_order = None
                self._exit_position_full(date=date, exit_price=o, reason=reason)

            # 2) execute pending order at open
            if self.pending_order is not None:
                self.account.ensure_trading_cash()
                order = self.pending_order
                self.pending_order = None

                fill = float(o)
                add_shares = int(order.unit_shares)
                if add_shares > 0:
                    if order.side == Side.LONG:
                        self.account.long_buy(fill, add_shares)
                        if self.pos_side == Side.FLAT:
                            # open long
                            self.pos_side = Side.LONG
                            self.pos_units = 1
                            self.pos_shares = add_shares
                            self.pos_avg_price = fill
                            self.pos_atr_ref = float(order.atr_ref)
                            self.pos_h_max = max(h, fill)
                            self.pos_ts_active = False
                            self.ts_c_level = 0.0
                            self.last_pyramid_idx = None
                            self.active_trade = Trade(
                                symbol=self.cfg.symbol,
                                side=Side.LONG,
                                entry_date=date,
                                entry_price=fill,
                                shares=add_shares,
                            )
                        else:
                            # pyramiding long
                            new_total = self.pos_shares + add_shares
                            self.pos_avg_price = (self.pos_avg_price * self.pos_shares + fill * add_shares) / new_total
                            self.pos_shares = new_total
                            self.pos_units += 1
                            self.pos_atr_ref = float(order.atr_ref)
                            self.pos_h_max = max(h, fill)
                            self.pos_ts_active = False
                            self.ts_c_level = 0.0
                            self.last_pyramid_idx = i
                            if self.active_trade:
                                self.active_trade.shares = self.pos_shares
                    else:
                        # short sell: notional gate
                        if self._can_open_short_with_limit(add_shares, sell_price=fill):
                            self.account.short_sell(fill, add_shares, sell_cost_rate=self.cfg.sell_cost_rate)
                            if self.pos_side == Side.FLAT:
                                # open short
                                self.pos_side = Side.SHORT
                                self.pos_units = 1
                                self.pos_shares = add_shares
                                self.pos_avg_price = fill
                                self.pos_atr_ref = float(order.atr_ref)
                                self.pos_l_min = min(l, fill)
                                self.pos_ts_active = False
                                self.ts_c_level = 0.0
                                self.last_pyramid_idx = None
                                self.short_hold_days = 0
                                self.short_open_date = date
                                self.short_basis_price = fill
                                self.active_trade = Trade(
                                    symbol=self.cfg.symbol,
                                    side=Side.SHORT,
                                    entry_date=date,
                                    entry_price=fill,
                                    shares=add_shares,
                                )
                            else:
                                # pyramiding short
                                new_total = self.pos_shares + add_shares
                                self.pos_avg_price = (self.pos_avg_price * self.pos_shares + fill * add_shares) / new_total
                                self.pos_shares = new_total
                                self.pos_units += 1
                                self.pos_atr_ref = float(order.atr_ref)
                                self.pos_l_min = min(l, fill)
                                self.pos_ts_active = False
                                self.ts_c_level = 0.0
                                self.last_pyramid_idx = i
                                self.short_basis_price = (
                                    self.short_basis_price * (new_total - add_shares) + fill * add_shares
                                ) / new_total
                                if self.active_trade:
                                    self.active_trade.shares = self.pos_shares
                        else:
                            # skipped due to notional limit
                            pass

            # 3) short max hold days: exit at open
            if self.pos_side == Side.SHORT and self.short_hold_days >= self.cfg.short_max_hold_days:
                self._exit_position_full(date=date, exit_price=o, reason="SHORT_MAX_HOLD_DAYS")
            else:
                # TS.C updates based on last 20-day box (known at open)
                if self.pos_side != Side.FLAT and self.cfg.ts_type == TrailingStopType.TS_C:
                    self._update_ts_c_level(i)

                # update extrema then stop check (intraday monitoring assumption)
                if self.pos_side != Side.FLAT:
                    self._update_trailing_extrema(high=h, low=l)

                exit_info = self._check_stop_trigger_and_exit(o=o, h=h, l=l)
                if exit_info is not None:
                    self._exit_position_full(date=date, exit_price=float(exit_info["exit_price"]), reason=str(exit_info["exit_reason"]))

            # accrue interest EOD
            if self.pos_side == Side.SHORT:
                self.account.accrue_daily_short_interest(self._short_notional_basis(), annual_rate=self.cfg.annual_short_interest_rate)
                self.short_hold_days += 1

            # 4) generate next-day orders (close-based)
            if i < len(df) - 1:
                atr10_today = df.get("atr10", pd.Series(index=df.index)).iloc[i] if "atr10" in df.columns else 0.0
                atr_ref = float(atr10_today) if not pd.isna(atr10_today) else 0.0

                # 4.1 next-open exits (regime exit / TS.B)
                if self.pos_side != Side.FLAT and self.pending_exit is None:
                    # Strategy C regime-based exit
                    if self.pos_side == Side.LONG and self.entry_rule.should_exit(df, i, self.ctx, self.pos_side):
                        self.pending_exit = {
                            "reason": f"REGIME_EXIT_{self.cfg.entry_rule.value}",
                            "signal_date": date,
                        }

                    # TS.B EMA cross exit
                    if self.pending_exit is None:
                        tsb_reason = self._ts_b_cross_exit_reason(i)
                        if tsb_reason is not None:
                            self.pending_exit = {
                                "reason": tsb_reason,
                                "signal_date": date,
                            }

                # 4.2 entries
                if self.pos_side == Side.FLAT and self.pending_order is None and self.pending_exit is None:
                    sig = self.entry_rule.evaluate(df, i, self.ctx)
                    if sig == Side.SHORT and not self.cfg.enable_short:
                        sig = Side.FLAT

                    if sig != Side.FLAT:
                        unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                        if unit_shares > 0:
                            if sig == Side.SHORT and not self._can_open_short_with_limit(unit_shares, sell_price=c):
                                pass
                            else:
                                self.pending_order = Order(
                                    symbol=self.cfg.symbol,
                                    side=sig,
                                    unit_shares=unit_shares,
                                    atr_ref=atr_ref,
                                    signal_date=date,
                                    reason=f"ENTRY_{self.cfg.entry_rule.value}",
                                )

                # 4.3 pyramiding
                if (
                    self.cfg.enable_pyramiding
                    and self.pos_side == Side.LONG
                    and self.pending_order is None
                    and self.pending_exit is None
                    and self._can_add_unit()
                ):
                    if self.cfg.pyramiding_type == PyramidingType.PRMD_A:
                        if c >= (1.0 + float(self.cfg.pyramid_trigger)) * float(self.pos_avg_price):
                            unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                            if unit_shares > 0:
                                self.pending_order = Order(
                                    symbol=self.cfg.symbol,
                                    side=Side.LONG,
                                    unit_shares=unit_shares,
                                    atr_ref=atr_ref,
                                    signal_date=date,
                                    reason="PYRAMID_LONG_PRMD_A",
                                )
                    else:
                        # PRMD.B: Donchian/Darvas breakout + cooldown
                        w = int(self.cfg.prmd_b_box_window)
                        col = f"donchian_high_{w}"
                        top = df[col].iloc[i] if col in df.columns else pd.NA
                        cooldown = int(self.cfg.prmd_b_cooldown_days)
                        ok_cd = True if self.last_pyramid_idx is None else (i - int(self.last_pyramid_idx) >= cooldown)
                        if ok_cd and (not pd.isna(top)) and float(c) >= float(top):
                            unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                            if unit_shares > 0:
                                self.pending_order = Order(
                                    symbol=self.cfg.symbol,
                                    side=Side.LONG,
                                    unit_shares=unit_shares,
                                    atr_ref=atr_ref,
                                    signal_date=date,
                                    reason="PYRAMID_LONG_PRMD_B",
                                )

                if (
                    self.cfg.enable_pyramiding
                    and self.pos_side == Side.SHORT
                    and self.pending_order is None
                    and self.pending_exit is None
                    and self._can_add_unit()
                ):
                    if not self.cfg.enable_short:
                        pass
                    elif self.cfg.pyramiding_type == PyramidingType.PRMD_A:
                        if c <= (1.0 - float(self.cfg.pyramid_trigger)) * float(self.pos_avg_price):
                            unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                            if unit_shares > 0 and self._can_open_short_with_limit(unit_shares, sell_price=c):
                                self.pending_order = Order(
                                    symbol=self.cfg.symbol,
                                    side=Side.SHORT,
                                    unit_shares=unit_shares,
                                    atr_ref=atr_ref,
                                    signal_date=date,
                                    reason="PYRAMID_SHORT_PRMD_A",
                                )
                    else:
                        w = int(self.cfg.prmd_b_box_window)
                        col = f"donchian_low_{w}"
                        bot = df[col].iloc[i] if col in df.columns else pd.NA
                        cooldown = int(self.cfg.prmd_b_cooldown_days)
                        ok_cd = True if self.last_pyramid_idx is None else (i - int(self.last_pyramid_idx) >= cooldown)
                        if ok_cd and (not pd.isna(bot)) and float(c) <= float(bot):
                            unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                            if unit_shares > 0 and self._can_open_short_with_limit(unit_shares, sell_price=c):
                                self.pending_order = Order(
                                    symbol=self.cfg.symbol,
                                    side=Side.SHORT,
                                    unit_shares=unit_shares,
                                    atr_ref=atr_ref,
                                    signal_date=date,
                                    reason="PYRAMID_SHORT_PRMD_B",
                                )

            # park to CMA if flat and no pending order
            if self.pos_side == Side.FLAT and self.pending_order is None:
                self.account.park_to_cma_if_flat()

            # snapshot EOD NAV (close-based)
            holding_value = self._holding_value(c)
            short_liab = self._short_liability(c)
            nav = self.account.nav(holding_value, short_liab)

            self.snapshots.append(
                {
                    "date": date,
                    "cash_cma": self.account.cash_cma,
                    "cash_free": self.account.cash_free,
                    "cash_locked": self.account.cash_locked,
                    "holding_value": holding_value,
                    "short_liability": short_liab,
                    "nav": nav,
                    "pos_side": int(self.pos_side.value),
                    "pos_units": int(self.pos_units),
                    "pos_shares": int(self.pos_shares),
                    "pos_avg_price": float(self.pos_avg_price),
                    "short_basis_price": float(self.short_basis_price),
                    "short_notional_basis": float(self._short_notional_basis()),
                    "accrued_interest": float(self.account.accrued_interest),
                }
            )

        equity_df = pd.DataFrame(self.snapshots).set_index("date")
        trades_df = pd.DataFrame(self.trade_logs)
        summary = self._summarize(equity_df, trades_df)
        return BacktestResult(equity_curve=equity_df, trades=trades_df, summary=summary)

    # -------------------------
    # Summary
    # -------------------------

    def _summarize(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        nav = equity_df["nav"].astype(float)

        start_nav = float(nav.iloc[0])
        end_nav = float(nav.iloc[-1])
        start_date = pd.to_datetime(equity_df.index[0])
        end_date = pd.to_datetime(equity_df.index[-1])
        years = (end_date - start_date).days / 365.0 if (end_date - start_date).days > 0 else 0.0
        cagr = (end_nav / start_nav) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        running_max = nav.cummax()
        dd = nav / running_max - 1.0
        mdd = float(dd.min())

        if len(trades_df) > 0:
            wins = trades_df["realized_pnl"] > 0
            win_rate = float(wins.mean())
            avg_win = float(trades_df.loc[wins, "realized_pnl"].mean()) if wins.any() else 0.0
            avg_loss = float(trades_df.loc[~wins, "realized_pnl"].mean()) if (~wins).any() else 0.0
            payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (float("inf") if avg_win > 0 else 0.0)
        else:
            win_rate, payoff = 0.0, 0.0

        return {
            "start_nav": start_nav,
            "end_nav": end_nav,
            "cagr": float(cagr),
            "mdd": float(mdd),
            "num_trades": int(len(trades_df)),
            "win_rate": float(win_rate),
            "payoff_ratio": float(payoff),
        }
