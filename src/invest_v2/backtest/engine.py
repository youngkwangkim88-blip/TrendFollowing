from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from invest_v2.core.types import EntryRuleType, Side, Order, TrailingStopType, PyramidingType, TradeMode
from invest_v2.strategy.entry_rules import build_entry_rule, EntryContext
from invest_v2.indicators.ma_cycle import cycle_allows
from invest_v2.utils.krx_ticks import tick_down, tick_up
from invest_v2.trading.trader_master import TraderMaster


@dataclass
class BacktestConfig:
    symbol: str = "005930"
    initial_capital: float = 700_000_000.0
    one_trading_risk: float = 0.01
    max_units_per_symbol: int = 4
    max_units_total: int = 10
    short_notional_limit: float = 570_000_000.0  # backtest assumption

    # Entry
    # - entry_rule is kept for backward compatibility.
    # - If entry_rule_long/entry_rule_short are provided, they override entry_rule by side.
    entry_rule: EntryRuleType = EntryRuleType.A_TURTLE
    entry_rule_long: Optional[EntryRuleType] = None
    entry_rule_short: Optional[EntryRuleType] = None

    # Directional mode
    trade_mode: TradeMode = TradeMode.LONG_SHORT

    # Optional filters (A/B/C)
    # Filter A (PL): 기본 ON.
    # 직전 거래가 이익이면, 바로 다음 반대 포지션 진입을 금지한다.
    filter_pl: bool = True
    # Filter B (ticker cycle): 기본 ON (본 프로젝트 기본 운용 가정)
    filter_cycle: bool = True
    filter_market_cycle: bool = False       # Filter C (market)

    # Minimum position mode (0.5 unit after N consecutive losses)
    min_position_mode: bool = False
    minpos_trigger_consecutive_losses: int = 2
    minpos_entry_factor: float = 0.5
    minpos_first_pyramid_factor: float = 0.5

    # Stop-loss (Initial Stop)
    stop_atr_mult: float = 2.0

    # Trailing Stop
    ts_type: TrailingStopType = TrailingStopType.A_PCT
    # TS.A parameters (% 기반)
    ts_activate_gain: float = 0.20
    ts_floor_gain: float = 0.10
    ts_trail_frac: float = 0.10
    # TS.C parameters (Darvas/Box)
    ts_box_window: int = 20

    # Even-stop (break-even) after MFE threshold
    even_stop_gain: float = 0.10

    # Emergency stop: if close-to-close adverse move exceeds this threshold, exit next open
    # (e.g., 0.05 = 5%)
    emergency_stop_pct: float = 0.05

    # Pyramiding
    pyramiding_type: PyramidingType = PyramidingType.A_PCT
    pyramid_trigger: float = 0.15           # PRMD.A
    pyramid_box_window: int = 20            # PRMD.B (Darvas/Box)
    pyramid_cooldown_days: int = 5          # PRMD.B cooldown

    # Costs / shorts
    sell_cost_rate: float = 0.003
    annual_short_interest_rate: float = 0.045
    short_max_hold_days: int = 90


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    summary: Dict[str, Any]


class SingleSymbolBacktester:
    """Single-symbol backtester with:

    - T close-based signal -> T+1 open execution
    - Initial stop-loss (ATR)
    - Trailing stop variants TS.A/TS.B/TS.C
    - Pyramiding variants PRMD.A/PRMD.B
    - Optional filters A/B/C (PL / cycle / market cycle)
    """

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        self.df = df.copy()
        self.cfg = cfg
        # Entry rules can be specified separately for long/short.
        long_type = cfg.entry_rule_long or cfg.entry_rule
        short_type = cfg.entry_rule_short or cfg.entry_rule
        self.entry_rule_long = build_entry_rule(long_type)
        self.entry_rule_short = build_entry_rule(short_type)
        self.ctx = EntryContext()

        # Minimum position mode state
        self.consecutive_losses: int = 0
        self.minpos_first_pyramid_pending: bool = False

        # --- Account & trader layer ---
        # TraderMaster owns the account ledger and per-symbol Trader(s).
        self.master = TraderMaster(initial_capital_krw=float(cfg.initial_capital))
        self.account = self.master.account
        self.trader = self.master.get_trader(cfg.symbol)

        # Unit sizing capital base (rebased annually at year boundary)
        self.unit_capital_base: float = float(cfg.initial_capital)

        # Position state
        self.pos_side: Side = Side.FLAT
        self.pos_shares: int = 0
        self.pos_units: int = 0
        self.pos_avg_price: float = 0.0  # X
        self.pos_atr_ref: float = 0.0
        self.pos_h_max: float = 0.0
        self.pos_l_min: float = 0.0
        self.pos_ts_active: bool = False
        self.pos_even_armed: bool = False

        # TS.C box trailing level
        self.pos_box_level: float = 0.0

        # short bookkeeping
        self.short_hold_days: int = 0
        self.short_basis_price: float = 0.0
        self.short_open_date: Optional[str] = None

        # cooldown for PRMD.B
        self.last_pyramid_fill_i: Optional[int] = None

        # orders
        self.pending_entry_order: Optional[Order] = None
        self.pending_exit_reason: Optional[str] = None

        self.snapshots: List[Dict[str, Any]] = []
        # Closed trade summaries are stored in Trader (self.trader.trades).


    # -------------------- helpers --------------------
    def _mode_allows(self, side: Side) -> bool:
        if side == Side.FLAT:
            return True
        if self.cfg.trade_mode == TradeMode.LONG_SHORT:
            return True
        if self.cfg.trade_mode == TradeMode.LONG_ONLY:
            return side == Side.LONG
        if self.cfg.trade_mode == TradeMode.SHORT_ONLY:
            return side == Side.SHORT
        return True

    def _holding_value(self, close_price: float) -> float:
        return float(self.pos_shares) * float(close_price) if self.pos_side == Side.LONG else 0.0

    def _short_liability(self, close_price: float) -> float:
        return float(self.pos_shares) * float(close_price) if self.pos_side == Side.SHORT else 0.0

    def _short_notional_basis(self) -> float:
        # Notional basis for short limits / interest should be based on the
        # executed short-sell prices (lot-level).
        if self.pos_side != Side.SHORT or self.pos_shares <= 0:
            return 0.0
        return float(self.trader.entry_notional_gross)

    def _compute_unit_shares(self, M: float, atr10: float) -> int:
        if atr10 <= 0 or np.isnan(atr10):
            return 0
        return int((self.cfg.one_trading_risk * M) // float(atr10))

    def _update_trailing_extrema(self, high: float, low: float) -> None:
        if self.pos_side == Side.LONG:
            self.pos_h_max = max(self.pos_h_max, float(high))
        elif self.pos_side == Side.SHORT:
            self.pos_l_min = float(low) if self.pos_l_min == 0.0 else min(self.pos_l_min, float(low))

    
    def _update_even_stop_arm(self) -> None:
        """Arm break-even stop after the trade has experienced >= +even_stop_gain unrealized profit.

        - Long: arm when H_max >= (1 + g) * X
        - Short: arm when L_min <= (1 - g) * X
        """
        if self.pos_side == Side.FLAT or self.pos_shares <= 0:
            return
        if self.pos_even_armed:
            return
        g = float(self.cfg.even_stop_gain)
        if g <= 0:
            return
        X = float(self.pos_avg_price)
        if X <= 0:
            return
        if self.pos_side == Side.LONG:
            if float(self.pos_h_max) >= (1.0 + g) * X:
                self.pos_even_armed = True
        else:
            if float(self.pos_l_min) > 0 and float(self.pos_l_min) <= (1.0 - g) * X:
                self.pos_even_armed = True

    def _update_box_level(self, df: pd.DataFrame, i: int) -> None:
        if self.cfg.ts_type != TrailingStopType.C_DARVAS_BOX:
            return
        w = int(self.cfg.ts_box_window)
        if self.pos_side == Side.LONG:
            lv = df.get(f"donchian_low_{w}", pd.Series(index=df.index)).iloc[i]
            if pd.isna(lv):
                return
            self.pos_box_level = max(self.pos_box_level, float(lv)) if self.pos_box_level > 0 else float(lv)
        elif self.pos_side == Side.SHORT:
            hv = df.get(f"donchian_high_{w}", pd.Series(index=df.index)).iloc[i]
            if pd.isna(hv):
                return
            self.pos_box_level = min(self.pos_box_level, float(hv)) if self.pos_box_level > 0 else float(hv)

    def _stop_levels(self) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        """Return (stop_loss, ts_level, effective, source).

        source ∈ {"STOP_LOSS", "TS_A", "TS_C", "EVEN_STOP"}
        """
        if self.pos_side == Side.FLAT:
            return None, None, None, "NONE"

        X = float(self.pos_avg_price)
        atr = float(self.pos_atr_ref)
        stop_mult = float(self.cfg.stop_atr_mult)

        # stop-loss is always active
        if self.pos_side == Side.LONG:
            stop_loss = tick_down(X - stop_mult * atr)
        else:
            stop_loss = tick_up(X + stop_mult * atr)

        # Even-stop (break-even) is a price-level stop, independent of TS type.
        even_level: Optional[float] = None
        if self.pos_even_armed and X > 0:
            even_level = tick_down(X) if self.pos_side == Side.LONG else tick_up(X)

        # TS.B is not a price stop (close-based cross), so only (stop-loss + even-stop) here
        if self.cfg.ts_type == TrailingStopType.B_EMA_CROSS:
            if even_level is None:
                return stop_loss, None, stop_loss, "STOP_LOSS"
            if self.pos_side == Side.LONG:
                effective = max(stop_loss, even_level)
                source = "EVEN_STOP" if even_level >= stop_loss else "STOP_LOSS"
            else:
                effective = min(stop_loss, even_level)
                source = "EVEN_STOP" if even_level <= stop_loss else "STOP_LOSS"
            return stop_loss, None, effective, source

        ts_level: Optional[float] = None

        if self.cfg.ts_type == TrailingStopType.A_PCT:
            ts_act = float(self.cfg.ts_activate_gain)
            ts_floor = float(self.cfg.ts_floor_gain)
            ts_trail = float(self.cfg.ts_trail_frac)

            if self.pos_side == Side.LONG:
                if self.pos_h_max >= (1.0 + ts_act) * X:
                    self.pos_ts_active = True
                if self.pos_ts_active:
                    ts_raw = max((1.0 + ts_floor) * X, (1.0 - ts_trail) * float(self.pos_h_max))
                    ts_level = tick_down(ts_raw)
            else:
                if self.pos_l_min > 0 and self.pos_l_min <= (1.0 - ts_act) * X:
                    self.pos_ts_active = True
                if self.pos_ts_active:
                    ts_raw = min((1.0 - ts_floor) * X, (1.0 + ts_trail) * float(self.pos_l_min))
                    ts_level = tick_up(ts_raw)

        # TS.C: Darvas/Box support/resistance trailing
        if self.cfg.ts_type == TrailingStopType.C_DARVAS_BOX:
            if self.pos_box_level > 0:
                ts_level = tick_down(self.pos_box_level) if self.pos_side == Side.LONG else tick_up(self.pos_box_level)

        # combine active price-level stops
        levels: List[Tuple[str, float]] = [("STOP_LOSS", float(stop_loss))]
        if ts_level is not None:
            levels.append(("TS_A" if self.cfg.ts_type == TrailingStopType.A_PCT else "TS_C", float(ts_level)))
        if even_level is not None:
            levels.append(("EVEN_STOP", float(even_level)))

        if self.pos_side == Side.LONG:
            # higher stop is tighter (closer to price from below)
            source, effective = max(levels, key=lambda kv: kv[1])
        else:
            # lower stop is tighter (closer to price from above)
            source, effective = min(levels, key=lambda kv: kv[1])

        return stop_loss, ts_level, float(effective), source


    def _check_stop_trigger(self, df: pd.DataFrame, i: int, o: float, h: float, l: float) -> Optional[Dict[str, Any]]:
        """Check price-level stops using OHLC (GAP/TOUCH model).

        Included stops (always evaluated when a position exists):
          - Initial stop-loss (ATR)
          - TS.A / TS.C (price-level trailing)
          - Even-stop (break-even) after +even_stop_gain MFE
          - Emergency stop ES1: open-based ±p% adverse move (intraday)
          - Emergency stop ES2: prev-close-based ±p% adverse touch (intraday)

        Note
        ----
        Emergency stop ES3 (close-to-close adverse move) is handled separately as a
        close-based scheduled exit at next open (see `_maybe_schedule_emergency_stop`).
        """
        # base effective stop among stop-loss / TS / even-stop
        _, _, base_eff, base_source = self._stop_levels()
        if base_eff is None:
            return None

        p = float(getattr(self.cfg, "emergency_stop_pct", 0.0))
        levels: List[Tuple[str, float]] = [(str(base_source), float(base_eff))]

        if self.pos_side != Side.FLAT and p > 0.0:
            if self.pos_side == Side.LONG:
                # ES1: open-based (O_t)
                levels.append(("EMERGENCY_OPEN", float(tick_down((1.0 - p) * float(o)))))
                # ES2: prev-close-based (C_{t-1})
                if i > 0:
                    c_prev = df.get("close", pd.Series(index=df.index)).iloc[i - 1]
                    if not pd.isna(c_prev) and float(c_prev) > 0:
                        levels.append(("EMERGENCY_PREVCLOSE", float(tick_down((1.0 - p) * float(c_prev)))))
            elif self.pos_side == Side.SHORT:
                levels.append(("EMERGENCY_OPEN", float(tick_up((1.0 + p) * float(o)))))
                if i > 0:
                    c_prev = df.get("close", pd.Series(index=df.index)).iloc[i - 1]
                    if not pd.isna(c_prev) and float(c_prev) > 0:
                        levels.append(("EMERGENCY_PREVCLOSE", float(tick_up((1.0 + p) * float(c_prev)))))

        if self.pos_side == Side.LONG:
            source, eff = max(levels, key=lambda kv: kv[1])
            eff = float(eff)
            if float(o) <= eff:
                return {"exit_price": float(o), "exit_reason": self._format_exit_reason(str(source), "GAP", pct=p)}
            if float(o) > eff and float(l) <= eff:
                return {"exit_price": eff, "exit_reason": self._format_exit_reason(str(source), "TOUCH", pct=p)}
            return None

        if self.pos_side == Side.SHORT:
            source, eff = min(levels, key=lambda kv: kv[1])
            eff = float(eff)
            if float(o) >= eff:
                return {"exit_price": float(o), "exit_reason": self._format_exit_reason(str(source), "GAP", pct=p)}
            if float(o) < eff and float(h) >= eff:
                return {"exit_price": eff, "exit_reason": self._format_exit_reason(str(source), "TOUCH", pct=p)}
            return None

        return None

    @staticmethod
    def _format_exit_reason(source: str, fill_type: str, pct: Optional[float] = None) -> str:
        if source == "STOP_LOSS":
            return f"STOP_LOSS_{fill_type}"
        if source == "TS_A":
            return f"TS_A_{fill_type}"
        if source == "TS_C":
            return f"TS_C_{fill_type}"
        if source == "EVEN_STOP":
            return f"EVEN_STOP_{fill_type}"
        if source == "EMERGENCY_OPEN":
            k = int(round(100.0 * float(pct or 0.0)))
            return f"EMERGENCY_OPEN_{k}PCT_{fill_type}"
        if source == "EMERGENCY_PREVCLOSE":
            k = int(round(100.0 * float(pct or 0.0)))
            return f"EMERGENCY_PREVCLOSE_{k}PCT_{fill_type}"
        return f"STOP_{fill_type}"

    def _exit_position_full(self, date: str, exit_price: float, reason: str) -> None:
        # NOTE: realized PnL must be computed from the actual executed entry lots.
        # Do NOT use a single entry_price field that gets overwritten by pyramiding.
        side = self.pos_side
        if side == Side.FLAT:
            return

        trade_row = self.trader.on_exit_fill(
            date=str(date),
            price=float(exit_price),
            reason=str(reason),
            sell_cost_rate=float(self.cfg.sell_cost_rate),
        )
        if not trade_row:
            return

        shares = int(trade_row.get("shares", 0))
        realized = float(trade_row.get("realized_pnl", 0.0))

        if side == Side.LONG:
            self.account.long_sell(float(exit_price), shares, sell_cost_rate=self.cfg.sell_cost_rate)
        else:
            self.account.short_cover(float(exit_price), shares)

        # update PL filter context
        self.ctx.last_trade_side = side
        self.ctx.last_trade_pnl = float(realized)

        # update min-position state based on outcome
        if self.cfg.min_position_mode:
            if float(realized) < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        if side == Side.SHORT:
            self.account.release_locked()

        # reset position
        self.minpos_first_pyramid_pending = False
        self.pos_side = Side.FLAT
        self.pos_shares = 0
        self.pos_units = 0
        self.pos_avg_price = 0.0
        self.pos_atr_ref = 0.0
        self.pos_h_max = 0.0
        self.pos_l_min = 0.0
        self.pos_ts_active = False
        self.pos_even_armed = False
        self.pos_box_level = 0.0
        self.short_hold_days = 0
        self.short_basis_price = 0.0
        self.short_open_date = None
        # Trader already reset its open-state.
        self.pending_exit_reason = None
        self.last_pyramid_fill_i = None

    def _can_add_unit(self) -> bool:
        return self.pos_units < self.cfg.max_units_per_symbol and self.pos_units < self.cfg.max_units_total

    def _can_open_short_with_limit(self, add_shares: int, sell_price: float) -> bool:
        new_basis_total = self._short_notional_basis() + float(sell_price) * int(add_shares)
        return new_basis_total <= self.cfg.short_notional_limit

    # -------------------- filters --------------------
    def _filter_pl_allows(self, sig: Side, bypass: bool) -> bool:
        if not self.cfg.filter_pl or bypass:
            return True
        if self.ctx.last_trade_side is None or self.ctx.last_trade_pnl is None:
            return True
        if float(self.ctx.last_trade_pnl) > 0 and sig != self.ctx.last_trade_side:
            return False
        return True

    def _filter_cycle_allows(self, df: pd.DataFrame, i: int, sig: Side) -> bool:
        if not self.cfg.filter_cycle:
            return True
        phase = df.get("cycle_phase", pd.Series(index=df.index)).iloc[i]
        return cycle_allows(int(sig.value), phase)

    def _filter_market_cycle_allows(self, df: pd.DataFrame, i: int, sig: Side) -> bool:
        if not self.cfg.filter_market_cycle:
            return True
        phase = df.get("market_cycle_phase", pd.Series(index=df.index)).iloc[i]
        return cycle_allows(int(sig.value), phase)

    # -------------------- TS.B: EMA cross exit --------------------
    def _maybe_schedule_ts_b_exit(self, df: pd.DataFrame, i: int) -> None:
        if self.cfg.ts_type != TrailingStopType.B_EMA_CROSS:
            return
        if self.pos_side == Side.FLAT or self.pending_exit_reason is not None:
            return
        if i <= 0:
            return
        e5_prev = df.get("ema5", pd.Series(index=df.index)).iloc[i - 1]
        e20_prev = df.get("ema20", pd.Series(index=df.index)).iloc[i - 1]
        e5 = df.get("ema5", pd.Series(index=df.index)).iloc[i]
        e20 = df.get("ema20", pd.Series(index=df.index)).iloc[i]
        if pd.isna(e5_prev) or pd.isna(e20_prev) or pd.isna(e5) or pd.isna(e20):
            return

        if self.pos_side == Side.LONG:
            dead = float(e5_prev) >= float(e20_prev) and float(e5) < float(e20)
            if dead:
                self.pending_exit_reason = "TS_B_DEAD_CROSS"
        elif self.pos_side == Side.SHORT:
            golden = float(e5_prev) <= float(e20_prev) and float(e5) > float(e20)
            if golden:
                self.pending_exit_reason = "TS_B_GOLDEN_CROSS"

    # -------------------- Emergency stop (daily adverse move) --------------------
    def _maybe_schedule_emergency_stop(self, df: pd.DataFrame, i: int) -> None:
        """Schedule an emergency exit at next open if today's close moved >= p% against the current position

        Rule (close-to-close):
        - Long:  C_t / C_{t-1} - 1 <= -p  -> exit at O_{t+1}
        - Short: C_t / C_{t-1} - 1 >= +p  -> exit at O_{t+1}

        This guard is independent of TS type and is evaluated only if the position
        still exists at today's close (i.e., not already stopped out intraday).
        """
        if self.pos_side == Side.FLAT:
            return
        if i <= 0:
            return
        p = float(getattr(self.cfg, 'emergency_stop_pct', 0.0))
        if p <= 0:
            return
        c_prev = df.get('close', pd.Series(index=df.index)).iloc[i - 1]
        c_now = df.get('close', pd.Series(index=df.index)).iloc[i]
        if pd.isna(c_prev) or pd.isna(c_now) or float(c_prev) <= 0:
            return
        ret = float(c_now) / float(c_prev) - 1.0
        adverse = (self.pos_side == Side.LONG and ret <= -p) or (self.pos_side == Side.SHORT and ret >= p)
        if adverse:
            # Emergency stop has priority over other close-based exits (e.g., TS.B).
            self.pending_exit_reason = f'EMERGENCY_STOP_{int(round(p * 100))}PCT'

    # -------------------- main loop --------------------
    def run(self) -> BacktestResult:
        df = self.df

        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"df must include columns {sorted(required)}")

        for i in range(len(df)):
            ts = df.index[i]
            date = str(ts.date())
            row = df.iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # hard guard: OHLC must be strictly positive
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                raise ValueError(f"Invalid OHLC (<=0) at {date}: O={o},H={h},L={l},C={c}. Please sanitize your data.")

            # first trading day of month?
            is_first_trading_day = True if i == 0 else (df.index[i - 1].month != ts.month)
            self.account.apply_monthly_interest_if_first_trading_day(is_first_trading_day_of_month=is_first_trading_day)

            # 1) execute pending exit at open
            if self.pending_exit_reason is not None and self.pos_side != Side.FLAT:
                self.account.ensure_trading_cash()
                self._exit_position_full(date=date, exit_price=float(o), reason=str(self.pending_exit_reason))
                self.pending_exit_reason = None
                # any pending entry becomes invalid (safety)
                self.pending_entry_order = None

            # 2) execute pending entry/pyramiding at open
            if self.pending_entry_order is not None:
                self.account.ensure_trading_cash()
                order = self.pending_entry_order
                self.pending_entry_order = None

                fill = float(o)
                add_shares = int(order.unit_shares)
                if add_shares > 0:
                    if order.side == Side.LONG:
                        self.account.long_buy(fill, add_shares)
                        if self.pos_side == Side.FLAT:
                            # Trader (lot-level) entry
                            self.trader.on_entry_fill(
                                date=date,
                                side=Side.LONG,
                                price=fill,
                                shares=add_shares,
                                reason=str(order.reason),
                                is_pyramid=False,
                            )
                            self.pos_side = Side.LONG
                            self.pos_units = int(self.trader.pos_units)
                            self.pos_shares = int(self.trader.pos_shares)
                            self.pos_avg_price = float(self.trader.avg_entry_price)
                            self.pos_atr_ref = float(order.atr_ref)
                            self.pos_h_max = max(h, fill)
                            self.pos_ts_active = False
                            self.pos_even_armed = False
                            self.pos_box_level = 0.0
                            if self.cfg.pyramiding_type == PyramidingType.B_DARVAS_BOX:
                                self.last_pyramid_fill_i = i
                        else:
                            # Pyramiding long: keep first entry date/price for reporting,
                            # while updating avg price via lot-level accounting.
                            self.trader.on_entry_fill(
                                date=date,
                                side=Side.LONG,
                                price=fill,
                                shares=add_shares,
                                reason=str(order.reason),
                                is_pyramid=True,
                            )
                            self.pos_shares = int(self.trader.pos_shares)
                            self.pos_units = int(self.trader.pos_units)
                            self.pos_avg_price = float(self.trader.avg_entry_price)
                            self.pos_atr_ref = float(order.atr_ref)

                            # NOTE: Do NOT reset trailing extrema / TS / even-stop state.
                            # We keep H_max/L_min and TS activation to avoid losing the protective stop.

                        if str(order.reason).upper().startswith("PYRAMID"):
                            self.last_pyramid_fill_i = i

                    else:
                        if self._can_open_short_with_limit(add_shares, sell_price=fill):
                            self.account.short_sell(fill, add_shares, sell_cost_rate=self.cfg.sell_cost_rate)
                            if self.pos_side == Side.FLAT:
                                self.trader.on_entry_fill(
                                    date=date,
                                    side=Side.SHORT,
                                    price=fill,
                                    shares=add_shares,
                                    reason=str(order.reason),
                                    is_pyramid=False,
                                )
                                self.pos_side = Side.SHORT
                                self.pos_units = int(self.trader.pos_units)
                                self.pos_shares = int(self.trader.pos_shares)
                                self.pos_avg_price = float(self.trader.avg_entry_price)
                                self.pos_atr_ref = float(order.atr_ref)
                                self.pos_l_min = min(l, fill)
                                self.pos_ts_active = False
                                self.pos_even_armed = False
                                self.pos_box_level = 0.0
                                self.short_hold_days = 0
                                self.short_open_date = date
                                # Keep legacy snapshot column (avg basis price)
                                self.short_basis_price = float(self.trader.avg_entry_price)
                                if self.cfg.pyramiding_type == PyramidingType.B_DARVAS_BOX:
                                    self.last_pyramid_fill_i = i
                            else:
                                self.trader.on_entry_fill(
                                    date=date,
                                    side=Side.SHORT,
                                    price=fill,
                                    shares=add_shares,
                                    reason=str(order.reason),
                                    is_pyramid=True,
                                )
                                self.pos_shares = int(self.trader.pos_shares)
                                self.pos_units = int(self.trader.pos_units)
                                self.pos_avg_price = float(self.trader.avg_entry_price)
                                self.pos_atr_ref = float(order.atr_ref)

                                # NOTE: Do NOT reset trailing extrema / TS / even-stop state.
                                self.short_basis_price = float(self.trader.avg_entry_price)

                            if str(order.reason).upper().startswith("PYRAMID"):
                                self.last_pyramid_fill_i = i

                        else:
                            # skipped due to notional limit
                            pass

            # 3) forced short exit (90 trading days) at open
            if self.pos_side == Side.SHORT and self.short_hold_days >= self.cfg.short_max_hold_days:
                self._exit_position_full(date=date, exit_price=o, reason="SHORT_MAX_HOLD_DAYS")
            else:
                # update extrema and TS.C box
                if self.pos_side != Side.FLAT:
                    self._update_trailing_extrema(high=h, low=l)
                    self._update_even_stop_arm()
                    self._update_box_level(df, i)

                # intraday stop / TS.A / TS.C check
                exit_info = self._check_stop_trigger(df=df, i=i, o=o, h=h, l=l)
                if exit_info is not None:
                    self._exit_position_full(date=date, exit_price=float(exit_info["exit_price"]), reason=str(exit_info["exit_reason"]))

            # accrue interest EOD
            if self.pos_side == Side.SHORT:
                self.account.accrue_daily_short_interest(self._short_notional_basis(), annual_rate=self.cfg.annual_short_interest_rate)
                self.short_hold_days += 1

            # schedule emergency stop using today's close (executes tomorrow open)
            self._maybe_schedule_emergency_stop(df, i)

            # schedule TS.B exit using today's close (executes tomorrow open)
            self._maybe_schedule_ts_b_exit(df, i)

            # generate next-day entry/pyramiding orders (signal evaluated at close)
            if i < len(df) - 1:
                # Annual unit sizing rebase (compounding):
                # If the next trading day is in a new calendar year, set next year's M to today's EOD NAV.
                next_ts = df.index[i + 1]
                if ts.year != next_ts.year:
                    holding_value_tmp = self._holding_value(c)
                    short_liab_tmp = self._short_liability(c)
                    nav_tmp = self.account.nav(holding_value_tmp, short_liab_tmp)
                    self.unit_capital_base = float(nav_tmp)

                atr10_today = df.get("atr10", pd.Series(index=df.index)).iloc[i]
                atr_ref = float(atr10_today) if not pd.isna(atr10_today) else 0.0

                # (a) entry
                if self.pos_side == Side.FLAT and self.pending_entry_order is None:
                    # Evaluate long/short entry independently (can use different rule types).
                    # Each rule may emit both sides; we gate by desired side.
                    d_long = self.entry_rule_long.evaluate(df, i, self.ctx)
                    d_short = self.entry_rule_short.evaluate(df, i, self.ctx)

                    sig_long = d_long.side if d_long.side == Side.LONG else Side.FLAT
                    sig_short = d_short.side if d_short.side == Side.SHORT else Side.FLAT

                    # Apply directional mode gating.
                    if not self._mode_allows(Side.LONG):
                        sig_long = Side.FLAT
                    if not self._mode_allows(Side.SHORT):
                        sig_short = Side.FLAT

                    # Resolve (rare) conflicts.
                    if sig_long != Side.FLAT and sig_short != Side.FLAT:
                        sig = Side.FLAT
                        decision_reason = "CONFLICT_LONG_AND_SHORT"
                        decision_bypass_pl = False
                    elif sig_long != Side.FLAT:
                        sig = sig_long
                        decision_reason = d_long.reason
                        decision_bypass_pl = bool(d_long.bypass_filter_pl)
                    elif sig_short != Side.FLAT:
                        sig = sig_short
                        decision_reason = d_short.reason
                        decision_bypass_pl = bool(d_short.bypass_filter_pl)
                    else:
                        sig = Side.FLAT
                        decision_reason = ""
                        decision_bypass_pl = False

                    if sig != Side.FLAT:
                        # filters A/B/C
                        if not self._filter_pl_allows(sig, bypass=bool(decision_bypass_pl)):
                            sig = Side.FLAT
                        elif not self._filter_cycle_allows(df, i, sig):
                            sig = Side.FLAT
                        elif not self._filter_market_cycle_allows(df, i, sig):
                            sig = Side.FLAT

                    if sig != Side.FLAT:
                        unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                        if self.cfg.min_position_mode and self.consecutive_losses >= int(self.cfg.minpos_trigger_consecutive_losses):
                            unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_entry_factor)))
                            # first pyramiding after this entry uses reduced size once
                            self.minpos_first_pyramid_pending = True
                        else:
                            self.minpos_first_pyramid_pending = False
                        if unit_shares > 0:
                            if sig == Side.SHORT and not self._can_open_short_with_limit(unit_shares, sell_price=c):
                                pass
                            else:
                                self.pending_entry_order = Order(
                                    symbol=self.cfg.symbol,
                                    side=sig,
                                    unit_shares=unit_shares,
                                    atr_ref=atr_ref,
                                    signal_date=date,
                                    reason=str(decision_reason) if decision_reason else "ENTRY",
                                )

                # (b) pyramiding
                if self.pos_side != Side.FLAT and self.pending_entry_order is None and self.pending_exit_reason is None and self._can_add_unit():
                    if self.cfg.pyramiding_type != PyramidingType.OFF:
                        if self.pos_side == Side.LONG:
                            if self._pyramid_long(df, i, c, atr_ref, date):
                                pass
                        elif self.pos_side == Side.SHORT and self._mode_allows(Side.SHORT):
                            if self._pyramid_short(df, i, c, atr_ref, date):
                                pass

            # park to CMA if flat and no pending order
            if self.pos_side == Side.FLAT and self.pending_entry_order is None:
                self.account.park_to_cma_if_flat()

            # snapshot EOD NAV (close-based)
            holding_value = self._holding_value(c)
            short_liab = self._short_liability(c)
            nav = self.account.nav(holding_value, short_liab)
            self.snapshots.append({
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
                "unit_capital_base": float(self.unit_capital_base),
            })

        equity_df = pd.DataFrame(self.snapshots).set_index("date")
        trades_df = pd.DataFrame(self.trader.trades)
        fills_df = pd.DataFrame(self.trader.fills)
        summary = self._summarize(equity_df, trades_df)
        return BacktestResult(equity_curve=equity_df, trades=trades_df, fills=fills_df, summary=summary)

    # -------------------- pyramiding --------------------
    def _pyramid_long(self, df: pd.DataFrame, i: int, close: float, atr_ref: float, date: str) -> bool:
        if self.cfg.pyramiding_type == PyramidingType.A_PCT:
            if close >= (1.0 + float(self.cfg.pyramid_trigger)) * float(self.pos_avg_price):
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    # consume reduced-size once (only for the first pyramiding after reduced entry)
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0:
                    self.pending_entry_order = Order(
                        symbol=self.cfg.symbol,
                        side=Side.LONG,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_LONG_PRMD.A",
                    )
                    return True
            return False

        if self.cfg.pyramiding_type == PyramidingType.B_DARVAS_BOX:
            # cooldown
            cd = int(self.cfg.pyramid_cooldown_days)
            if self.last_pyramid_fill_i is not None and (i - int(self.last_pyramid_fill_i)) < cd:
                return False
            w = int(self.cfg.pyramid_box_window)
            dh = df.get(f"donchian_high_{w}", pd.Series(index=df.index)).iloc[i]
            if pd.isna(dh):
                return False
            if close >= float(dh):
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    # consume reduced-size once (only for the first pyramiding after reduced entry)
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0:
                    self.pending_entry_order = Order(
                        symbol=self.cfg.symbol,
                        side=Side.LONG,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_LONG_PRMD.B",
                    )
                    return True
            return False

        return False

    def _pyramid_short(self, df: pd.DataFrame, i: int, close: float, atr_ref: float, date: str) -> bool:
        if self.cfg.pyramiding_type == PyramidingType.A_PCT:
            if close <= (1.0 - float(self.cfg.pyramid_trigger)) * float(self.pos_avg_price):
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    # consume reduced-size once (only for the first pyramiding after reduced entry)
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0 and self._can_open_short_with_limit(unit_shares, sell_price=close):
                    self.pending_entry_order = Order(
                        symbol=self.cfg.symbol,
                        side=Side.SHORT,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_SHORT_PRMD.A",
                    )
                    return True
            return False

        if self.cfg.pyramiding_type == PyramidingType.B_DARVAS_BOX:
            cd = int(self.cfg.pyramid_cooldown_days)
            if self.last_pyramid_fill_i is not None and (i - int(self.last_pyramid_fill_i)) < cd:
                return False
            w = int(self.cfg.pyramid_box_window)
            dl = df.get(f"donchian_low_{w}", pd.Series(index=df.index)).iloc[i]
            if pd.isna(dl):
                return False
            if close <= float(dl):
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    # consume reduced-size once (only for the first pyramiding after reduced entry)
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0 and self._can_open_short_with_limit(unit_shares, sell_price=close):
                    self.pending_entry_order = Order(
                        symbol=self.cfg.symbol,
                        side=Side.SHORT,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_SHORT_PRMD.B",
                    )
                    return True
            return False

        return False

    # -------------------- summary --------------------
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

        if len(trades_df) > 0 and "realized_pnl" in trades_df.columns:
            wins = trades_df["realized_pnl"].astype(float) > 0
            win_rate = float(wins.mean())
            avg_win = float(trades_df.loc[wins, "realized_pnl"].astype(float).mean()) if wins.any() else 0.0
            avg_loss = float(trades_df.loc[~wins, "realized_pnl"].astype(float).mean()) if (~wins).any() else 0.0
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