from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from invest_v2.core.types import (
    EntryRuleType,
    Order,
    PyramidingType,
    Side,
    Trade,
    TradeMode,
    TrailingStopType,
)
from invest_v2.strategy.entry_rules import build_entry_rule, EntryContext
from invest_v2.indicators.ma_cycle import cycle_allows
from invest_v2.utils.krx_ticks import tick_down, tick_up


@dataclass
class FillEvent:
    """Execution-level record (one fill)."""

    trader_id: str
    symbol: str
    side: int  # 1=LONG, -1=SHORT
    fill_date: str
    fill_price: float
    fill_shares: int
    fill_type: str  # ENTRY | PYRAMID | EXIT
    reason: str

    pos_units_after: int
    pos_shares_after: int
    avg_price_after: float


@dataclass
class TraderConfig:
    """Per-trader configuration.

    This is intentionally close to the legacy BacktestConfig, but scoped to a
    trader instance (symbol + strategy + sizing).

    The portfolio-wide constraints (e.g., max total units) are enforced by TraderMaster.
    """

    trader_id: str
    symbol: str

    # allocation / sizing
    initial_capital: float = 700_000_000.0
    one_trading_risk: float = 0.01
    max_units_per_symbol: int = 4
    short_notional_limit: float = 570_000_000.0

    # entry rules
    entry_rule: EntryRuleType = EntryRuleType.A_TURTLE
    entry_rule_long: Optional[EntryRuleType] = None
    entry_rule_short: Optional[EntryRuleType] = None

    # direction
    trade_mode: TradeMode = TradeMode.LONG_SHORT

    # filters
    filter_pl: bool = True
    filter_cycle: bool = True
    filter_market_cycle: bool = False

    # min position mode
    min_position_mode: bool = False
    minpos_trigger_consecutive_losses: int = 2
    minpos_entry_factor: float = 0.5
    minpos_first_pyramid_factor: float = 0.5

    # initial stop
    stop_atr_mult: float = 2.0

    # trailing stop
    ts_type: TrailingStopType = TrailingStopType.A_PCT
    ts_activate_gain: float = 0.20
    ts_floor_gain: float = 0.10
    ts_trail_frac: float = 0.10
    ts_box_window: int = 20

    # even-stop
    even_stop_gain: float = 0.10

    # emergency stop
    emergency_stop_pct: float = 0.05

    # pyramiding
    pyramiding_type: PyramidingType = PyramidingType.A_PCT
    pyramid_trigger: float = 0.15
    pyramid_box_window: int = 20
    pyramid_cooldown_days: int = 5

    # costs / shorts
    sell_cost_rate: float = 0.003
    annual_short_interest_rate: float = 0.045
    short_max_hold_days: int = 90

    # priority for arbitration when multiple traders target the same symbol
    priority: int = 0


class Trader:
    """Trader owns *position state* and trading logs for one symbol.

    The trader is responsible for:
      - Maintaining position state (avg price, units, TS/even/emergency states)
      - Generating orders based on entry rules + filters
      - Scheduling next-open orders (T close -> T+1 open)
      - Recording fills and closed trades

    Execution and account cashflows are delegated to TraderMaster.
    """

    def __init__(self, cfg: TraderConfig):
        self.cfg = cfg
        self.trader_id: str = str(cfg.trader_id)
        self.symbol: str = str(cfg.symbol)

        # Entry rules can differ by side.
        long_type = cfg.entry_rule_long or cfg.entry_rule
        short_type = cfg.entry_rule_short or cfg.entry_rule
        self.entry_rule_long = build_entry_rule(long_type)
        self.entry_rule_short = build_entry_rule(short_type)
        self.ctx = EntryContext()

        # Minimum position mode state
        self.consecutive_losses: int = 0
        self.minpos_first_pyramid_pending: bool = False

        # Unit sizing base (rebased annually)
        self.unit_capital_base: float = float(cfg.initial_capital)

        # Position state (owned by Trader)
        self.pos_side: Side = Side.FLAT
        self.pos_shares: int = 0
        self.pos_units: int = 0
        self.pos_avg_price: float = 0.0  # X (avg entry)
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

        # orders (scheduled)
        self.pending_entry_order: Optional[Order] = None
        self.pending_exit_reason: Optional[str] = None

        # trade aggregation + logs
        self.active_trade: Optional[Trade] = None
        self.closed_trades: List[Trade] = []
        self.fills: List[FillEvent] = []

        # optional debugging flags
        self.debug: bool = False

    # -------------------- introspection --------------------
    def how_did_you_trade(self, max_lines: int = 200) -> str:
        """Human-readable trading journal (fills + closed trades)."""

        lines: List[str] = []
        lines.append(f"Trader[{self.trader_id}] symbol={self.symbol} mode={self.cfg.trade_mode}")
        lines.append(f"Closed trades: {len(self.closed_trades)}, fills: {len(self.fills)}")
        lines.append("-")

        # last N fills
        for f in self.fills[-max(0, int(max_lines)) :]:
            lines.append(
                f"{f.fill_date} | {f.fill_type:<7} | side={f.side:+d} | px={f.fill_price:.2f} | sh={f.fill_shares} | units={f.pos_units_after} | reason={f.reason}"
            )

        if len(self.closed_trades):
            lines.append("-")
            lines.append("Closed trade summaries:")
            for t in self.closed_trades[-min(30, len(self.closed_trades)) :]:
                lines.append(
                    f"{t.entry_date} -> {t.exit_date} | side={int(t.side.value):+d} | first={t.entry_price:.2f} | avg={t.avg_entry_price:.2f} | sh={t.shares} | pnl={float(t.realized_pnl or 0.0):,.0f} | {t.exit_reason}"
                )

        return "\n".join(lines)

    def trades_df(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for t in self.closed_trades:
            rows.append(
                {
                    "trader_id": self.trader_id,
                    "symbol": t.symbol,
                    "side": int(t.side.value),
                    "entry_date": t.entry_date,
                    "entry_price": float(t.entry_price),
                    "avg_entry_price": float(t.avg_entry_price),
                    "num_entries": int(t.num_entries),
                    "shares": int(t.shares),
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "realized_pnl": t.realized_pnl,
                    "exit_reason": t.exit_reason,
                }
            )
        return pd.DataFrame(rows)

    def fills_df(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for f in self.fills:
            rows.append(
                {
                    "trader_id": f.trader_id,
                    "symbol": f.symbol,
                    "side": int(f.side),
                    "fill_date": f.fill_date,
                    "fill_price": float(f.fill_price),
                    "fill_shares": int(f.fill_shares),
                    "fill_type": f.fill_type,
                    "reason": f.reason,
                    "pos_units_after": int(f.pos_units_after),
                    "pos_shares_after": int(f.pos_shares_after),
                    "avg_price_after": float(f.avg_price_after),
                }
            )
        return pd.DataFrame(rows)

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
        if self.pos_side != Side.SHORT or self.pos_shares <= 0:
            return 0.0
        return float(self.short_basis_price) * float(self.pos_shares)

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
        """Return (stop_loss, ts_level, effective, source)."""
        if self.pos_side == Side.FLAT:
            return None, None, None, "NONE"

        X = float(self.pos_avg_price)
        atr = float(self.pos_atr_ref)
        stop_mult = float(self.cfg.stop_atr_mult)

        if self.pos_side == Side.LONG:
            stop_loss = tick_down(X - stop_mult * atr)
        else:
            stop_loss = tick_up(X + stop_mult * atr)

        even_level: Optional[float] = None
        if self.pos_even_armed and X > 0:
            even_level = tick_down(X) if self.pos_side == Side.LONG else tick_up(X)

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

        if self.cfg.ts_type == TrailingStopType.C_DARVAS_BOX:
            if self.pos_box_level > 0:
                ts_level = tick_down(self.pos_box_level) if self.pos_side == Side.LONG else tick_up(self.pos_box_level)

        levels: List[Tuple[str, float]] = [("STOP_LOSS", float(stop_loss))]
        if ts_level is not None:
            levels.append(("TS_A" if self.cfg.ts_type == TrailingStopType.A_PCT else "TS_C", float(ts_level)))
        if even_level is not None:
            levels.append(("EVEN_STOP", float(even_level)))

        if self.pos_side == Side.LONG:
            source, effective = max(levels, key=lambda kv: kv[1])
        else:
            source, effective = min(levels, key=lambda kv: kv[1])

        return stop_loss, ts_level, float(effective), source

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

    def _check_stop_trigger(self, df: pd.DataFrame, i: int, o: float, h: float, l: float) -> Optional[Dict[str, Any]]:
        """Price-level stop trigger check (GAP/TOUCH model)."""

        _, _, base_eff, base_source = self._stop_levels()
        if base_eff is None:
            return None

        p = float(getattr(self.cfg, "emergency_stop_pct", 0.0))
        levels: List[Tuple[str, float]] = [(str(base_source), float(base_eff))]

        if self.pos_side != Side.FLAT and p > 0.0:
            if self.pos_side == Side.LONG:
                levels.append(("EMERGENCY_OPEN", float(tick_down((1.0 - p) * float(o)))))
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

    # -------------------- exits scheduled at next open (TS.B / ES3) --------------------
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

    def _maybe_schedule_emergency_stop_c2c(self, df: pd.DataFrame, i: int) -> None:
        if self.pos_side == Side.FLAT:
            return
        if i <= 0:
            return
        p = float(getattr(self.cfg, "emergency_stop_pct", 0.0))
        if p <= 0:
            return
        c_prev = df.get("close", pd.Series(index=df.index)).iloc[i - 1]
        c_now = df.get("close", pd.Series(index=df.index)).iloc[i]
        if pd.isna(c_prev) or pd.isna(c_now) or float(c_prev) <= 0:
            return
        ret = float(c_now) / float(c_prev) - 1.0
        adverse = (self.pos_side == Side.LONG and ret <= -p) or (self.pos_side == Side.SHORT and ret >= p)
        if adverse:
            self.pending_exit_reason = f"EMERGENCY_STOP_{int(round(p * 100))}PCT"

    # -------------------- execution primitives --------------------
    def _record_fill(self, *, date: str, price: float, shares: int, fill_type: str, reason: str) -> None:
        self.fills.append(
            FillEvent(
                trader_id=self.trader_id,
                symbol=self.symbol,
                side=int(self.pos_side.value if self.pos_side != Side.FLAT else (1 if "LONG" in reason else -1 if "SHORT" in reason else 0)),
                fill_date=date,
                fill_price=float(price),
                fill_shares=int(shares),
                fill_type=str(fill_type),
                reason=str(reason),
                pos_units_after=int(self.pos_units),
                pos_shares_after=int(self.pos_shares),
                avg_price_after=float(self.pos_avg_price),
            )
        )

    def _enter_new_position(self, *, date: str, fill: float, add_shares: int, atr_ref: float, side: Side, reason: str, master: Any, bar_high: float, bar_low: float, i: int) -> None:
        """Apply a new entry fill at open."""

        # execution (cashflow) is master-owned
        if side == Side.LONG:
            master.long_buy(self.trader_id, fill, add_shares)
        else:
            master.short_sell(self.trader_id, fill, add_shares, sell_cost_rate=float(self.cfg.sell_cost_rate))

        self.pos_side = side
        self.pos_units = 1
        self.pos_shares = int(add_shares)
        self.pos_avg_price = float(fill)
        self.pos_atr_ref = float(atr_ref)
        self.pos_ts_active = False
        self.pos_even_armed = False
        self.pos_box_level = 0.0

        if side == Side.LONG:
            self.pos_h_max = max(float(bar_high), float(fill))
            self.pos_l_min = 0.0
        else:
            self.pos_l_min = min(float(bar_low), float(fill))
            self.pos_h_max = 0.0
            self.short_hold_days = 0
            self.short_open_date = date
            self.short_basis_price = float(fill)

        # trade record (entry_price = FIRST fill; avg_entry_price = current avg)
        self.active_trade = Trade(
            symbol=self.symbol,
            side=side,
            entry_date=date,
            entry_price=float(fill),
            shares=int(add_shares),
            avg_entry_price=float(fill),
            num_entries=1,
            entry_notional_gross=float(fill) * int(add_shares),
        )

        # PRMD.B cooldown baseline should start at initial entry
        if self.cfg.pyramiding_type == PyramidingType.B_DARVAS_BOX:
            self.last_pyramid_fill_i = int(i)

        # master symbol ownership
        master.on_position_opened(self.trader_id, self.symbol, side)

        self._record_fill(date=date, price=fill, shares=add_shares, fill_type="ENTRY", reason=reason)

    def _pyramid_existing(self, *, date: str, fill: float, add_shares: int, atr_ref: float, reason: str, master: Any, i: int) -> None:
        """Apply pyramiding fill."""

        if self.pos_side == Side.LONG:
            master.long_buy(self.trader_id, fill, add_shares)
        else:
            master.short_sell(self.trader_id, fill, add_shares, sell_cost_rate=float(self.cfg.sell_cost_rate))

        new_total = int(self.pos_shares) + int(add_shares)
        if new_total <= 0:
            return

        # update avg entry price (X)
        self.pos_avg_price = (float(self.pos_avg_price) * int(self.pos_shares) + float(fill) * int(add_shares)) / float(new_total)
        self.pos_shares = int(new_total)
        self.pos_units += 1
        self.pos_atr_ref = float(atr_ref)

        # Short basis (for interest / notional limit)
        if self.pos_side == Side.SHORT:
            self.short_basis_price = (float(self.short_basis_price) * (new_total - int(add_shares)) + float(fill) * int(add_shares)) / float(new_total)

        # Update aggregated trade record (do NOT overwrite first entry)
        if self.active_trade:
            self.active_trade.shares = int(self.pos_shares)
            self.active_trade.num_entries += 1
            self.active_trade.entry_notional_gross += float(fill) * int(add_shares)
            self.active_trade.avg_entry_price = float(self.active_trade.entry_notional_gross) / float(self.active_trade.shares)

        # PRMD.B cooldown anchor (each pyramid fill)
        self.last_pyramid_fill_i = int(i)

        self._record_fill(date=date, price=fill, shares=add_shares, fill_type="PYRAMID", reason=reason)

    def _exit_position_full(self, *, date: str, exit_price: float, reason: str, master: Any) -> None:
        shares = int(self.pos_shares)
        if shares <= 0 or self.active_trade is None or self.pos_side == Side.FLAT:
            return

        if self.pos_side == Side.LONG:
            entry_cost = float(self.active_trade.entry_notional_gross)
            exit_net = float(exit_price) * shares * (1.0 - float(self.cfg.sell_cost_rate))
            realized = float(exit_net - entry_cost)
            master.long_sell(self.trader_id, float(exit_price), shares, sell_cost_rate=float(self.cfg.sell_cost_rate))
        else:
            entry_net = float(self.active_trade.entry_notional_gross) * (1.0 - float(self.cfg.sell_cost_rate))
            cover_cost = float(exit_price) * shares
            realized = float(entry_net - cover_cost)
            master.short_cover(self.trader_id, float(exit_price), shares)

        self.active_trade.exit_date = date
        self.active_trade.exit_price = float(exit_price)
        self.active_trade.realized_pnl = float(realized)
        self.active_trade.exit_reason = str(reason)

        self.closed_trades.append(self.active_trade)

        # update PL filter context
        self.ctx.last_trade_side = self.active_trade.side
        self.ctx.last_trade_pnl = float(realized)

        # update min-position state based on outcome
        if self.cfg.min_position_mode:
            if float(realized) < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        if self.pos_side == Side.SHORT:
            master.release_locked(self.trader_id)

        # master symbol ownership release
        master.on_position_closed(self.trader_id, self.symbol)

        # record fill
        self._record_fill(date=date, price=float(exit_price), shares=shares, fill_type="EXIT", reason=str(reason))

        # reset position state
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
        self.active_trade = None
        self.pending_exit_reason = None
        self.last_pyramid_fill_i = None

    # -------------------- pyramiding decision --------------------
    def _pyramid_long(self, df: pd.DataFrame, i: int, close: float, atr_ref: float, date: str, master: Any) -> bool:
        if self.cfg.pyramiding_type == PyramidingType.A_PCT:
            if close >= (1.0 + float(self.cfg.pyramid_trigger)) * float(self.pos_avg_price):
                if not master.can_add_unit(self.trader_id):
                    return False
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0:
                    self.pending_entry_order = Order(
                        symbol=self.symbol,
                        side=Side.LONG,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_LONG_PRMD.A",
                    )
                    return True
            return False

        if self.cfg.pyramiding_type == PyramidingType.B_DARVAS_BOX:
            cd = int(self.cfg.pyramid_cooldown_days)
            if self.last_pyramid_fill_i is not None and (i - int(self.last_pyramid_fill_i)) < cd:
                return False
            w = int(self.cfg.pyramid_box_window)
            dh = df.get(f"donchian_high_{w}", pd.Series(index=df.index)).iloc[i]
            if pd.isna(dh):
                return False
            if close >= float(dh):
                if not master.can_add_unit(self.trader_id):
                    return False
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0:
                    self.pending_entry_order = Order(
                        symbol=self.symbol,
                        side=Side.LONG,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_LONG_PRMD.B",
                    )
                    return True
            return False

        return False

    def _pyramid_short(self, df: pd.DataFrame, i: int, close: float, atr_ref: float, date: str, master: Any) -> bool:
        if self.cfg.pyramiding_type == PyramidingType.A_PCT:
            if close <= (1.0 - float(self.cfg.pyramid_trigger)) * float(self.pos_avg_price):
                if not master.can_add_unit(self.trader_id):
                    return False
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0 and (self._short_notional_basis() + float(close) * int(unit_shares)) <= float(self.cfg.short_notional_limit):
                    self.pending_entry_order = Order(
                        symbol=self.symbol,
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
                if not master.can_add_unit(self.trader_id):
                    return False
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.minpos_first_pyramid_pending:
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_first_pyramid_factor)))
                    self.minpos_first_pyramid_pending = False
                if unit_shares > 0 and (self._short_notional_basis() + float(close) * int(unit_shares)) <= float(self.cfg.short_notional_limit):
                    self.pending_entry_order = Order(
                        symbol=self.symbol,
                        side=Side.SHORT,
                        unit_shares=unit_shares,
                        atr_ref=atr_ref,
                        signal_date=date,
                        reason="PYRAMID_SHORT_PRMD.B",
                    )
                    return True
            return False

        return False

    # -------------------- main callbacks (per day) --------------------
    def on_day_open(self, df: pd.DataFrame, i: int, date: str, o: float, h: float, l: float, c: float, master: Any) -> None:
        """Execute pending scheduled orders at today's open."""

        # 1) pending exit at open
        if self.pending_exit_reason is not None and self.pos_side != Side.FLAT:
            master.ensure_trading_cash(self.trader_id)

            # Collision handling: if a price-level stop is already gapped at today's open,
            # classify the exit as that stop (reason priority) rather than the scheduled
            # close-based exit (TS.B / ES3). This makes reporting of severe gap events
            # more faithful and prevents "double exit" style regressions.
            gap_stop = self._check_stop_trigger(df=df, i=i, o=float(o), h=float(o), l=float(o))
            reason = gap_stop["exit_reason"] if gap_stop is not None else str(self.pending_exit_reason)
            self._exit_position_full(date=date, exit_price=float(o), reason=reason, master=master)
            self.pending_exit_reason = None
            self.pending_entry_order = None

        # 2) pending entry/pyramiding at open
        if self.pending_entry_order is not None:
            order = self.pending_entry_order
            self.pending_entry_order = None

            # symbol exclusivity / global units check
            if not master.can_open_symbol(self.trader_id, self.symbol):
                return

            # move CMA cash into trading cash before any fill
            master.ensure_trading_cash(self.trader_id)

            fill = float(o)
            add_shares = int(order.unit_shares)
            if add_shares <= 0:
                return

            # If flat -> entry; else -> pyramid
            if order.side == Side.LONG:
                if self.pos_side == Side.FLAT:
                    self._enter_new_position(
                        date=date,
                        fill=fill,
                        add_shares=add_shares,
                        atr_ref=float(order.atr_ref),
                        side=Side.LONG,
                        reason=str(order.reason) or "ENTRY",
                        master=master,
                        bar_high=h,
                        bar_low=l,
                        i=i,
                    )
                else:
                    # pyramiding
                    if self.pos_units < int(self.cfg.max_units_per_symbol) and master.can_add_unit(self.trader_id):
                        self._pyramid_existing(
                            date=date,
                            fill=fill,
                            add_shares=add_shares,
                            atr_ref=float(order.atr_ref),
                            reason=str(order.reason) or "PYRAMID",
                            master=master,
                            i=i,
                        )

            elif order.side == Side.SHORT:
                # short notional limit
                if (self._short_notional_basis() + float(fill) * int(add_shares)) > float(self.cfg.short_notional_limit):
                    return

                if self.pos_side == Side.FLAT:
                    self._enter_new_position(
                        date=date,
                        fill=fill,
                        add_shares=add_shares,
                        atr_ref=float(order.atr_ref),
                        side=Side.SHORT,
                        reason=str(order.reason) or "ENTRY",
                        master=master,
                        bar_high=h,
                        bar_low=l,
                        i=i,
                    )
                else:
                    if self.pos_units < int(self.cfg.max_units_per_symbol) and master.can_add_unit(self.trader_id):
                        self._pyramid_existing(
                            date=date,
                            fill=fill,
                            add_shares=add_shares,
                            atr_ref=float(order.atr_ref),
                            reason=str(order.reason) or "PYRAMID",
                            master=master,
                            i=i,
                        )

    def on_day_intraday(self, df: pd.DataFrame, i: int, date: str, o: float, h: float, l: float, c: float, master: Any) -> None:
        """Intraday stop handling and forced exits."""

        # forced short exit at open already handled by caller (we handle here after fills)
        if self.pos_side == Side.SHORT and self.short_hold_days >= int(self.cfg.short_max_hold_days):
            self._exit_position_full(date=date, exit_price=float(o), reason="SHORT_MAX_HOLD_DAYS", master=master)
            return

        if self.pos_side != Side.FLAT:
            self._update_trailing_extrema(high=h, low=l)
            self._update_even_stop_arm()
            self._update_box_level(df, i)

        # intraday price-level stops
        exit_info = self._check_stop_trigger(df=df, i=i, o=o, h=h, l=l)
        if exit_info is not None:
            self._exit_position_full(date=date, exit_price=float(exit_info["exit_price"]), reason=str(exit_info["exit_reason"]), master=master)

    def on_day_close(self, df: pd.DataFrame, i: int, date: str, o: float, h: float, l: float, c: float, master: Any, next_ts: Optional[pd.Timestamp] = None) -> None:
        """EOD accounting and T-close signal evaluation (schedule next-open orders)."""

        # accrue short interest EOD (basis notional)
        if self.pos_side == Side.SHORT:
            master.accrue_daily_short_interest(self.trader_id, self._short_notional_basis(), annual_rate=float(self.cfg.annual_short_interest_rate))
            self.short_hold_days += 1

        # schedule emergency stop ES3 (close-to-close) and TS.B
        self._maybe_schedule_emergency_stop_c2c(df, i)
        self._maybe_schedule_ts_b_exit(df, i)

        # annual unit sizing rebase at year boundary (for next year's sizing)
        if next_ts is not None and pd.Timestamp(next_ts).year != pd.Timestamp(df.index[i]).year:
            nav_tmp = master.nav(self.trader_id, self._holding_value(c), self._short_liability(c))
            self.unit_capital_base = float(nav_tmp)

        # generate next-day entry/pyramiding
        if i >= len(df) - 1:
            return

        atr10_today = df.get("atr10", pd.Series(index=df.index)).iloc[i]
        atr_ref = float(atr10_today) if not pd.isna(atr10_today) else 0.0

        # (a) entry
        if self.pos_side == Side.FLAT and self.pending_entry_order is None:
            d_long = self.entry_rule_long.evaluate(df, i, self.ctx)
            d_short = self.entry_rule_short.evaluate(df, i, self.ctx)

            sig_long = d_long.side if d_long.side == Side.LONG else Side.FLAT
            sig_short = d_short.side if d_short.side == Side.SHORT else Side.FLAT

            if not self._mode_allows(Side.LONG):
                sig_long = Side.FLAT
            if not self._mode_allows(Side.SHORT):
                sig_short = Side.FLAT

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
                if not self._filter_pl_allows(sig, bypass=decision_bypass_pl):
                    sig = Side.FLAT
                if sig != Side.FLAT and not self._filter_cycle_allows(df, i, sig):
                    sig = Side.FLAT
                if sig != Side.FLAT and not self._filter_market_cycle_allows(df, i, sig):
                    sig = Side.FLAT

            if sig != Side.FLAT:
                unit_shares = self._compute_unit_shares(M=self.unit_capital_base, atr10=atr_ref)
                if self.cfg.min_position_mode and self.consecutive_losses >= int(self.cfg.minpos_trigger_consecutive_losses):
                    unit_shares = max(1, int(unit_shares * float(self.cfg.minpos_entry_factor)))
                    self.minpos_first_pyramid_pending = True
                else:
                    self.minpos_first_pyramid_pending = False

                if unit_shares > 0:
                    if sig == Side.SHORT and (float(c) * int(unit_shares) > float(self.cfg.short_notional_limit)):
                        pass
                    else:
                        self.pending_entry_order = Order(
                            symbol=self.symbol,
                            side=sig,
                            unit_shares=int(unit_shares),
                            atr_ref=float(atr_ref),
                            signal_date=date,
                            reason=str(decision_reason) if decision_reason else "ENTRY",
                        )

        # (b) pyramiding
        if (
            self.pos_side != Side.FLAT
            and self.pending_entry_order is None
            and self.pending_exit_reason is None
            and self.pos_units < int(self.cfg.max_units_per_symbol)
            and master.can_add_unit(self.trader_id)
        ):
            if self.cfg.pyramiding_type != PyramidingType.OFF:
                if self.pos_side == Side.LONG:
                    self._pyramid_long(df, i, c, atr_ref, date, master)
                elif self.pos_side == Side.SHORT and self._mode_allows(Side.SHORT):
                    self._pyramid_short(df, i, c, atr_ref, date, master)

    # -------------------- snapshot helpers --------------------
    def eod_snapshot(self, date: str, close_price: float, master: Any) -> Dict[str, Any]:
        holding_value = self._holding_value(close_price)
        short_liab = self._short_liability(close_price)
        nav = master.nav(self.trader_id, holding_value, short_liab)
        return {
            "date": date,
            "nav": nav,
            "cash_cma": master.account(self.trader_id).cash_cma,
            "cash_free": master.account(self.trader_id).cash_free,
            "cash_locked": master.account(self.trader_id).cash_locked,
            "holding_value": holding_value,
            "short_liability": short_liab,
            "pos_side": int(self.pos_side.value),
            "pos_units": int(self.pos_units),
            "pos_shares": int(self.pos_shares),
            "pos_avg_price": float(self.pos_avg_price),
            "short_basis_price": float(self.short_basis_price),
            "short_notional_basis": float(self._short_notional_basis()),
            "accrued_interest": float(master.account(self.trader_id).accrued_interest),
            "unit_capital_base": float(self.unit_capital_base),
        }
