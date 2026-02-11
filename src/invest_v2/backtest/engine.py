from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd

from invest_v2.core.types import EntryRuleType, Side, Order, Trade
from invest_v2.strategy.entry_rules import build_entry_rule, EntryContext
from invest_v2.utils.krx_ticks import tick_down, tick_up
from invest_v2.backtest.accounting import Account


@dataclass
class BacktestConfig:
    symbol: str = "005930"
    initial_capital: float = 700_000_000.0
    one_trading_risk: float = 0.01
    max_units_per_symbol: int = 4
    max_units_total: int = 10
    short_notional_limit: float = 570_000_000.0  # backtest assumption
    entry_rule: EntryRuleType = EntryRuleType.A_20_PL
    sell_cost_rate: float = 0.003
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
        self.entry_rule = build_entry_rule(cfg.entry_rule)
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
        self.pos_ts_active: bool = False

        self.short_hold_days: int = 0
        self.short_basis_price: float = 0.0  # weighted avg of short sell prices
        self.short_open_date: Optional[str] = None

        self.pending_order: Optional[Order] = None
        self.active_trade: Optional[Trade] = None

        self.snapshots: List[Dict[str, Any]] = []
        self.trade_logs: List[Dict[str, Any]] = []

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

    def _update_trailing_extrema(self, high: float, low: float) -> None:
        if self.pos_side == Side.LONG:
            self.pos_h_max = max(self.pos_h_max, float(high))
        elif self.pos_side == Side.SHORT:
            if self.pos_l_min == 0.0:
                self.pos_l_min = float(low)
            else:
                self.pos_l_min = min(self.pos_l_min, float(low))

    def _stop_levels(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if self.pos_side == Side.FLAT:
            return None, None, None

        X = float(self.pos_avg_price)
        atr = float(self.pos_atr_ref)

        if self.pos_side == Side.LONG:
            stop_loss = tick_down(X - 2.0 * atr)
            ts_level = None
            if self.pos_h_max >= 1.2 * X:
                self.pos_ts_active = True
            if self.pos_ts_active:
                ts_raw = max(1.1 * X, 0.9 * float(self.pos_h_max))
                ts_level = tick_down(ts_raw)
            effective = stop_loss if ts_level is None else max(stop_loss, ts_level)
            return stop_loss, ts_level, effective

        # SHORT
        stop_loss = tick_up(X + 2.0 * atr)
        ts_level = None
        if self.pos_l_min > 0 and self.pos_l_min <= 0.8 * X:
            self.pos_ts_active = True
        if self.pos_ts_active:
            ts_raw = min(0.9 * X, 1.1 * float(self.pos_l_min))
            ts_level = tick_up(ts_raw)
        effective = stop_loss if ts_level is None else min(stop_loss, ts_level)
        return stop_loss, ts_level, effective

    def _check_stop_trigger_and_exit(self, o: float, h: float, l: float) -> Optional[Dict[str, Any]]:
        _, _, eff = self._stop_levels()
        if eff is None:
            return None
        eff = float(eff)

        if self.pos_side == Side.LONG:
            if float(o) <= eff:
                return {"exit_price": float(o), "exit_reason": "STOP_GAP"}
            if float(o) > eff and float(l) <= eff:
                return {"exit_price": eff, "exit_reason": "STOP_TOUCH"}
            return None

        # SHORT
        if float(o) >= eff:
            return {"exit_price": float(o), "exit_reason": "STOP_GAP"}
        if float(o) < eff and float(h) >= eff:
            return {"exit_price": eff, "exit_reason": "STOP_TOUCH"}
        return None

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

        self.trade_logs.append({
            "symbol": self.active_trade.symbol,
            "side": int(self.active_trade.side.value),
            "entry_date": self.active_trade.entry_date,
            "entry_price": self.active_trade.entry_price,
            "shares": self.active_trade.shares,
            "exit_date": self.active_trade.exit_date,
            "exit_price": self.active_trade.exit_price,
            "realized_pnl": self.active_trade.realized_pnl,
            "exit_reason": self.active_trade.exit_reason,
        })

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
        self.short_hold_days = 0
        self.short_basis_price = 0.0
        self.short_open_date = None
        self.active_trade = None

    def _can_add_unit(self) -> bool:
        return self.pos_units < self.cfg.max_units_per_symbol and self.pos_units < self.cfg.max_units_total

    def _can_open_short_with_limit(self, add_shares: int, sell_price: float) -> bool:
        new_basis_total = self._short_notional_basis() + float(sell_price) * int(add_shares)
        return new_basis_total <= self.cfg.short_notional_limit

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

            # execute pending order at open
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
                            self.pos_side = Side.LONG
                            self.pos_units = 1
                            self.pos_shares = add_shares
                            self.pos_avg_price = fill
                            self.pos_atr_ref = float(order.atr_ref)
                            self.pos_h_max = max(h, fill)
                            self.pos_ts_active = False
                            self.active_trade = Trade(symbol=self.cfg.symbol, side=Side.LONG, entry_date=date, entry_price=fill, shares=add_shares)
                        else:
                            # pyramiding long (reset H_max/ATR/TS per spec)
                            new_total = self.pos_shares + add_shares
                            self.pos_avg_price = (self.pos_avg_price * self.pos_shares + fill * add_shares) / new_total
                            self.pos_shares = new_total
                            self.pos_units += 1
                            self.pos_atr_ref = float(order.atr_ref)
                            self.pos_h_max = max(h, fill)
                            self.pos_ts_active = False
                            if self.active_trade:
                                self.active_trade.shares = self.pos_shares
                    else:
                        # short sell: notional gate
                        if self._can_open_short_with_limit(add_shares, sell_price=fill):
                            self.account.short_sell(fill, add_shares, sell_cost_rate=self.cfg.sell_cost_rate)
                            if self.pos_side == Side.FLAT:
                                self.pos_side = Side.SHORT
                                self.pos_units = 1
                                self.pos_shares = add_shares
                                self.pos_avg_price = fill
                                self.pos_atr_ref = float(order.atr_ref)
                                self.pos_l_min = min(l, fill)
                                self.pos_ts_active = False
                                self.short_hold_days = 0
                                self.short_open_date = date
                                self.short_basis_price = fill
                                self.active_trade = Trade(symbol=self.cfg.symbol, side=Side.SHORT, entry_date=date, entry_price=fill, shares=add_shares)
                            else:
                                new_total = self.pos_shares + add_shares
                                self.pos_avg_price = (self.pos_avg_price * self.pos_shares + fill * add_shares) / new_total
                                self.pos_shares = new_total
                                self.pos_units += 1
                                self.pos_atr_ref = float(order.atr_ref)
                                self.pos_l_min = min(l, fill)
                                self.pos_ts_active = False
                                self.short_basis_price = (self.short_basis_price * (new_total - add_shares) + fill * add_shares) / new_total
                                if self.active_trade:
                                    self.active_trade.shares = self.pos_shares
                        else:
                            # skipped due to notional limit
                            pass

            # short max hold days: exit at open
            if self.pos_side == Side.SHORT and self.short_hold_days >= self.cfg.short_max_hold_days:
                self._exit_position_full(date=date, exit_price=o, reason="SHORT_MAX_HOLD_DAYS")
            else:
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

            # generate next-day orders (entry/pyramiding)
            if i < len(df) - 1:
                atr10_today = df["atr10"].iloc[i]
                atr_ref = float(atr10_today) if not pd.isna(atr10_today) else 0.0

                if self.pos_side == Side.FLAT and self.pending_order is None:
                    sig = self.entry_rule.evaluate(df, i, self.ctx)
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
                                    reason=f"ENTRY_{self.cfg.entry_rule.value}"
                                )

                # pyramiding
                if self.pos_side == Side.LONG and self.pending_order is None and self._can_add_unit():
                    if c >= 1.15 * float(self.pos_avg_price):
                        unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                        if unit_shares > 0:
                            self.pending_order = Order(
                                symbol=self.cfg.symbol,
                                side=Side.LONG,
                                unit_shares=unit_shares,
                                atr_ref=atr_ref,
                                signal_date=date,
                                reason="PYRAMID_LONG"
                            )

                if self.pos_side == Side.SHORT and self.pending_order is None and self._can_add_unit():
                    if c <= 0.85 * float(self.pos_avg_price):
                        unit_shares = self._compute_unit_shares(M=self.cfg.initial_capital, atr10=atr_ref)
                        if unit_shares > 0 and self._can_open_short_with_limit(unit_shares, sell_price=c):
                            self.pending_order = Order(
                                symbol=self.cfg.symbol,
                                side=Side.SHORT,
                                unit_shares=unit_shares,
                                atr_ref=atr_ref,
                                signal_date=date,
                                reason="PYRAMID_SHORT"
                            )

            # park to CMA if flat and no pending order
            if self.pos_side == Side.FLAT and self.pending_order is None:
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
            })

        equity_df = pd.DataFrame(self.snapshots).set_index("date")
        trades_df = pd.DataFrame(self.trade_logs)
        summary = self._summarize(equity_df, trades_df)
        return BacktestResult(equity_curve=equity_df, trades=trades_df, summary=summary)

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
