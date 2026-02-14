from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from invest_v2.core.types import EntryRuleType, PyramidingType, Side, TradeMode, TrailingStopType
from invest_v2.trading.trader import Trader, TraderConfig
from invest_v2.trading.trader_master import TraderMaster, TraderMasterConfig


@dataclass
class BacktestConfig:
    """Legacy single-symbol config.

    This remains for backward compatibility with existing scripts.
    Internally, it is converted to a per-symbol TraderConfig and executed via
    Trader + TraderMaster.
    """

    symbol: str = "005930"
    initial_capital: float = 700_000_000.0
    one_trading_risk: float = 0.01
    max_units_per_symbol: int = 4
    max_units_total: int = 10
    short_notional_limit: float = 570_000_000.0  # backtest assumption

    # Entry
    entry_rule: EntryRuleType = EntryRuleType.A_TURTLE
    entry_rule_long: Optional[EntryRuleType] = None
    entry_rule_short: Optional[EntryRuleType] = None

    # Directional mode
    trade_mode: TradeMode = TradeMode.LONG_SHORT

    # Optional filters
    filter_pl: bool = True
    filter_cycle: bool = True  # NOTE: v2 운영 가정상 상시 적용(ON)
    filter_market_cycle: bool = False

    # Minimum position mode
    min_position_mode: bool = False
    minpos_trigger_consecutive_losses: int = 2
    minpos_entry_factor: float = 0.5
    minpos_first_pyramid_factor: float = 0.5

    # Stop-loss (Initial Stop)
    stop_atr_mult: float = 2.0

    # Trailing Stop
    ts_type: TrailingStopType = TrailingStopType.A_PCT
    ts_activate_gain: float = 0.20
    ts_floor_gain: float = 0.10
    ts_trail_frac: float = 0.10
    ts_box_window: int = 20

    # Even-stop (break-even)
    even_stop_gain: float = 0.10

    # Emergency stop
    emergency_stop_pct: float = 0.05

    # Pyramiding
    pyramiding_type: PyramidingType = PyramidingType.A_PCT
    pyramid_trigger: float = 0.15
    pyramid_box_window: int = 20
    pyramid_cooldown_days: int = 5

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
    """Single-symbol backtester (T-close -> T+1-open execution).

    Implementation note
    -------------------
    The v3 refactor introduces Trader/TraderMaster. All position state lives in
    Trader, and the engine becomes a thin event loop.
    """

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfig):
        self.df = df.copy()
        self.cfg = cfg

        # Enforce v2 operating assumption: ticker cycle filter always ON.
        self.cfg.filter_cycle = True

        self.master = TraderMaster(TraderMasterConfig(max_units_total=int(cfg.max_units_total)))

        trader_cfg = TraderConfig(
            trader_id=f"TRADER_{cfg.symbol}",
            symbol=str(cfg.symbol),
            initial_capital=float(cfg.initial_capital),
            one_trading_risk=float(cfg.one_trading_risk),
            max_units_per_symbol=int(cfg.max_units_per_symbol),
            short_notional_limit=float(cfg.short_notional_limit),
            entry_rule=cfg.entry_rule,
            entry_rule_long=cfg.entry_rule_long,
            entry_rule_short=cfg.entry_rule_short,
            trade_mode=cfg.trade_mode,
            filter_pl=bool(cfg.filter_pl),
            filter_cycle=True,
            filter_market_cycle=bool(cfg.filter_market_cycle),
            min_position_mode=bool(cfg.min_position_mode),
            minpos_trigger_consecutive_losses=int(cfg.minpos_trigger_consecutive_losses),
            minpos_entry_factor=float(cfg.minpos_entry_factor),
            minpos_first_pyramid_factor=float(cfg.minpos_first_pyramid_factor),
            stop_atr_mult=float(cfg.stop_atr_mult),
            ts_type=cfg.ts_type,
            ts_activate_gain=float(cfg.ts_activate_gain),
            ts_floor_gain=float(cfg.ts_floor_gain),
            ts_trail_frac=float(cfg.ts_trail_frac),
            ts_box_window=int(cfg.ts_box_window),
            even_stop_gain=float(cfg.even_stop_gain),
            emergency_stop_pct=float(cfg.emergency_stop_pct),
            pyramiding_type=cfg.pyramiding_type,
            pyramid_trigger=float(cfg.pyramid_trigger),
            pyramid_box_window=int(cfg.pyramid_box_window),
            pyramid_cooldown_days=int(cfg.pyramid_cooldown_days),
            sell_cost_rate=float(cfg.sell_cost_rate),
            annual_short_interest_rate=float(cfg.annual_short_interest_rate),
            short_max_hold_days=int(cfg.short_max_hold_days),
        )

        self.trader = Trader(trader_cfg)
        self.master.register_trader(self.trader, trader_id=self.trader.trader_id, symbol=str(cfg.symbol), initial_capital=float(cfg.initial_capital))

        self.snapshots = []

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
                raise ValueError(
                    f"Invalid OHLC (<=0) at {date}: O={o},H={h},L={l},C={c}. Please sanitize your data."
                )

            # monthly interest (first trading day of month)
            is_first_trading_day = True if i == 0 else (df.index[i - 1].month != ts.month)
            self.master.apply_monthly_interest_if_first_trading_day(self.trader.trader_id, is_first_trading_day_of_month=is_first_trading_day)

            # --- open phase ---
            self.trader.on_day_open(df=df, i=i, date=date, o=o, h=h, l=l, c=c, master=self.master)

            # --- intraday phase (stops) ---
            self.trader.on_day_intraday(df=df, i=i, date=date, o=o, h=h, l=l, c=c, master=self.master)

            # --- close phase (signals schedule) ---
            next_ts = df.index[i + 1] if i < len(df) - 1 else None
            self.trader.on_day_close(df=df, i=i, date=date, o=o, h=h, l=l, c=c, master=self.master, next_ts=next_ts)

            # park to CMA if flat and no pending order
            if self.trader.pos_side == Side.FLAT and self.trader.pending_entry_order is None:
                self.master.park_to_cma_if_flat(self.trader.trader_id)

            # snapshot EOD NAV
            self.snapshots.append(self.trader.eod_snapshot(date=date, close_price=c, master=self.master))

        equity_df = pd.DataFrame(self.snapshots).set_index("date")
        trades_df = self.trader.trades_df()
        fills_df = self.trader.fills_df()
        summary = self._summarize(equity_df, trades_df)
        return BacktestResult(equity_curve=equity_df, trades=trades_df, fills=fills_df, summary=summary)

    @staticmethod
    def _summarize(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
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
