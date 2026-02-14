from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from invest_v2.core.types import Side
from invest_v2.trading.trader import Trader, TraderConfig
from invest_v2.trading.trader_master import TraderMaster, TraderMasterConfig


@dataclass
class PortfolioBacktestResult:
    master_equity_curve: pd.DataFrame
    trader_equity_curves: Dict[str, pd.DataFrame]
    trades: pd.DataFrame
    fills: pd.DataFrame
    summary: Dict[str, Any]


class TraderMasterBacktester:
    """Multi-trader (portfolio) backtester.

    This is the forward-looking execution layer:
      - Each Trader owns position state for its symbol.
      - TraderMaster owns per-trader sleeve accounts and portfolio constraints.

    Data model
    ----------
    Provide `symbol_dfs` as {symbol: dataframe}.
    Each dataframe must contain daily OHLC and required indicators.

    Notes
    -----
    - Currently assumes daily bar calendars are aligned (KRX equities mostly are).
    - For missing bars, that trader is skipped on that date.
    """

    def __init__(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        trader_cfgs: List[TraderConfig],
        master_cfg: Optional[TraderMasterConfig] = None,
    ):
        self.symbol_dfs = {str(k): v.copy() for k, v in symbol_dfs.items()}
        self.trader_cfgs = list(trader_cfgs)

        self.master = TraderMaster(master_cfg or TraderMasterConfig())
        self.traders: Dict[str, Trader] = {}

        for cfg in self.trader_cfgs:
            t = Trader(cfg)
            self.traders[t.trader_id] = t
            self.master.register_trader(t, trader_id=t.trader_id, symbol=t.symbol, initial_capital=float(cfg.initial_capital))

        self._snapshots_master: List[Dict[str, Any]] = []
        self._snapshots_trader: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in self.traders}

    # -------------------- arbitration --------------------
    def _arbitrate_pending_entries(self) -> None:
        """Resolve conflicts when multiple traders want to open the same symbol."""

        # symbol -> list of (priority, trader_id)
        wants: Dict[str, List[Tuple[int, str]]] = {}
        for tid, t in self.traders.items():
            if t.pos_side != Side.FLAT:
                continue
            if t.pending_entry_order is None:
                continue
            wants.setdefault(t.symbol, []).append((int(t.cfg.priority), tid))

        for sym, lst in wants.items():
            if len(lst) <= 1:
                continue
            # pick unique max priority
            lst_sorted = sorted(lst, key=lambda x: x[0], reverse=True)
            top_pri = lst_sorted[0][0]
            top = [tid for pri, tid in lst_sorted if pri == top_pri]
            if len(top) != 1:
                # ambiguous: cancel all
                for _, tid in lst:
                    self.traders[tid].pending_entry_order = None
                continue
            winner = top[0]
            for _, tid in lst:
                if tid != winner:
                    self.traders[tid].pending_entry_order = None

    # -------------------- run loop --------------------
    def run(self) -> PortfolioBacktestResult:
        # union calendar
        idx = pd.Index([])
        for df in self.symbol_dfs.values():
            idx = idx.union(df.index)
        idx = idx.sort_values()

        # align dataframes
        aligned: Dict[str, pd.DataFrame] = {}
        for sym, df in self.symbol_dfs.items():
            aligned[sym] = df.reindex(idx)

        prev_month: Dict[str, Optional[int]] = {tid: None for tid in self.traders}

        for k, ts in enumerate(idx):
            date = str(ts.date())

            # arbitration is needed because entry orders are scheduled at T close
            # and executed at this open.
            self._arbitrate_pending_entries()

            # --- per trader: process day ---
            for tid, trader in self.traders.items():
                sym = trader.symbol
                df = aligned.get(sym)
                if df is None:
                    continue

                row = df.iloc[k]
                if pd.isna(row.get("open", pd.NA)):
                    continue

                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])

                # monthly interest (per sleeve)
                m = int(ts.month)
                is_first = prev_month[tid] is None or prev_month[tid] != m
                prev_month[tid] = m
                self.master.apply_monthly_interest_if_first_trading_day(tid, is_first_trading_day_of_month=is_first)

                # open/intraday/close
                trader.on_day_open(df=df, i=k, date=date, o=o, h=h, l=l, c=c, master=self.master)
                trader.on_day_intraday(df=df, i=k, date=date, o=o, h=h, l=l, c=c, master=self.master)
                next_ts = idx[k + 1] if k < len(idx) - 1 else None
                trader.on_day_close(df=df, i=k, date=date, o=o, h=h, l=l, c=c, master=self.master, next_ts=next_ts)

                if trader.pos_side == Side.FLAT and trader.pending_entry_order is None:
                    self.master.park_to_cma_if_flat(tid)

                self._snapshots_trader[tid].append(trader.eod_snapshot(date=date, close_price=c, master=self.master))

            # master snapshot (sum NAV)
            holding_values = {tid: float(self.traders[tid]._holding_value(float(aligned[self.traders[tid].symbol].iloc[k]["close"]))) if self.traders[tid].symbol in aligned and not pd.isna(aligned[self.traders[tid].symbol].iloc[k].get("close", pd.NA)) else 0.0 for tid in self.traders}
            short_liabilities = {tid: float(self.traders[tid]._short_liability(float(aligned[self.traders[tid].symbol].iloc[k]["close"]))) if self.traders[tid].symbol in aligned and not pd.isna(aligned[self.traders[tid].symbol].iloc[k].get("close", pd.NA)) else 0.0 for tid in self.traders}

            total_nav = self.master.total_nav(holding_values, short_liabilities)
            self._snapshots_master.append({"date": date, "nav": total_nav})

        # build outputs
        master_eq = pd.DataFrame(self._snapshots_master).set_index("date")
        trader_eq = {tid: pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame() for tid, rows in self._snapshots_trader.items()}

        trades = pd.concat([t.trades_df() for t in self.traders.values()], ignore_index=True) if len(self.traders) else pd.DataFrame()
        fills = pd.concat([t.fills_df() for t in self.traders.values()], ignore_index=True) if len(self.traders) else pd.DataFrame()

        summary = {
            "master_start_nav": float(master_eq["nav"].iloc[0]) if len(master_eq) else 0.0,
            "master_end_nav": float(master_eq["nav"].iloc[-1]) if len(master_eq) else 0.0,
            "num_traders": int(len(self.traders)),
            "num_trades": int(len(trades)) if trades is not None else 0,
        }

        return PortfolioBacktestResult(
            master_equity_curve=master_eq,
            trader_equity_curves=trader_eq,
            trades=trades,
            fills=fills,
            summary=summary,
        )
