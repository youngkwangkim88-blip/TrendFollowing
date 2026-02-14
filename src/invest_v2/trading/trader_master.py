from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from invest_v2.backtest.accounting import Account
from invest_v2.core.types import Side


@dataclass
class TraderMasterConfig:
    """Global constraints / master-level knobs."""

    # Portfolio-wide unit constraint (spec: total 10 units)
    max_units_total: int = 10

    # Symbol exclusivity: if True, only one trader can hold a position per symbol.
    # This matches the legal constraint "동일 종목 롱/숏 동시 보유 금지" and makes
    # multi-trader composition well-defined.
    exclusive_symbol_position: bool = True


class TraderMaster:
    """Master orchestrator.

    Responsibilities
    ---------------
    - Own one or more accounts (currently we model one *sleeve account per trader*).
    - Enforce portfolio-wide constraints (max total units, symbol exclusivity).
    - Provide execution services to traders (cashflows / short notional limit checks are
      partially in TraderConfig; master can reject orders).

    Notes
    -----
    For now, each Trader is allocated an independent Account sleeve.
    This makes capital allocation explicit and avoids hidden coupling.
    """

    def __init__(self, cfg: Optional[TraderMasterConfig] = None):
        self.cfg = cfg or TraderMasterConfig()

        # trader_id -> Account (sleeve)
        self._accounts: Dict[str, Account] = {}

        # symbol -> (trader_id, side)
        self._symbol_owner: Dict[str, Tuple[str, Side]] = {}

        # trader registry (for unit counting and symbol ownership)
        self._traders: Dict[str, object] = {}

    # -------------------- registration --------------------
    def register_trader(self, trader: object, trader_id: str, symbol: str, initial_capital: float) -> None:
        if trader_id in self._accounts:
            raise ValueError(f"Trader already registered: {trader_id}")
        self._accounts[trader_id] = Account(cash_cma=float(initial_capital))
        self._traders[trader_id] = trader
        # No symbol owner until a position is opened.

    def account(self, trader_id: str) -> Account:
        if trader_id not in self._accounts:
            raise KeyError(f"Unknown trader_id: {trader_id}")
        return self._accounts[trader_id]

    # -------------------- global constraints --------------------
    def total_units(self) -> int:
        total = 0
        for t in self._traders.values():
            # Trader is expected to expose `pos_units`.
            total += int(getattr(t, "pos_units", 0) or 0)
        return int(total)

    def can_add_unit(self, trader_id: str) -> bool:
        return (self.total_units() + 1) <= int(self.cfg.max_units_total)

    def can_open_symbol(self, trader_id: str, symbol: str) -> bool:
        if not self.cfg.exclusive_symbol_position:
            return True
        owner = self._symbol_owner.get(str(symbol))
        if owner is None:
            return True
        owner_id, owner_side = owner
        # allow if the current owner is the same trader (pyramiding)
        if owner_id == trader_id:
            return True
        # otherwise deny opening any position in that symbol
        return False

    def on_position_opened(self, trader_id: str, symbol: str, side: Side) -> None:
        if self.cfg.exclusive_symbol_position:
            self._symbol_owner[str(symbol)] = (trader_id, side)

    def on_position_closed(self, trader_id: str, symbol: str) -> None:
        if not self.cfg.exclusive_symbol_position:
            return
        owner = self._symbol_owner.get(str(symbol))
        if owner is None:
            return
        if owner[0] == trader_id:
            self._symbol_owner.pop(str(symbol), None)

    # -------------------- execution helpers --------------------
    def ensure_trading_cash(self, trader_id: str) -> None:
        self.account(trader_id).ensure_trading_cash()

    def park_to_cma_if_flat(self, trader_id: str) -> None:
        self.account(trader_id).park_to_cma_if_flat()

    def apply_monthly_interest_if_first_trading_day(self, trader_id: str, is_first_trading_day_of_month: bool) -> float:
        return self.account(trader_id).apply_monthly_interest_if_first_trading_day(is_first_trading_day_of_month)

    # ---- cashflows (delegated to Account) ----
    def long_buy(self, trader_id: str, price: float, shares: int) -> None:
        self.account(trader_id).long_buy(price, shares)

    def long_sell(self, trader_id: str, price: float, shares: int, sell_cost_rate: float) -> None:
        self.account(trader_id).long_sell(price, shares, sell_cost_rate=sell_cost_rate)

    def short_sell(self, trader_id: str, price: float, shares: int, sell_cost_rate: float) -> None:
        self.account(trader_id).short_sell(price, shares, sell_cost_rate=sell_cost_rate)

    def short_cover(self, trader_id: str, price: float, shares: int) -> None:
        self.account(trader_id).short_cover(price, shares)

    def release_locked(self, trader_id: str) -> None:
        self.account(trader_id).release_locked()

    def accrue_daily_short_interest(self, trader_id: str, short_notional_basis: float, annual_rate: float) -> float:
        return self.account(trader_id).accrue_daily_short_interest(short_notional_basis, annual_rate=annual_rate)

    # -------------------- account metrics --------------------
    def nav(self, trader_id: str, holding_value: float, short_liability: float) -> float:
        return self.account(trader_id).nav(holding_value, short_liability)

    def total_nav(self, holding_values: Dict[str, float], short_liabilities: Dict[str, float]) -> float:
        total = 0.0
        for tid, acc in self._accounts.items():
            hv = float(holding_values.get(tid, 0.0))
            sl = float(short_liabilities.get(tid, 0.0))
            total += acc.nav(hv, sl)
        return float(total)
