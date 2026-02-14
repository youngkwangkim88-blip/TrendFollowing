from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Account:
    cash_cma: float
    cash_free: float = 0.0
    cash_locked: float = 0.0

    accrued_interest: float = 0.0
    last_interest_month: Optional[int] = None

    def nav(self, holding_value: float, short_liability: float) -> float:
        return float(self.cash_cma + self.cash_free + self.cash_locked + holding_value - short_liability)

    def ensure_trading_cash(self) -> None:
        # Move all CMA cash into trading free cash.
        if self.cash_cma != 0.0:
            self.cash_free += self.cash_cma
            self.cash_cma = 0.0

    def park_to_cma_if_flat(self) -> None:
        # Move free cash back to CMA (toy model).
        if self.cash_free != 0.0:
            self.cash_cma += self.cash_free
            self.cash_free = 0.0

    def apply_monthly_interest_if_first_trading_day(self, is_first_trading_day_of_month: bool) -> float:
        # Deduct accrued interest on the first trading day of the month.
        if not is_first_trading_day_of_month:
            return 0.0
        deducted = float(self.accrued_interest)
        if deducted != 0.0:
            # Deduct regardless of where the cash is parked (free vs CMA).
            # (This keeps NAV consistent even if we "park" cash back into CMA.)
            if self.cash_free >= deducted:
                self.cash_free -= deducted
            else:
                remain = deducted - self.cash_free
                self.cash_free = 0.0
                self.cash_cma -= remain
        self.accrued_interest = 0.0
        return deducted

    def accrue_daily_short_interest(self, short_notional_basis: float, annual_rate: float = 0.045) -> float:
        # Daily accrual (365-day convention); deducted monthly.
        daily = float(short_notional_basis) * (annual_rate / 365.0)
        self.accrued_interest += daily
        return daily

    # ---- simplified cashflows (no T+2) ----
    def long_buy(self, price: float, shares: int) -> None:
        self.cash_free -= float(price) * int(shares)  # buy fee 0%

    def long_sell(self, price: float, shares: int, sell_cost_rate: float = 0.003) -> None:
        gross = float(price) * int(shares)
        self.cash_free += gross * (1.0 - sell_cost_rate)

    def short_sell(self, price: float, shares: int, sell_cost_rate: float = 0.003) -> None:
        gross = float(price) * int(shares)
        self.cash_locked += gross * (1.0 - sell_cost_rate)

    def short_cover(self, price: float, shares: int) -> None:
        self.cash_locked -= float(price) * int(shares)  # buy fee 0%

    def release_locked(self) -> None:
        if self.cash_locked != 0.0:
            self.cash_free += self.cash_locked
            self.cash_locked = 0.0
