from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from invest_v2.backtest.accounting import Account

from .trader import Trader


@dataclass
class AccountBook:
    """Container for multiple ledgers.

    For now, v2 backtests use a single KRW account that already models
    CMA vs trading cash.
    """

    krw: Account


class TraderMaster:
    """Owns account(s) and provides per-symbol Traders.

    In this repo, the execution logic (signals, stops, order generation) lives
    in the backtest engine or live runner. TraderMaster is intentionally thin:
    it is the place where multi-asset / multi-currency account reconciliation
    will be added.
    """

    def __init__(self, *, initial_capital_krw: float):
        self.book = AccountBook(krw=Account(cash_cma=float(initial_capital_krw)))
        self._traders: Dict[str, Trader] = {}

    @property
    def account(self) -> Account:
        return self.book.krw

    def get_trader(self, symbol: str) -> Trader:
        sym = str(symbol)
        if sym not in self._traders:
            self._traders[sym] = Trader(symbol=sym)
        return self._traders[sym]

    def how_did_you_trade(self, symbol: Optional[str] = None, max_trades: int = 50) -> str:
        if symbol is not None:
            return self.get_trader(symbol).how_did_you_trade(max_trades=max_trades)

        # all traders
        parts = []
        for sym in sorted(self._traders.keys()):
            parts.append(self._traders[sym].how_did_you_trade(max_trades=max_trades))
        return "\n\n".join(parts)
