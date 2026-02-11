from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any


@dataclass
class AccountSnapshot:
    asof: str
    nav: float
    cash_available: float
    positions: Dict[str, Any]  # symbol -> position fields


class Broker(Protocol):
    def get_account_snapshot(self) -> AccountSnapshot: ...
    def get_orderable_qty(self, symbol: str, side: str) -> int: ...
    def place_market_order(self, symbol: str, side: str, qty: int) -> str: ...
    def get_order_status(self, order_id: str) -> Dict[str, Any]: ...


class KISBrokerStub:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError("Implement using KIS API endpoints.")

    def get_orderable_qty(self, symbol: str, side: str) -> int:
        raise NotImplementedError("Implement using KIS API endpoints (매수가능수량/매도가능수량/대주가능수량).")

    def place_market_order(self, symbol: str, side: str, qty: int) -> str:
        raise NotImplementedError("Implement using KIS API endpoints (시장가 주문).")

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Implement using KIS API endpoints (체결/미체결 조회).")
