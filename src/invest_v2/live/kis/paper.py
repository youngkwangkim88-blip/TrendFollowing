from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from invest_v2.live.broker_kis_stub import AccountSnapshot, Broker

from .client import KISClient
from .settings import KISSettings
from ..journal import OrderJournal


@dataclass
class PaperEndpoints:
    """Endpoint paths (configurable)."""

    order_cash: str = "/uapi/domestic-stock/v1/trading/order-cash"
    inquire_balance: str = "/uapi/domestic-stock/v1/trading/inquire-balance"
    inquire_psbl: str = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"


class KISPaperBroker(Broker):
    """KIS paper-trading broker implementation.

    Note: Short (대주) endpoints are intentionally left as TODO because they vary by
    account eligibility and TR-ID/endpoint revisions.
    """

    def __init__(
        self,
        settings: KISSettings,
        *,
        endpoints: Optional[PaperEndpoints] = None,
        journal: Optional[OrderJournal] = None,
    ):
        self.settings = settings
        self.client = KISClient(settings)
        self.endpoints = endpoints or PaperEndpoints()
        self.journal = journal or OrderJournal()

        if not settings.tr_id_buy or not settings.tr_id_sell:
            raise RuntimeError(
                "Missing TR-ID env vars. Set KIS_TR_ID_BUY and KIS_TR_ID_SELL (paper/real differ)."
            )
        if not settings.tr_id_inquire_balance:
            raise RuntimeError("Missing env var: KIS_TR_ID_INQUIRE_BALANCE")

    def get_account_snapshot(self) -> AccountSnapshot:
        params = {
            "CANO": self.settings.cano,
            "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
            # The remaining params vary by API version; keep minimal and let
            # the user extend via monkeypatch if needed.
        }
        resp = self.client.request(
            "GET",
            self.endpoints.inquire_balance,
            tr_id=self.settings.tr_id_inquire_balance,
            params=params,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"KIS inquire-balance failed: {resp.status_code} {resp.text}")

        js = resp.json or {}
        # Different responses expose different keys. We keep a tolerant parser.
        # Users can refine mapping after verifying their paper account response.
        output1 = js.get("output1") or js.get("output") or []
        output2 = js.get("output2") or {}

        # cash_available: prefer 'ord_psbl_cash' if present
        cash_available = float(output2.get("ord_psbl_cash", 0) or output2.get("dnca_tot_amt", 0) or 0)
        nav = float(output2.get("tot_evlu_amt", 0) or output2.get("tot_asst_amt", 0) or 0)

        positions: Dict[str, Any] = {}
        if isinstance(output1, list):
            for p in output1:
                sym = str(p.get("pdno") or p.get("prdt_no") or p.get("symbol") or "").zfill(6)
                if not sym:
                    continue
                positions[sym] = p

        return AccountSnapshot(asof=str(js.get("rt_cd", "")), nav=nav, cash_available=cash_available, positions=positions)

    def get_orderable_qty(self, symbol: str, side: str) -> int:
        # Optional API; not mandatory for market orders if you already computed qty.
        if not self.settings.tr_id_inquire_psbl:
            raise RuntimeError("Missing env var: KIS_TR_ID_INQUIRE_PSBL")

        symbol = str(symbol).zfill(6)
        params = {
            "CANO": self.settings.cano,
            "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
            "PDNO": symbol,
            # price 0 for market
            "ORD_UNPR": "0",
            # Order division: '01' often means market; confirm in your portal doc.
            "ORD_DVSN": "01",
        }
        resp = self.client.request("GET", self.endpoints.inquire_psbl, tr_id=self.settings.tr_id_inquire_psbl, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"KIS inquire-psbl-order failed: {resp.status_code} {resp.text}")
        js = resp.json or {}
        out = js.get("output") or js.get("output1") or {}
        key = "ord_psbl_qty" if side.lower().startswith("b") else "sell_psbl_qty"
        try:
            return int(float(out.get(key, 0) or 0))
        except Exception:
            return 0

    def place_market_order(self, symbol: str, side: str, qty: int) -> str:
        symbol = str(symbol).zfill(6)
        side_l = side.lower()
        if qty <= 0:
            raise ValueError("qty must be positive")

        tr_id = self.settings.tr_id_buy if side_l in {"buy", "long"} else self.settings.tr_id_sell
        order = {
            "CANO": self.settings.cano,
            "ACNT_PRDT_CD": self.settings.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_DVSN": "01",  # market
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0",  # market => 0
        }

        idem = f"{symbol}:{side_l}:{qty}:{order['ORD_DVSN']}:{order['ORD_UNPR']}"
        existing = self.journal.get(idem)
        if existing and existing.response:
            # Return previously recorded order id
            return str(existing.response.get("order_id", existing.response.get("output", "")))

        self.journal.put_request(idem, order)
        resp = self.client.request("POST", self.endpoints.order_cash, tr_id=tr_id, body=order, use_hashkey=True)
        if resp.status_code != 200:
            self.journal.put_response(idem, {"error": resp.text, "status_code": resp.status_code})
            raise RuntimeError(f"KIS order failed: {resp.status_code} {resp.text}")

        js = resp.json or {}
        # Typical: output.ord_no
        order_id = (
            (js.get("output") or {}).get("ord_no")
            or (js.get("output") or {}).get("ODNO")
            or js.get("ord_no")
            or ""
        )
        self.journal.put_response(idem, {"order_id": order_id, "raw": js})
        return str(order_id)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        # TODO: implement using "주식일별주문체결조회" or similar.
        raise NotImplementedError("Implement using KIS '주식일별주문체결조회' endpoint.")
