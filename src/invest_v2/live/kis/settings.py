from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class KISSettings:
    """Runtime configuration for KIS OpenAPI.

    Values are loaded from environment variables to avoid committing secrets.
    """

    app_key: str
    app_secret: str
    base_url: str
    cano: str
    acnt_prdt_cd: str

    # Optional overrides (TR-IDs can differ between paper/real and buy/sell)
    tr_id_buy: str
    tr_id_sell: str
    tr_id_inquire_balance: str
    tr_id_inquire_psbl: str

    @staticmethod
    def from_env(prefix: str = "KIS_") -> "KISSettings":
        def req(name: str) -> str:
            v = os.getenv(prefix + name)
            if not v:
                raise RuntimeError(f"Missing env var: {prefix}{name}")
            return v

        # Common defaults seen in KIS docs/samples; keep overridable.
        # Paper/testbed base is often openapivts.koreainvestment.com:29443.
        base_url = os.getenv(prefix + "BASE_URL") or req("BASE_URL")

        return KISSettings(
            app_key=req("APP_KEY"),
            app_secret=req("APP_SECRET"),
            base_url=base_url.rstrip("/"),
            cano=req("CANO"),
            acnt_prdt_cd=req("ACNT_PRDT_CD"),
            tr_id_buy=os.getenv(prefix + "TR_ID_BUY", ""),
            tr_id_sell=os.getenv(prefix + "TR_ID_SELL", ""),
            tr_id_inquire_balance=os.getenv(prefix + "TR_ID_INQUIRE_BALANCE", ""),
            tr_id_inquire_psbl=os.getenv(prefix + "TR_ID_INQUIRE_PSBL", ""),
        )
