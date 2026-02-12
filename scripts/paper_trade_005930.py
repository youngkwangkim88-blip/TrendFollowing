#!/usr/bin/env python3
"""Minimal KIS paper order smoke test.

Usage:
  export KIS_BASE_URL=https://openapivts.koreainvestment.com:29443
  export KIS_APP_KEY=...
  export KIS_APP_SECRET=...
  export KIS_CANO=XXXXXXXX
  export KIS_ACNT_PRDT_CD=01
  export KIS_TR_ID_BUY=...     # from KIS portal
  export KIS_TR_ID_SELL=...
  export KIS_TR_ID_INQUIRE_BALANCE=...

  python scripts/paper_trade_005930.py --buy 1
"""

from __future__ import annotations

import argparse
import json

from invest_v2.live.kis.settings import KISSettings
from invest_v2.live.kis.paper import KISPaperBroker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="005930")
    p.add_argument("--buy", type=int, default=0)
    p.add_argument("--sell", type=int, default=0)
    p.add_argument("--snapshot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = KISSettings.from_env()
    broker = KISPaperBroker(settings)

    if args.snapshot:
        snap = broker.get_account_snapshot()
        print(json.dumps(snap.__dict__, ensure_ascii=False, indent=2))

    if args.buy:
        oid = broker.place_market_order(args.symbol, "buy", int(args.buy))
        print(f"BUY order_id={oid}")
    if args.sell:
        oid = broker.place_market_order(args.symbol, "sell", int(args.sell))
        print(f"SELL order_id={oid}")


if __name__ == "__main__":
    main()
