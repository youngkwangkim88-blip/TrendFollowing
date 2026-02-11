#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from pykrx import stock

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    tickers: List[str]


def _today_seoul() -> date:
    if ZoneInfo is None:
        return datetime.now().date()
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def find_latest_trading_day(
    anchor_ticker: str = "005930",
    lookback_days: int = 14,
) -> str:
    """
    Find the most recent trading day by probing a known liquid ticker.
    Returns YYYYMMDD string.
    """
    base = _today_seoul()
    last_err: Optional[Exception] = None
    for i in range(lookback_days):
        d = base - timedelta(days=i)
        ds = _yyyymmdd(d)
        try:
            df = stock.get_market_ohlcv(ds, ds, anchor_ticker)
            if df is not None and not df.empty:
                return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not find a recent trading day within {lookback_days} days. Last error: {last_err}")


def get_universe_krx100(asof: str) -> UniverseConfig:
    """
    Get constituents of KRX 100 index via pykrx index list + name match.
    We avoid hardcoding the index code by searching index names.

    pykrx supports index markets KRX/KOSPI/KOSDAQ. :contentReference[oaicite:2]{index=2}
    """
    # Try date-aware call first; fall back if version differs
    try:
        idx_list = stock.get_index_ticker_list(asof, market="KRX")
    except TypeError:
        idx_list = stock.get_index_ticker_list(market="KRX")

    # Find the index whose name looks like "KRX 100"
    target_code: Optional[str] = None
    target_name: Optional[str] = None
    for code in idx_list:
        nm = stock.get_index_ticker_name(code)
        s = str(nm).strip()
        if ("KRX" in s) and ("100" in s) and ("KRX 100" in s or "KRX100" in s or s.replace(" ", "") == "KRX100"):
            target_code = code
            target_name = s
            break

    if target_code is None:
        # If we couldn't identify, print candidates for debugging
        sample = [(c, stock.get_index_ticker_name(c)) for c in idx_list[:30]]
        raise RuntimeError(
            "Could not locate 'KRX 100' index code from pykrx index list. "
            f"Sample index entries: {sample}"
        )

    # Constituents
    # get_index_portfolio_deposit_file(index_code) returns list of tickers. :contentReference[oaicite:3]{index=3}
    tickers = stock.get_index_portfolio_deposit_file(target_code)
    return UniverseConfig(name=f"KRX100({target_name}:{target_code})", tickers=[str(t).zfill(6) for t in tickers])


def get_universe_kospi_mcap100(asof: str) -> UniverseConfig:
    """
    Get top-100 market-cap tickers in KOSPI as-of `asof` (YYYYMMDD).
    """
    # get_market_cap(date, market=...) is commonly used for per-date market caps.
    cap = stock.get_market_cap(asof, market="KOSPI")
    if cap is None or cap.empty:
        raise RuntimeError(f"get_market_cap returned empty for {asof}. Try another date.")

    # Sort by market cap desc
    if "시가총액" not in cap.columns:
        raise RuntimeError(f"Unexpected market cap columns: {list(cap.columns)}")

    cap = cap.sort_values("시가총액", ascending=False)
    tickers = [str(t).zfill(6) for t in cap.index[:100].tolist()]
    return UniverseConfig(name=f"KOSPI_MCAP100(asof={asof})", tickers=tickers)


def _estimate_start_date_yyyymmdd(end_yyyymmdd: str, bars: int) -> str:
    """
    We need ~5000 trading days. Use a calendar-day buffer (~25 years) then tail(bars).
    """
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()

    # 5000 trading days ≈ 20 years; add buffer
    buffer_years = max(10, int(bars / 252) + 6)  # e.g. 5000 -> ~25~26y
    start_dt = end_dt - timedelta(days=int(buffer_years * 365.25))
    return _yyyymmdd(start_dt)


def fetch_ohlcv_adjusted(
    ticker: str,
    name: str,
    start: str,
    end: str,
    bars: int,
) -> pd.DataFrame:
    """
    Fetch adjusted OHLCV using pykrx get_market_ohlcv.
    By default it returns adjusted prices based on the last requested day; adjusted=False disables it. :contentReference[oaicite:4]{index=4}
    """
    df = stock.get_market_ohlcv(start, end, ticker)
    if df is None or df.empty:
        raise RuntimeError(f"Empty OHLCV for {ticker} ({name}) in {start}~{end}")

    # Expected columns are Korean: 시가/고가/저가/종가/거래량 (+ maybe 거래대금/등락률)
    needed = ["시가", "고가", "저가", "종가", "거래량"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"OHLCV missing columns {missing} for {ticker}. Got: {list(df.columns)}")

    out = df[needed].copy()
    out = out.rename(columns={"시가": "O", "고가": "H", "저가": "L", "종가": "C", "거래량": "V"})

    # Keep last N bars
    if len(out) > bars:
        out = out.tail(bars)

    out = out.reset_index()  # index name is usually '날짜' -> becomes a column
    if "날짜" not in out.columns:
        # Some versions might name the index differently
        out = out.rename(columns={out.columns[0]: "날짜"})

    out["날짜"] = pd.to_datetime(out["날짜"]).dt.strftime("%Y-%m-%d")
    out.insert(0, "ticker", str(ticker).zfill(6))
    out.insert(0, "이름", name)

    # Reorder exactly as requested
    out = out[["이름", "날짜", "ticker", "O", "H", "L", "C", "V"]]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download adjusted daily OHLCV for a KRX universe and save a panel CSV."
    )
    p.add_argument(
        "--universe",
        type=str,
        default="kospi_mcap100",
        choices=["krx100", "kospi_mcap100"],
        help="Universe selection: 'krx100' (KRX 100 index constituents) or 'kospi_mcap100' (top-100 market cap KOSPI).",
    )
    p.add_argument("--bars", type=int, default=5000, help="Number of daily bars to keep per ticker (tail).")
    p.add_argument("--out", type=str, default=None, help="Output CSV path. Default: data/<universe>_adj_<bars>.csv")
    p.add_argument("--sleep", type=float, default=0.3, help="Sleep seconds between ticker requests (rate-limit safety).")
    p.add_argument("--max-tickers", type=int, default=None, help="Limit tickers for quick testing.")
    p.add_argument("--end", type=str, default=None, help="End date YYYYMMDD. Default: latest trading day.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    end = args.end or find_latest_trading_day()
    start = _estimate_start_date_yyyymmdd(end, args.bars)

    if args.universe == "krx100":
        uni = get_universe_krx100(end)
    else:
        uni = get_universe_kospi_mcap100(end)

    tickers = uni.tickers
    if args.max_tickers is not None:
        tickers = tickers[: int(args.max_tickers)]

    out_path = Path(args.out) if args.out else Path("data") / f"{args.universe}_adj_{args.bars}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Universe: {uni.name}")
    print(f"[INFO] Date range request: {start} ~ {end} (then tail {args.bars} bars)")
    print(f"[INFO] Tickers: {len(tickers)}")
    print(f"[INFO] Output: {out_path}")

    rows: List[pd.DataFrame] = []
    failures: List[str] = []

    for i, tkr in enumerate(tickers, start=1):
        try:
            name = stock.get_market_ticker_name(tkr)
            panel = fetch_ohlcv_adjusted(tkr, name, start, end, args.bars)
            rows.append(panel)
            print(f"[{i:03d}/{len(tickers):03d}] OK  {tkr} {name}  rows={len(panel)}")
        except Exception as e:  # noqa: BLE001
            msg = f"{tkr}: {e}"
            failures.append(msg)
            print(f"[{i:03d}/{len(tickers):03d}] FAIL {msg}", file=sys.stderr)

        time.sleep(max(0.0, float(args.sleep)))

    if not rows:
        raise RuntimeError("No data fetched successfully. Check connectivity / pykrx status / rate limits.")

    out_df = pd.concat(rows, ignore_index=True)
    out_df = out_df.sort_values(["ticker", "날짜"]).reset_index(drop=True)

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved {len(out_df):,} rows to {out_path}")

    if failures:
        fail_path = out_path.with_suffix(".failures.txt")
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"[WARN] {len(failures)} tickers failed. Saved: {fail_path}")


if __name__ == "__main__":
    main()
