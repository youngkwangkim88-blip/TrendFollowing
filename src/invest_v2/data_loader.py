from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def sanitize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize OHLC data.

    Motivation
    ----------
    Some KRX panel datasets contain 0-valued OHLC (especially O/H/L) while C is valid.
    If left as-is, the backtester can create impossible fills (e.g., exit at 0 KRW),
    distorting PnL/MDD and plots.

    Rules (conservative)
    --------------------
    - Any non-positive values in open/high/low/close are treated as missing.
    - Rows with missing close are dropped.
    - Missing open/high/low are filled with close (same-day) -- no look-ahead.
    - high is forced >= max(open, close); low forced <= min(open, close).
    """
    out = df.copy()

    required = ["open", "high", "low", "close"]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"sanitize_ohlc: missing required column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] <= 0, c] = pd.NA

    # Drop rows without a valid close.
    out = out.dropna(subset=["close"]).copy()

    # Fill missing O/H/L with close (same-day).
    for c in ["open", "high", "low"]:
        out[c] = out[c].fillna(out["close"])

    # Enforce OHLC consistency.
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)

    return out


def load_ohlc_csv(path: str) -> pd.DataFrame:
    """Load a simple single-symbol OHLC CSV.

    Expected columns: date,open,high,low,close,(volume)
    """
    p = Path(path)
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError("CSV must include 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
        df[c] = df[c].astype(float)

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = sanitize_ohlc(df)
    return df


def load_ohlc_auto(csv_path: str, symbol: str) -> pd.DataFrame:
    """Auto-detect CSV schema and return OHLC(V) indexed by datetime.

    Supported schemas
    -----------------
    1) Panel format: 이름,날짜,ticker,O,H,L,C,(V)
    2) Old single-symbol: date,open,high,low,close,(volume)

    Output columns: open, high, low, close, (volume)
    """
    df = pd.read_csv(Path(csv_path), dtype={"ticker": str}, low_memory=False)

    # New panel format: 이름, 날짜, ticker, O,H,L,C,V
    panel_cols = {"날짜", "ticker", "O", "H", "L", "C"}
    if panel_cols.issubset(df.columns):
        sym = str(symbol).zfill(6)
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df = df[df["ticker"] == sym].copy()
        if df.empty:
            raise ValueError(f"No rows found for symbol={sym} in panel CSV: {csv_path}")

        df["date"] = pd.to_datetime(df["날짜"])
        df = df.rename(columns={"O": "open", "H": "high", "L": "low", "C": "close", "V": "volume"})
        keep = ["date", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
        df = df[keep].set_index("date").sort_index()

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = sanitize_ohlc(df)
        return df

    # Old single-symbol format: date,open,high,low,close,(volume)
    old_cols = {"date", "open", "high", "low", "close"}
    if old_cols.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = sanitize_ohlc(df)
        return df

    raise ValueError(
        "Unrecognized CSV schema. Expected either old format (date,open,high,low,close,...) "
        "or panel format (이름,날짜,ticker,O,H,L,C,V). "
        f"Got columns: {list(df.columns)}"
    )


def load_market_ohlc_auto(csv_path: str, ticker: Optional[str] = None) -> pd.DataFrame:
    """Load market index/futures OHLC(V) for market regime filters.

    Supported schemas:
      - Old: date,open,high,low,close,(volume)
      - Close-only: date,close
      - Panel: 이름,날짜,(ticker),O,H,L,C,(V)

    If OHLC is missing, it is synthesized from close (open=high=low=close).
    The result is then sanitized (0/negative treated as missing).
    """
    df = pd.read_csv(Path(csv_path), dtype={"ticker": str}, low_memory=False)

    panel_cols = {"날짜", "O", "H", "L", "C"}
    if panel_cols.issubset(df.columns):
        # Optional ticker selection
        if "ticker" in df.columns:
            if ticker is None:
                uniq = df["ticker"].astype(str).unique()
                if len(uniq) != 1:
                    raise ValueError(
                        f"Market CSV has multiple tickers ({len(uniq)}). Use --market-ticker. Example: {uniq[:5]}"
                    )
                ticker = str(uniq[0])
            df["ticker"] = df["ticker"].astype(str)
            df = df[df["ticker"] == str(ticker)].copy()
            if df.empty:
                raise ValueError(f"No rows found for market ticker={ticker} in {csv_path}")

        df["date"] = pd.to_datetime(df["날짜"])
        df = df.rename(columns={"O": "open", "H": "high", "L": "low", "C": "close", "V": "volume"})
        keep = ["date", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
        df = df[keep].set_index("date").sort_index()

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = sanitize_ohlc(df)
        return df

    # Old / close-only schema
    if "날짜" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"날짜": "date"})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        if "close" not in df.columns and "C" in df.columns:
            df = df.rename(columns={"C": "close"})
        if "close" not in df.columns:
            raise ValueError(f"Market CSV missing 'close' (or 'C') column: {csv_path}")

        df = df.set_index("date").sort_index()

        # If OHLC missing, synthesize from close for indicator computation.
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        if "open" not in df.columns:
            df["open"] = df["close"]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]

        for c in ["open", "high", "low"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = sanitize_ohlc(df)
        return df

    raise ValueError(
        "Unrecognized market CSV schema. Expected either old format (date,close,...) or panel format (날짜,O,H,L,C,...). "
        f"Got columns: {list(df.columns)}"
    )
