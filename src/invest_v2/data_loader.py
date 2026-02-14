from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd


def sanitize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize OHLC data.

    Background
    ----------
    Some KRX panel exports contain O/H/L=0 placeholders while close is valid.
    That leads to pathological fills (exit at 0) when the engine uses next open as fill.

    Rules
    -----
    - Convert O/H/L <= 0 to NaN (close <=0 is treated as fatal and rows are dropped)
    - Drop rows with close NaN
    - Fill missing open/high/low with close
    - Enforce high >= max(open, close) and low <= min(open, close)
    """

    out = df.copy()
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # close must exist
    out.loc[out["close"] <= 0, "close"] = pd.NA
    out = out.dropna(subset=["close"]).copy()

    # treat non-positive O/H/L as missing
    for c in ["open", "high", "low"]:
        out.loc[out[c] <= 0, c] = pd.NA

    # fill O/H/L from close
    for c in ["open", "high", "low"]:
        out[c] = out[c].fillna(out["close"])

    # enforce consistency
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)

    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    return out


@dataclass(frozen=True)
class CsvSchema:
    date_col: str
    open_col: str
    high_col: str
    low_col: str
    close_col: str
    volume_col: Optional[str] = None


def _detect_schema(df: pd.DataFrame) -> Optional[CsvSchema]:
    # Old single-symbol: date,open,high,low,close,(volume)
    if {"date", "open", "high", "low", "close"}.issubset(df.columns):
        vol = "volume" if "volume" in df.columns else None
        return CsvSchema("date", "open", "high", "low", "close", vol)

    # Panel (KRX100): 이름,날짜,ticker,O,H,L,C,V
    if {"날짜", "O", "H", "L", "C"}.issubset(df.columns):
        vol = "V" if "V" in df.columns else None
        return CsvSchema("날짜", "O", "H", "L", "C", vol)

    # Another common Korean naming
    if {"일자", "시가", "고가", "저가", "종가"}.issubset(df.columns):
        vol = "거래량" if "거래량" in df.columns else None
        return CsvSchema("일자", "시가", "고가", "저가", "종가", vol)

    return None


def load_ohlc_auto(csv_path: str, symbol: Optional[str] = None, ticker_col: str = "ticker") -> pd.DataFrame:
    """Load OHLC CSV with schema auto-detection.

    Supports:
      - Old format: date,open,high,low,close,(volume)
      - KRX panel:  이름,날짜,ticker,O,H,L,C,V
      - Some Korean formats: 일자,시가,고가,저가,종가
    """
    df = pd.read_csv(Path(csv_path), dtype={ticker_col: str}, low_memory=False)
    schema = _detect_schema(df)
    if schema is None:
        raise ValueError(f"Unrecognized CSV schema: {csv_path}. Columns={list(df.columns)}")

    # panel ticker filtering
    if schema.date_col in {"날짜"} and symbol is not None and ticker_col in df.columns:
        sym = str(symbol).zfill(6)
        df[ticker_col] = df[ticker_col].astype(str).str.zfill(6)
        df = df[df[ticker_col] == sym].copy()
        if df.empty:
            raise ValueError(f"No rows found for symbol={sym} in panel CSV: {csv_path}")

    df = df.rename(
        columns={
            schema.date_col: "date",
            schema.open_col: "open",
            schema.high_col: "high",
            schema.low_col: "low",
            schema.close_col: "close",
        }
    )

    keep = ["date", "open", "high", "low", "close"]
    if schema.volume_col and schema.volume_col in df.columns:
        df = df.rename(columns={schema.volume_col: "volume"})
        keep.append("volume")

    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = sanitize_ohlc(df)
    return df


def load_market_csv(csv_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """Load market index / futures CSV.

    For the toy system, we only require 'close'. If open/high/low are missing, they are
    filled from close during sanitization.
    """
    m = load_ohlc_auto(csv_path, symbol=symbol)
    return m


def load_ohlc_panel_symbols(csv_path: str, symbols: List[str], ticker_col: str = "ticker") -> Dict[str, pd.DataFrame]:
    """Load multiple symbols from a KRX-style panel CSV in one pass.

    Parameters
    ----------
    csv_path:
        Panel CSV path (e.g., data/krx100_adj_5000.csv)
    symbols:
        List of tickers (6-digit strings or ints)

    Returns
    -------
    dict[symbol] -> OHLC dataframe (date-indexed)

    Notes
    -----
    - This is more efficient than calling `load_ohlc_auto` repeatedly for many traders.
    - It assumes the KRX panel schema (columns like 날짜,ticker,O,H,L,C,(V)).
    """

    sym_set = {str(s).zfill(6) for s in symbols}

    # read minimal columns when available
    usecols = None
    try:
        # include volume if present
        usecols = ["날짜", ticker_col, "O", "H", "L", "C", "V"]
    except Exception:
        usecols = None

    df = pd.read_csv(Path(csv_path), dtype={ticker_col: str}, low_memory=False, usecols=lambda c: True if usecols is None else c in usecols)

    # schema check
    if "날짜" not in df.columns or ticker_col not in df.columns:
        raise ValueError(f"load_ohlc_panel_symbols expects KRX panel schema with '날짜' and '{ticker_col}'. Columns={list(df.columns)}")

    df[ticker_col] = df[ticker_col].astype(str).str.zfill(6)
    df = df[df[ticker_col].isin(sym_set)].copy()
    if df.empty:
        raise ValueError(f"No rows found for symbols={sorted(sym_set)} in panel CSV: {csv_path}")

    # normalize
    df = df.rename(columns={"날짜": "date", "O": "open", "H": "high", "L": "low", "C": "close"})
    if "V" in df.columns:
        df = df.rename(columns={"V": "volume"})

    df["date"] = pd.to_datetime(df["date"])
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby(ticker_col):
        g = g.sort_values("date")
        cols = ["date", "open", "high", "low", "close"] + (["volume"] if "volume" in g.columns else [])
        gg = g[cols].copy()
        gg = gg.set_index("date")
        gg = sanitize_ohlc(gg)
        out[str(sym)] = gg

    # ensure all requested symbols exist
    missing = [s for s in sym_set if s not in out]
    if missing:
        raise ValueError(f"Missing symbols in panel CSV: {missing}")

    return out
