from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_ohlc_csv(path: str) -> pd.DataFrame:
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
        # keep as numeric if possible
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df
