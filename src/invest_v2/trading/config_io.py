from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

from invest_v2.core.types import EntryRuleType, PyramidingType, TradeMode, TrailingStopType
from .trader import TraderConfig


def _read_table(path: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(p)
    return df


def load_trader_configs(path: str | Path, sheet: Optional[str] = None) -> List[TraderConfig]:
    """Load TraderConfig list from CSV/XLSX.

    Expected columns (minimal)
    --------------------------
    - trader_id
    - symbol

    Optional columns map directly to TraderConfig fields.

    Enum fields accept either the enum value string (recommended) or the enum name.
    """

    df = _read_table(path, sheet=sheet)
    if df.empty:
        return []

    # normalize column names
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    required = {"trader_id", "symbol"}
    if not required.issubset(df.columns):
        raise ValueError(f"Trader config missing required columns {sorted(required)}. Columns={list(df.columns)}")

    # Optional enable flag
    if "enabled" in df.columns:
        df = df[df["enabled"].astype(int) != 0].copy()

    out: List[TraderConfig] = []
    for _, r in df.iterrows():
        kwargs = {}

        # direct fields
        for k in asdict(TraderConfig(trader_id="_", symbol="_")).keys():
            if k in {"trader_id", "symbol"}:
                continue
            if k in df.columns and not pd.isna(r[k]):
                kwargs[k] = r[k]

        # enums
        def _as_enum(v, enum_cls):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            s = str(v).strip()
            if not s:
                return None
            # allow name
            if s in enum_cls.__members__:
                return enum_cls[s]
            return enum_cls(s)

        if "entry_rule" in kwargs:
            kwargs["entry_rule"] = _as_enum(kwargs["entry_rule"], EntryRuleType) or EntryRuleType.A_TURTLE
        if "entry_rule_long" in kwargs:
            kwargs["entry_rule_long"] = _as_enum(kwargs["entry_rule_long"], EntryRuleType)
        if "entry_rule_short" in kwargs:
            kwargs["entry_rule_short"] = _as_enum(kwargs["entry_rule_short"], EntryRuleType)
        if "trade_mode" in kwargs:
            kwargs["trade_mode"] = _as_enum(kwargs["trade_mode"], TradeMode) or TradeMode.LONG_SHORT
        if "ts_type" in kwargs:
            kwargs["ts_type"] = _as_enum(kwargs["ts_type"], TrailingStopType) or TrailingStopType.A_PCT
        if "pyramiding_type" in kwargs:
            kwargs["pyramiding_type"] = _as_enum(kwargs["pyramiding_type"], PyramidingType) or PyramidingType.A_PCT

        cfg = TraderConfig(trader_id=str(r["trader_id"]), symbol=str(r["symbol"]), **kwargs)
        out.append(cfg)

    return out
