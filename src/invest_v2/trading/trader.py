from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from invest_v2.core.types import Side


@dataclass
class PositionLot:
    """A single fill-lot used for average price and auditability."""

    date: str
    price: float
    shares: int
    reason: str = ""


class Trader:
    """Per-symbol position manager and trade logger.

    Responsibilities
    ----------------
    - Maintain *lot-level* fills for the currently open position.
    - Provide stable reporting fields:
        - `first_entry_date/price` never change.
        - `avg_entry_price` changes with pyramiding.
    - Emit two log streams:
        - fills: every executed fill (ENTRY/PYRAMID/EXIT)
        - trades: one summary row per completed position

    Non-goals (for now)
    -------------------
    - Risk management / signal generation
    - Cash ledger management (handled by Account / higher layer)
    """

    def __init__(self, symbol: str):
        self.symbol = str(symbol)

        # Current position (open) state
        self.pos_side: Side = Side.FLAT
        self.pos_units: int = 0
        self.pos_shares: int = 0

        # Open lots (used for avg price + realized PnL)
        self._lots: List[PositionLot] = []

        # Trade identity
        self._trade_seq: int = 0
        self._active_trade_id: Optional[str] = None

        # Logs
        self.fills: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

    # ---------- computed properties ----------
    @property
    def first_entry_date(self) -> Optional[str]:
        return self._lots[0].date if self._lots else None

    @property
    def first_entry_price(self) -> Optional[float]:
        return float(self._lots[0].price) if self._lots else None

    @property
    def avg_entry_price(self) -> float:
        if not self._lots:
            return 0.0
        notional = sum(float(l.price) * int(l.shares) for l in self._lots)
        shares = sum(int(l.shares) for l in self._lots)
        return float(notional / max(1, shares))

    @property
    def entry_notional_gross(self) -> float:
        """Sum(entry_price * shares) across lots (gross, no fees)."""
        return float(sum(float(l.price) * int(l.shares) for l in self._lots))

    @property
    def num_entries(self) -> int:
        return int(len(self._lots))

    # ---------- public API (fills) ----------
    def on_entry_fill(
        self,
        *,
        date: str,
        side: Side,
        price: float,
        shares: int,
        reason: str,
        is_pyramid: bool,
    ) -> None:
        """Apply an entry fill (new position or pyramiding)."""
        date = str(date)
        price = float(price)
        shares = int(shares)
        reason = str(reason or "")

        if shares <= 0:
            return

        if self.pos_side == Side.FLAT:
            # Open a new trade
            self._trade_seq += 1
            self._active_trade_id = f"{self.symbol}-{self._trade_seq:06d}"
            self.pos_side = side
            self.pos_units = 1
            self.pos_shares = shares
            self._lots = [PositionLot(date=date, price=price, shares=shares, reason=reason)]

            self.fills.append(
                {
                    "trade_id": self._active_trade_id,
                    "symbol": self.symbol,
                    "side": int(side.value),
                    "action": "ENTRY",
                    "date": date,
                    "price": price,
                    "shares": shares,
                    "reason": reason,
                    "pos_units_after": int(self.pos_units),
                    "pos_shares_after": int(self.pos_shares),
                    "avg_entry_price_after": float(self.avg_entry_price),
                }
            )
            return

        # Add to existing position
        if side != self.pos_side:
            raise ValueError(
                f"Trader({self.symbol}): cannot add lot with side={side} when pos_side={self.pos_side}."
            )

        self._lots.append(PositionLot(date=date, price=price, shares=shares, reason=reason))
        self.pos_shares += shares
        self.pos_units += 1

        self.fills.append(
            {
                "trade_id": self._active_trade_id,
                "symbol": self.symbol,
                "side": int(side.value),
                "action": "PYRAMID" if is_pyramid else "ADD",
                "date": date,
                "price": price,
                "shares": shares,
                "reason": reason,
                "pos_units_after": int(self.pos_units),
                "pos_shares_after": int(self.pos_shares),
                "avg_entry_price_after": float(self.avg_entry_price),
            }
        )

    def on_exit_fill(
        self,
        *,
        date: str,
        price: float,
        reason: str,
        sell_cost_rate: float,
    ) -> Dict[str, Any]:
        """Close the current position and emit a trade summary.

        Returns
        -------
        trade_row:
            A dict suitable for `trades.csv`.
        """
        if self.pos_side == Side.FLAT or self.pos_shares <= 0 or not self._lots:
            return {}

        date = str(date)
        price = float(price)
        reason = str(reason or "")
        sell_cost_rate = float(sell_cost_rate)

        side = self.pos_side
        total_shares = int(self.pos_shares)
        entry_gross = float(self.entry_notional_gross)
        avg_px = float(self.avg_entry_price)

        if side == Side.LONG:
            exit_net = price * total_shares * (1.0 - sell_cost_rate)
            realized = exit_net - entry_gross
        else:
            # Short: fee applies on entry (sell), cover has 0% fee
            entry_net = entry_gross * (1.0 - sell_cost_rate)
            cover_cost = price * total_shares
            realized = entry_net - cover_cost

        trade_row: Dict[str, Any] = {
            "trade_id": self._active_trade_id,
            "symbol": self.symbol,
            "side": int(side.value),
            # Keep legacy column names for backward compatibility.
            "entry_date": self.first_entry_date,
            "entry_price": self.first_entry_price,
            # New explicit fields.
            "avg_entry_price": avg_px,
            "num_entries": int(self.num_entries),
            "shares": total_shares,
            "exit_date": date,
            "exit_price": price,
            "realized_pnl": float(realized),
            "exit_reason": reason,
        }

        self.fills.append(
            {
                "trade_id": self._active_trade_id,
                "symbol": self.symbol,
                "side": int(side.value),
                "action": "EXIT",
                "date": date,
                "price": price,
                "shares": total_shares,
                "reason": reason,
                "pos_units_after": 0,
                "pos_shares_after": 0,
                "avg_entry_price_after": 0.0,
            }
        )
        self.trades.append(trade_row)

        # Reset open state
        self.pos_side = Side.FLAT
        self.pos_units = 0
        self.pos_shares = 0
        self._lots = []
        self._active_trade_id = None

        return trade_row

    # ---------- reporting helpers ----------
    def how_did_you_trade(self, max_trades: int = 50) -> str:
        """Human-readable trade history (for quick debugging)."""

        lines: List[str] = []
        lines.append(f"Trader(symbol={self.symbol})")
        lines.append(f"- completed_trades={len(self.trades)}  fills={len(self.fills)}")
        lines.append("")

        # Trades (latest first)
        show = list(reversed(self.trades))[: int(max_trades)]
        for t in show:
            lines.append(
                " | ".join(
                    [
                        str(t.get("trade_id", "")),
                        "LONG" if int(t.get("side", 0)) == 1 else "SHORT",
                        f"{t.get('entry_date')} @ {float(t.get('entry_price') or 0):,.2f}",
                        f"avg={float(t.get('avg_entry_price') or 0):,.2f}",
                        f"x{int(t.get('num_entries') or 0)}",
                        f"shares={int(t.get('shares') or 0)}",
                        f"exit {t.get('exit_date')} @ {float(t.get('exit_price') or 0):,.2f}",
                        f"pnl={float(t.get('realized_pnl') or 0):,.0f}",
                        str(t.get("exit_reason", "")),
                    ]
                )
            )
        return "\n".join(lines)
