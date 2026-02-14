"""Trading domain layer.

This package introduces two core concepts:

- **Trader**: manages a single symbol's position lifecycle (entry, pyramiding, exit)
  and produces auditable logs (fills + trade summaries).
- **TraderMaster**: (thin for now) owns account ledgers and provides access to
  per-symbol Traders.

The immediate motivation is to avoid the classic "ghost entry" reporting bug:
when pyramiding updates the *average entry price*, it must not overwrite the
*first entry date/price* used for chart markers and human review.
"""

from .trader import Trader
from .trader_master import TraderMaster

__all__ = ["Trader", "TraderMaster"]
