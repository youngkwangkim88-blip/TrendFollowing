"""Trading layer (strategy execution + position management).

This package introduces two core concepts:

- Trader: owns *position state* for a single symbol (or a sleeve) and produces
  orders from strategy/filters/position-management rules.

- TraderMaster: owns *accounts* (KRW, USD, CMA, ... or sleeves) and arbitrates
  capital allocation and cross-trader constraints.

The goal is to decouple:
  (1) signal generation (strategy)
  (2) position operation (trader)
  (3) capital allocation / risk budget (master)

so that optimization can tune strategies per symbol/sector and a higher-level
optimizer can allocate capital across traders.
"""

from .trader import Trader, TraderConfig, FillEvent
from .trader_master import TraderMaster, TraderMasterConfig
from .config_io import load_trader_configs

__all__ = [
    "Trader",
    "TraderConfig",
    "FillEvent",
    "TraderMaster",
    "TraderMasterConfig",
    "load_trader_configs",
]
