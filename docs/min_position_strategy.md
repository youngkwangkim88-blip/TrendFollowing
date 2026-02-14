# Minimum Position Mode (0.5 unit)

Purpose: reduce risk during regime-transition / choppy periods.

## Trigger
If **two consecutive trades** are closed with a loss, the next entry uses **0.5 unit** sizing.

## Behavior
- **Entry size:** 0.5 × (normal 1-unit shares), floored with minimum 1 share.
- **First pyramiding after that entry:** also uses 0.5 unit once.
- **Second pyramiding and later:** returns to normal 1-unit sizing.
- The consecutive-loss counter resets to 0 after a winning trade.
