# Smoke test dataset

This folder contains **synthetic OHLCV data** and a simple runner script to sanity-check
core execution logic (fills, stop models, emergency stops, even-stop, short interest).

## Files

- `cases/*.csv`: synthetic daily OHLCV with an additional `manual_entry` column.
  - `manual_entry = 1`  -> emit **LONG** entry decision at that day's close (fills at next day's open)
  - `manual_entry = -1` -> emit **SHORT** entry decision at that day's close (fills at next day's open)
  - `manual_entry = 0`  -> no entry

- `manifest.json`: defines per-case config overrides and expected fill events.
- Run script: `python scripts/run_smoke_tests.py`

## Notes

- These cases rely on a special entry rule `EntryRuleType.MANUAL` (added for smoke tests).
- All cases disable PL/cycle/market filters to keep signals deterministic.
- Prices are chosen to exercise:
  - ES1 (open-based) emergency stop (TOUCH)
  - ES2 (prev-close-based) emergency stop (TOUCH / GAP)
  - ES3 (close-to-close) emergency stop (scheduled exit at next open)
  - EVEN-STOP (TOUCH / GAP)
  - Monthly short interest deduction (monthly cashflow)
  - Forced short max-hold-days exit
  - Fee accounting consistency (sell_cost_rate applied on sell / short-sell)
  - Unit sizing defense (unit_shares=0 -> no order / no fill)
  - Scheduled-exit vs gap-stop collision (single-exit + reason priority)
