# AGENTS.md (draft)

## Golden commands
- Install: uv sync  (or: pip install -r requirements.txt)
- Tests: pytest -q
- Lint: ruff check . && ruff format --check .
- Typecheck: mypy src
- Backtest smoke: python scripts/toy_005930_backtest.py --csv data/sample_005930.csv --entry-rule A_20_PL

## Safety / trading constraints
- Never place live orders. Only edit backtest code and broker stubs.
- Any broker integration must be behind a "dry-run" default.
- Do not print or commit secrets. Use env vars only.

## Project rules (high level)
- Signals are computed on daily close; execution is next-day 09:00 market order.
- Keep backtest accounting rules consistent with docs/.
