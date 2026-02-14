from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from invest_v2.data_loader import load_market_csv, load_ohlc_panel_symbols
from invest_v2.prep import add_indicators
from invest_v2.reporting.plotly_abc import write_interactive_abc
from invest_v2.trading import load_trader_configs
from invest_v2.backtest.master_engine import TraderMasterBacktester
from invest_v2.trading.trader_master import TraderMasterConfig


def _normalize_symbol(sym: object) -> str:
    s = str(sym).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6) if s.isdigit() else s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run portfolio backtest with Trader/TraderMaster from CSV/XLSX config")
    p.add_argument("--csv", required=True, help="KRX panel CSV (e.g., data/krx100_adj_5000.csv)")
    p.add_argument("--trader-config", required=True, help="Trader config CSV/XLSX")
    p.add_argument("--sheet", default=None, help="Excel sheet name (if xlsx)")
    p.add_argument("--market-csv", default=None, help="Optional market CSV for market_cycle_phase")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out-dir", default="runs/trader_master_run")
    p.add_argument("--max-units-total", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trader_cfgs = load_trader_configs(args.trader_config, sheet=args.sheet)
    if not trader_cfgs:
        raise SystemExit("No enabled traders in config.")

    for cfg in trader_cfgs:
        cfg.symbol = _normalize_symbol(cfg.symbol)

    symbols = sorted({cfg.symbol for cfg in trader_cfgs})

    market_df = None
    if args.market_csv:
        market_df = load_market_csv(args.market_csv)

    panel = load_ohlc_panel_symbols(args.csv, symbols=symbols)

    # date filtering + indicators per symbol
    symbol_dfs = {}
    for sym, df in panel.items():
        if args.start:
            df = df[df.index >= pd.to_datetime(args.start)]
        if args.end:
            df = df[df.index <= pd.to_datetime(args.end)]
        df = add_indicators(df, market_df=market_df)
        symbol_dfs[sym] = df

    master_cfg = TraderMasterConfig(max_units_total=int(args.max_units_total))
    bt = TraderMasterBacktester(symbol_dfs=symbol_dfs, trader_cfgs=trader_cfgs, master_cfg=master_cfg)
    res = bt.run()

    # Save master outputs
    res.master_equity_curve.reset_index().rename(columns={"index": "date"}).to_csv(out_dir / "master_equity_curve.csv", index=False)
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    res.fills.to_csv(out_dir / "fills.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(res.summary, indent=2, default=str), encoding="utf-8")

    # Per-trader outputs
    traders_dir = out_dir / "traders"
    traders_dir.mkdir(parents=True, exist_ok=True)
    for tid, trader in bt.traders.items():
        tdir = traders_dir / tid
        tdir.mkdir(parents=True, exist_ok=True)

        eq = res.trader_equity_curves.get(tid)
        if eq is not None and len(eq):
            eq.reset_index().rename(columns={"index": "date"}).to_csv(tdir / "equity_curve.csv", index=False)

        trader.trades_df().to_csv(tdir / "trades.csv", index=False)
        trader.fills_df().to_csv(tdir / "fills.csv", index=False)

        # plot data (OHLC + indicators)
        trader_symbol = _normalize_symbol(trader.symbol)
        sym_df = symbol_dfs[trader_symbol].copy().reset_index().rename(columns={"index": "date"})
        sym_df.to_csv(tdir / "plot_data.csv", index=False)

        write_interactive_abc(
            plot_data_csv=tdir / "plot_data.csv",
            equity_curve_csv=tdir / "equity_curve.csv",
            trades_csv=tdir / "trades.csv",
            fills_csv=tdir / "fills.csv",
            out_html=tdir / "abc.html",
            title=f"{tid} ({trader_symbol})",
        )

        (tdir / "trader_report.txt").write_text(trader.how_did_you_trade(max_lines=5000), encoding="utf-8")

    print(f"Saved portfolio run to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
