from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Allow running this script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.plotly_viewer import create_interactive_abc_html


def _split_csv_list(s: str) -> List[str]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def _find_one(run_dir: Path, patterns: List[str]) -> Optional[Path]:
    for ptn in patterns:
        hits = sorted(run_dir.glob(ptn))
        if hits:
            return hits[0]
    return None


def _make_index_html(outdir: Path, rows: List[dict]) -> None:
    # simple index to open from browser
    lines = [
        "<html><head><meta charset='utf-8'><title>Combo Results</title></head><body>",
        "<h2>Combo Results</h2>",
        "<p>Click HTML to open interactive A/B/C (shared X-axis).</p>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>Entry</th><th>TS</th><th>PRMD</th><th>CAGR</th><th>MDD</th><th>Payoff</th><th>Trades</th><th>HTML</th></tr>",
    ]
    for r in rows:
        html_rel = r.get("html_rel", "")
        link = f"<a href='{html_rel}'>open</a>" if html_rel else ""
        lines.append(
            "<tr>"
            f"<td>{r.get('entry','')}</td>"
            f"<td>{r.get('ts','')}</td>"
            f"<td>{r.get('prmd','')}</td>"
            f"<td>{r.get('cagr','')}</td>"
            f"<td>{r.get('mdd','')}</td>"
            f"<td>{r.get('payoff','')}</td>"
            f"<td>{r.get('trades','')}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    lines += ["</table>", "</body></html>"]
    (outdir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Panel CSV (e.g., krx100_adj_5000.csv)")
    ap.add_argument("--ticker", default="005930", help="Ticker for toy run")
    ap.add_argument("--outdir", default="outputs_combo", help="Output directory for combo runs")
    ap.add_argument("--entry-rules", required=True, help="Comma-separated entry rules (e.g., C_TSMOM_CYCLE,A_TURTLE)")
    ap.add_argument("--ts-types", default="TS.A", help="Comma-separated TS types (e.g., TS.A,TS.B,TS.C)")
    ap.add_argument("--pyramiding-types", default="OFF", help="Comma-separated PRMD types (e.g., OFF,PRMD.A,PRMD.B)")
    ap.add_argument("--market-csv", default=None)
    ap.add_argument("--market-ticker", default=None)
    ap.add_argument("--disable-short", action="store_true")
    ap.add_argument("--c-no-market-filter", action="store_true")
    ap.add_argument("--make-html", action="store_true", default=True, help="Generate interactive_abc.html for each run")
    ap.add_argument("--no-make-html", action="store_false", dest="make_html")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    entry_rules = _split_csv_list(args.entry_rules)
    ts_types = _split_csv_list(args.ts_types)
    prmd_types = _split_csv_list(args.pyramiding_types)

    summary_rows: List[dict] = []

    for er in entry_rules:
        for ts in ts_types:
            for prmd in prmd_types:
                run_name = f"{er}__{ts}__{prmd}"
                run_dir = runs_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    "scripts/toy_005930_backtest.py",
                    "--csv", args.csv,
                    "--entry-rule", er,
                    "--ts-type", ts,
                    "--pyramiding-type", prmd,
                    "--outdir", str(run_dir),
                ]
                if args.market_csv:
                    cmd += ["--market-csv", args.market_csv]
                if args.market_ticker:
                    cmd += ["--market-ticker", args.market_ticker]
                if args.disable_short:
                    cmd += ["--disable-short"]
                if args.c_no_market_filter:
                    cmd += ["--c-no-market-filter"]

                print("RUN:", " ".join(cmd))
                subprocess.run(cmd, check=True)

                # collect metrics if summary.json exists
                summary_json = run_dir / "summary.json"
                cagr = mdd = payoff = trades = ""
                if summary_json.exists():
                    import json
                    sj = json.loads(summary_json.read_text(encoding="utf-8"))
                    cagr = sj.get("cagr", sj.get("CAGR", ""))
                    mdd = sj.get("mdd", sj.get("MDD", ""))
                    payoff = sj.get("payoff_ratio", sj.get("payoff", ""))
                    trades = sj.get("num_trades", sj.get("trades", ""))

                html_rel = ""
                if args.make_html:
                    # detect price/equity/trades csv inside run_dir
                    price_csv = _find_one(run_dir, ["*price*indicator*.csv", "*indicators*.csv", "price*.csv", "*plot_data*.csv"])
                    equity_csv = _find_one(run_dir, ["equity_curve.csv", "*equity*.csv", "*nav*.csv"])
                    trades_csv = _find_one(run_dir, ["trades.csv", "*trade*.csv"])
                    if price_csv and equity_csv:
                        out_html = run_dir / "interactive_abc.html"
                        try:
                            create_interactive_abc_html(
                                price_csv=str(price_csv),
                                equity_csv=str(equity_csv),
                                trades_csv=str(trades_csv) if trades_csv else None,
                                out_html=str(out_html),
                                title=f"{args.ticker} | {run_name}",
                            )
                            html_rel = str(out_html.relative_to(outdir))
                        except Exception as e:
                            print(f"[WARN] HTML generation failed for {run_name}: {e}")
                    else:
                        print(f"[WARN] Cannot find price/equity csv in {run_dir} -> skip HTML")

                summary_rows.append(
                    dict(entry=er, ts=ts, prmd=prmd, cagr=cagr, mdd=mdd, payoff=payoff, trades=trades, html_rel=html_rel)
                )

    # write summary table
    outdir.mkdir(parents=True, exist_ok=True)
    table_csv = outdir / "summary_table.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["entry", "ts", "prmd", "cagr", "mdd", "payoff", "trades", "html_rel"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    _make_index_html(outdir, summary_rows)
    print(f"Saved summary: {table_csv}")
    print(f"Open: {outdir / 'index.html'}")


if __name__ == "__main__":
    main()
