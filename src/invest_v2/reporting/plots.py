from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity: pd.DataFrame, outpath: str) -> None:
    nav = equity["nav"].astype(float)
    plt.figure()
    plt.plot(pd.to_datetime(equity.index), nav.values)
    plt.title("Equity Curve (NAV)")
    plt.xlabel("Date")
    plt.ylabel("NAV (KRW)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
