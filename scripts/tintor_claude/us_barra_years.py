"""Faktorski portfelji iz regresije: dnevni prinosi, kumulativni prinos i Sharpe po godinama.

Izlaz: results/tintor_claude_us_book/barra_dnevni_prinosi.parquet, barra_po_godinama.csv, barra_kumulativno.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from us_barra import RES, SCORES, fmp_returns, fmp_weights, load  # noqa: E402

SHOW = ["kvaliteta", "kvaliteta+vrednovanje", "bez koreliranih", "stil kredit", "4 stila", "altman+kralicek", "zalihe (pad dana)"]
PLOT = {"kvaliteta": "#2a78d6", "kvaliteta+vrednovanje": "#eb5a2a", "4 stila": "#1b9e77", "zalihe (pad dana)": "#8a4fd0"}


def main() -> None:
    x, R, base, cost = load()
    ser = {}
    for v in SCORES:
        for label, full in [("jednostavna", False), ("puna", True)]:
            W = {s: fmp_weights(g, v, full) for s, g in x.groupby("signal_date")}
            ser[(label, v)] = fmp_returns({s: w for s, w in W.items() if len(w)}, R, cost)[0]
    d = pd.DataFrame(ser).sort_index()
    d.columns = [f"{a}|{b}" for a, b in d.columns]
    d.to_parquet(RES / "barra_dnevni_prinosi.parquet")
    g = d.groupby(d.index.year)
    yr = pd.concat({"prinos": g.apply(lambda z: (1 + z.fillna(0)).prod() - 1), "sharpe": g.mean() / g.std() * np.sqrt(252),
                    "kumulativno": ((1 + d.fillna(0)).cumprod() - 1).groupby(d.index.year).last()}, axis=1)
    yr.to_csv(RES / "barra_po_godinama.csv")
    pd.set_option("display.width", 260)
    cols = ["puna|" + v for v in SHOW]
    for k, mult, nd in [("prinos", 100, 1), ("kumulativno", 100, 0), ("sharpe", 1, 2)]:
        t = (yr[k][cols] * mult).round(nd)
        t.columns = SHOW
        print("\npuna regresija: " + k)
        print(t.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    cum = ((1 + d.fillna(0)).cumprod() - 1) * 100
    for v, c in PLOT.items():
        s = cum["puna|" + v]
        ax.plot(s.index, s, color=c, lw=1.6, label=v)
        ax.annotate(f"{v}  {s.iloc[-1]:.0f} %", (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#333333")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(pd.Timestamp("2011-01-01"), color="#999999", lw=0.8, ls=":")
    ax.set_title("Faktorski portfelji kvalitete (puna regresija), kumulativni prinos long-short, neto od troška", loc="left", fontsize=11)
    ax.set_ylabel("%")
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=1900))
    fig.tight_layout()
    fig.savefig(RES / "barra_kumulativno.png")


if __name__ == "__main__":
    main()
