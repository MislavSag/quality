"""Raspodjela prinosa SPY-a i strategija: percentili mjesecnih i 12-mjesecnih prinosa te prinosi po godinama.

Izlaz: results/tintor_claude_us_book/raspodjela_percentili.csv, raspodjela_po_godinama.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from us import CACHE, CFG, simulate  # noqa: E402
from us_book import P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "tintor_claude_us_book"
Q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def main() -> None:
    old = pd.read_parquet(ROOT / "results/tintor_claude_us/nav.parquet")
    new = pd.read_parquet(RES / "nav.parquet")
    b = pd.read_parquet(CACHE / "book_features.parquet")
    b = b[b.book_ok]
    R = pd.read_parquet(CACHE / "returns.parquet")
    d = b[P + "INVENTORY_TIE_UP_DAYS_TTM_PROXY_DIRECTION_3Y"]
    grp = {k: simulate({s: list(x.security_id) for s, x in b[d == v].groupby("signal_date")}, R, R.index,
                       CFG["execution"]["cost_per_side"])[0] for k, v in [("zalihe padaju", -1.0), ("zalihe rastu", 1.0)]}
    nav = pd.DataFrame({"SPY": old["SPY"], "baza": new["baza"], "kvaliteta": old["kvaliteta"],
                        "kvaliteta+vrednovanje": old["kvaliteta+vrednovanje"], "4 stila": new["4 stila"],
                        "zalihe padaju": grp["zalihe padaju"], "zalihe rastu": grp["zalihe rastu"]}).dropna()
    nav = nav / nav.iloc[0]
    day = nav.pct_change().dropna()
    ls = (1 + day["zalihe padaju"] - day["zalihe rastu"]).cumprod()
    nav["padaju minus rastu"] = ls.reindex(nav.index).fillna(1.0)
    m = nav.resample("ME").last()
    mr = m.pct_change().dropna()
    r12 = m.pct_change(12).dropna()
    dd = nav / nav.cummax() - 1
    out = []
    for name, x in [("mjesecni prinos", mr), ("prinos u 12 mjeseci", r12), ("pad od vrha (dnevno)", dd)]:
        t = x.quantile(Q).T
        t.columns = [f"p{int(q * 100)}" for q in Q]
        t.insert(0, "prosjek", x.mean())
        if name != "pad od vrha (dnevno)":
            t["udio_negativnih"] = (x < 0).mean()
            t["asimetrija"] = x.skew()
        out.append(t.assign(mjera=name).reset_index(names="strategija"))
    pct = pd.concat(out, ignore_index=True)
    pct.to_csv(RES / "raspodjela_percentili.csv", index=False)
    yr = nav.resample("YE").last()
    yr = pd.concat([nav.iloc[[0]], yr]).pct_change().dropna()
    yr.index = yr.index.year
    yr.to_csv(RES / "raspodjela_po_godinama.csv")
    # petogodisnja razdoblja: godisnji prinos, volatilnost, najveci pad
    rows = []
    for a in range(1999, 2027, 5):
        x = nav.loc[str(a):str(a + 4)]
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        r = x.pct_change().dropna()
        rows.append(pd.DataFrame({"razdoblje": f"{a}-{min(a + 4, 2026)}", "godisnji_prinos": (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1,
                                  "volatilnost": r.std() * np.sqrt(252), "max_pad": (x / x.cummax() - 1).min()}).reset_index(names="strategija"))
    per = pd.concat(rows, ignore_index=True)
    per.to_csv(RES / "raspodjela_po_razdobljima.csv", index=False)
    pd.set_option("display.width", 260)
    for name in pct.mjera.unique():
        print("\n" + name)
        print((pct[pct.mjera == name].drop(columns="mjera").set_index("strategija") * 100).round(1).dropna(axis=1, how="all").to_string())
    print("\nprinos po godinama (%)")
    print((yr * 100).round(1).to_string())
    for c in ["godisnji_prinos", "volatilnost", "max_pad"]:
        print("\n" + c + " po razdobljima (%)")
        print((per.pivot(index="strategija", columns="razdoblje", values=c).reindex(nav.columns) * 100).round(1).to_string())
    print(nav.index[0], nav.index[-1])


if __name__ == "__main__":
    main()
