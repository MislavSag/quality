"""Jedan pokazatelj zasebno: skupine po vrijednosti, prinosi skupina i razlika, po razdobljima.

Primjer: python -P scripts/tintor_claude/us_book_factor.py INVENTORY_TIE_UP_DAYS_TTM_PROXY_DIRECTION_3Y
Za pokazatelj smjera (-1/0/+1) skupine su vrijednosti; za neprekidni su petine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from us import CACHE, CFG, simulate, stats  # noqa: E402
from us_book import P, forward_returns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERIODS = [("cijelo", None, None), ("do_2010", None, "2010-12-31"), ("od_2011", "2011-01-01", None), ("od_2019", "2019-01-01", None)]


def main(feat: str) -> None:
    b = pd.read_parquet(CACHE / "book_features.parquet")
    b = b[b.book_ok].reset_index(drop=True)
    R = pd.read_parquet(CACHE / "returns.parquet")
    cal, cost = R.index, CFG["execution"]["cost_per_side"]
    v = b[P + feat].astype("float64")
    if v.nunique() <= 3:
        b["g"] = v.map({-1.0: "pada", 0.0: "isto", 1.0: "raste"}).fillna("nema podatka")
        lo, hi = "pada", "raste"
    else:
        q = v.groupby(b.signal_date).transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
        b["g"] = q.map(lambda k: f"petina {int(k)}" if k == k else "nema podatka")
        lo, hi = "petina 1", "petina 5"
    print(b.groupby("signal_date").g.value_counts().unstack().iloc[::12].fillna(0).astype(int).to_string())
    print("udjeli:", b.g.value_counts(normalize=True).round(3).to_dict())
    sec = b.assign(s=b.FUND_EODHD_GIC_SECTOR.fillna("?")).groupby("s").g.value_counts(normalize=True).unstack().round(2)
    print(sec.to_string())

    navs = {"baza": simulate({s: list(x.security_id) for s, x in b.groupby("signal_date")}, R, cal, cost)[0]}
    for g, x in b.groupby("g"):
        if x.groupby("signal_date").size().min() < 5 or x.signal_date.nunique() < b.signal_date.nunique():
            continue
        navs[g] = simulate({s: list(y.security_id) for s, y in x.groupby("signal_date")}, R, cal, cost)[0]
    rows = []
    for per, a, e in PERIODS:
        for g, nv in navs.items():
            rows.append({"razdoblje": per, "skupina": g, **stats(nv, a, e)})
        x = pd.concat([navs[lo], navs[hi], navs["baza"]], axis=1, keys=["lo", "hi", "m"]).loc[a:e].pct_change().dropna()
        d = x.lo - x.hi
        cum = (1 + d).cumprod()
        m = (1 + x).resample("ME").prod() - 1
        dm = m.lo - m.hi
        beta = np.cov(dm, m.m)[0, 1] / m.m.var()
        print(f"{per}: {lo} minus {hi}: godisnje {d.mean()*252:.3%}, vol {d.std()*np.sqrt(252):.3%}, Sharpe {d.mean()/d.std()*np.sqrt(252):.2f}, "
              f"t {dm.mean()/dm.std()*np.sqrt(len(dm)):.2f}, max pad {(cum/cum.cummax()-1).min():.1%}, beta {beta:.2f}")
    print(pd.DataFrame(rows).pivot(index="skupina", columns="razdoblje", values="godisnji_prinos")[[p[0] for p in PERIODS]].round(4).to_string())
    print(pd.DataFrame(rows).pivot(index="skupina", columns="razdoblje", values="prinos_kroz_volatilnost")[[p[0] for p in PERIODS]].round(2).to_string())

    # veza s prinosom iduceg tromjesecja po datumu signala, unutar sektora i ukupno
    b["fwd"] = forward_returns(b, R)
    k = b[b.g.isin([lo, hi]) & b.fwd.notna()]
    by = k.groupby(["signal_date", "g"]).fwd.mean().unstack()
    d = by[lo] - by[hi]
    for per, a, e in PERIODS:
        z = d.loc[a:e]
        print(f"{per}: tromjesecna razlika prosjeka {z.mean():.3%}, t {z.mean()/z.std()*np.sqrt(len(z)):.2f}, pozitivnih {int((z>0).sum())}/{len(z)}")
    k = k.assign(s=k.FUND_EODHD_GIC_SECTOR.fillna("?"))
    ws = k.groupby(["signal_date", "s", "g"]).fwd.mean().unstack().dropna()
    dn = (ws[lo] - ws[hi]).groupby("signal_date").mean()
    for per, a, e in PERIODS:
        z = dn.loc[a:e]
        print(f"{per}: unutar sektora {z.mean():.3%}, t {z.mean()/z.std()*np.sqrt(len(z)):.2f}")
    yr = d.groupby(d.index.year).sum()
    print("zbroj tromjesecnih razlika po godini:", yr.round(3).to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
