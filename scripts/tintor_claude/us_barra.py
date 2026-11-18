"""Ocjene kvalitete na dva nacina: razvrstavanje u kvantile i faktorski portfelji iz presjecne regresije (Barra).

Razvrstavanje: gornjih naprema donjih 20 %, 10 % i 5 % po ocjeni, jednaki ponderi, neto od troska.
Regresija: na svaki datum signala prinosi se objasnjavaju standardiziranom ocjenom; ponderi su korijen
trzisne kapitalizacije. Redak rjesenja (X'WX)^-1 X'W za ocjenu je portfelj koji ima jedinicnu izlozenost
ocjeni i nultu ostalim varijablama. Dvije specifikacije:
  jednostavna: konstanta + ocjena
  puna: sektori + velicina + beta + momentum + knjigovodstvena/trzisna vrijednost + ocjena
Portfelj se skalira na 1 dugo i 1 kratko, drzi do iduceg signala (dnevno vracanje na zadane pondere)
i placa trosak na promjenu pondera. Izlaz: results/tintor_claude_us_book/barra_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_short import ls_stats  # noqa: E402
from us import CACHE, CFG, simulate, stats  # noqa: E402
from us_book import P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "tintor_claude_us_book"
PERIODS = [("cijelo", None, None), ("do_2010", None, "2010-12-31"), ("od_2011", "2011-01-01", None)]
SCORES = ["kvaliteta", "kvaliteta+vrednovanje", "altman", "kralicek", "altman+kralicek", "altman+kralicek 3g i smjer",
          "stil zarada", "stil kredit", "stil financiranje", "stil proizvodnja", "4 stila", "4 stila, prosjek 3g",
          "bez koreliranih", "ponder slicnosti", "filtar ugrozenih + 4 stila", "zalihe (pad dana)"]
FRACS = [("petine", 0.20), ("desetine", 0.10), ("5 %", 0.05)]


def zscore(v: pd.Series, w: pd.Series) -> pd.Series:
    """Standardizacija kao kod Barre: sredina ponderirana kapitalizacijom, odstupanje jednako ponderirano, rez na +-3."""
    v = v.astype("float64")
    lo, hi = v.quantile(0.01), v.quantile(0.99)
    v = v.clip(lo, hi)
    ok = v.notna()
    mu = np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
    return ((v - mu) / v.std()).clip(-3, 3)


def controls(b: pd.DataFrame, R: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Beta (252 dana prema bazi) i momentum (12 mjeseci bez zadnjeg) na datum signala."""
    A = R.to_numpy(dtype="float64")
    col = {c: i for i, c in enumerate(R.columns)}
    m = mkt.reindex(R.index).to_numpy()
    out = []
    for s, x in b.groupby("signal_date"):
        i = R.index.get_loc(s)
        ids = [col[c] for c in x.security_id]
        w = A[max(i - 251, 0):i + 1][:, ids]
        mm = m[max(i - 251, 0):i + 1]
        ok = np.isfinite(w) & np.isfinite(mm)[:, None]
        n = ok.sum(axis=0)
        wz, mz = np.where(ok, w, 0.0), np.where(ok, mm[:, None], 0.0)
        cov = (wz * mz).sum(0) / n - wz.sum(0) / n * mz.sum(0) / n
        var = (mz ** 2).sum(0) / n - (mz.sum(0) / n) ** 2
        beta = np.where(n >= 120, cov / var, np.nan)
        mom = np.where(n >= 200, np.exp(np.log1p(np.where(ok[:-21], w[:-21], 0.0)).sum(0)) - 1, np.nan)
        out.append(pd.DataFrame({"beta": beta, "mom": mom}, index=x.index))
    return pd.concat(out)


def fmp_weights(x: pd.DataFrame, score: str, full: bool) -> pd.Series:
    d = x.dropna(subset=[score])
    if len(d) < 60:
        return pd.Series(dtype=float)
    w = np.sqrt(d.mcap.to_numpy())
    f = zscore(d[score], pd.Series(w, index=d.index)).to_numpy()
    if full:
        sec = pd.get_dummies(d.sector).to_numpy(dtype="float64")
        sec = sec[:, sec.sum(0) > 0]
        st = np.column_stack([zscore(d[c], pd.Series(w, index=d.index)).fillna(0.0).to_numpy() for c in ["size", "beta", "mom", "bm"]])
        X = np.column_stack([f, st, sec])
    else:
        X = np.column_stack([f, np.ones(len(d))])
    XtW = X.T * w
    row = np.linalg.pinv(XtW @ X)[0] @ XtW
    return pd.Series(row / np.abs(row).sum() * 2.0, index=d.security_id.to_numpy())   # 1 dugo + 1 kratko


def fmp_returns(weights: dict, R: pd.DataFrame, cost: float) -> tuple[pd.Series, float]:
    A = np.nan_to_num(R.to_numpy(dtype="float64"), nan=0.0)
    col = {c: i for i, c in enumerate(R.columns)}
    sig = sorted(weights)
    tix = [R.index.get_loc(s) + 1 for s in sig]
    r = np.full(len(R), np.nan)
    prev, turn = np.zeros(A.shape[1]), 0.0
    for k, t in enumerate(tix):
        end = tix[k + 1] if k + 1 < len(tix) else len(R) - 1
        w = np.zeros(A.shape[1])
        ws = weights[sig[k]]
        w[[col[c] for c in ws.index]] = ws.to_numpy()
        tr = np.abs(w - prev).sum()
        turn += tr
        r[t + 1:end + 1] = A[t + 1:end + 1] @ w
        r[t + 1] -= tr * cost
        prev = w
    s = pd.Series(r, index=R.index).dropna()
    return s, turn / ((s.index[-1] - s.index[0]).days / 365.25)


def ret_stats(r: pd.Series, base: pd.Series, a, e) -> dict:
    x = pd.concat([r, base.pct_change()], axis=1, keys=["f", "m"]).sort_index().loc[a:e].dropna()
    cum = (1 + x.f).cumprod()
    m = (1 + x).resample("ME").prod() - 1
    return {"godisnje": x.f.mean() * 252, "volatilnost": x.f.std() * np.sqrt(252), "sharpe": x.f.mean() / x.f.std() * np.sqrt(252),
            "t": m.f.mean() / m.f.std() * np.sqrt(len(m)), "max_pad": (cum / cum.cummax() - 1).min(),
            "beta": np.cov(m.f, m.m)[0, 1] / m.m.var()}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, float]:
    """Ocjene i kontrolne varijable po datumu signala, dnevni prinosi, baza i trosak."""
    b = pd.read_parquet(CACHE / "book_features.parquet")
    b = b[b.book_ok].reset_index(drop=True)
    sc = pd.read_parquet(RES / "scores.parquet").drop(columns=["ticker", "adt"])
    old = pd.read_parquet(ROOT / "results/tintor_claude_us/scores.parquet")[["signal_date", "security_id", "s_kvaliteta", "s_kvaliteta+vrednovanje"]]
    old = old.rename(columns={"s_kvaliteta": "kvaliteta", "s_kvaliteta+vrednovanje": "kvaliteta+vrednovanje"})
    x = b[["signal_date", "security_id", "adt", "mcap"]].copy()
    x["sector"] = b.FUND_EODHD_GIC_SECTOR.fillna("nepoznato")
    x["size"] = np.log(b.mcap)
    mb = b[P + "MARKET_BOOK_TTM_PROXY"]
    x["bm"] = (1.0 / mb).where(mb > 0)
    x["zalihe (pad dana)"] = -b[P + "INVENTORY_TIE_UP_DAYS_TTM_PROXY_DIRECTION_3Y"]
    x = x.merge(sc, on=["signal_date", "security_id"], how="left").merge(old, on=["signal_date", "security_id"], how="left")
    R = pd.read_parquet(CACHE / "returns.parquet")
    cost = CFG["execution"]["cost_per_side"]
    base = simulate({s: list(g.security_id) for s, g in x.groupby("signal_date")}, R, R.index, cost)[0]
    x = x.join(controls(x, R, base.pct_change()))
    return x, R, base, cost


def main() -> None:
    x, R, base, cost = load()
    rows = []
    for v in SCORES:
        # 1) razvrstavanje
        if x[v].nunique() > 10:
            for label, frac in FRACS:
                top, bot = {}, {}
                for s, g in x.dropna(subset=[v]).groupby("signal_date"):
                    g = g.sort_values([v, "adt"], ascending=False)
                    k = max(int(round(len(g) * frac)), 1)
                    top[s], bot[s] = list(g.security_id.head(k)), list(g.security_id.tail(k))
                L, S = simulate(top, R, R.index, cost)[0], simulate(bot, R, R.index, cost)[0]
                for per, a, e in PERIODS:
                    ls, lo = ls_stats(L, S, base, a, e), stats(L, a, e)
                    rows.append({"ocjena": v, "metoda": "razvrstavanje " + label, "razdoblje": per,
                                 "long_prinos": lo["godisnji_prinos"], "long_omjer": lo["prinos_kroz_volatilnost"], "long_max_pad": lo["max_pad"],
                                 "godisnje": ls["razlika_godisnje"], "volatilnost": ls["volatilnost"], "sharpe": ls["sharpe"],
                                 "t": ls["t"], "max_pad": ls["max_pad"], "beta": ls["beta_prema_bazi"]})
        # 2) regresija
        for label, full in [("regresija jednostavna", False), ("regresija puna", True)]:
            W = {s: fmp_weights(g, v, full) for s, g in x.groupby("signal_date")}
            W = {s: w for s, w in W.items() if len(w)}
            r, turn = fmp_returns(W, R, cost)
            for per, a, e in PERIODS:
                rows.append({"ocjena": v, "metoda": label, "razdoblje": per, **ret_stats(r, base, a, e), "promet": turn})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "barra_i_razvrstavanje.csv", index=False)
    pd.set_option("display.width", 260)
    for per in ["cijelo", "od_2011"]:
        o = out[out.razdoblje == per]
        print("\n" + per + ": Sharpe long-short")
        print(o.pivot(index="ocjena", columns="metoda", values="sharpe").reindex(SCORES).round(2).to_string())
        print("\n" + per + ": godisnji prinos long-short (%)")
        print((o.pivot(index="ocjena", columns="metoda", values="godisnje").reindex(SCORES) * 100).round(1).to_string())
    o = out[out.razdoblje == "cijelo"]
    for c in ["max_pad", "volatilnost", "beta", "t"]:
        print("\ncijelo: " + c)
        print(o.pivot(index="ocjena", columns="metoda", values=c).reindex(SCORES).round(2).to_string())
    print("\ncijelo: long only (gornji kvantil), godisnji prinos % / prinos kroz volatilnost / najveci pad")
    lo = o[o.metoda.str.startswith("razvrst")]
    for c in ["long_prinos", "long_omjer", "long_max_pad"]:
        print(lo.pivot(index="ocjena", columns="metoda", values=c).reindex(SCORES).round(3).to_string())
    print("baza:", {k: round(v, 3) for k, v in stats(base).items()})
    print("promet regresijskih portfelja (godisnje, na bruto 2):")
    print(o[o.metoda.str.startswith("regres")].pivot(index="ocjena", columns="metoda", values="promet").reindex(SCORES).round(1).to_string())


if __name__ == "__main__":
    main()
