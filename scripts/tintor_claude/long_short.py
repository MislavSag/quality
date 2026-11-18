"""Long-short inacice Tintorovih ocjena: kupnja najbolje, prodaja najlosije ocijenjenih.

SAD: gornja naprema donjoj petini i 50 naprema 50 dionica, trosak 0,10 % po strani na obje noge.
ZSE: 8 naprema 8 dionica, zatvaranje-zatvaranje s procijenjenim dividendama, trosak 0,80 % po strani.
Kratka prodaja na ZSE-u u praksi nije izvediva; taj dio je teorijska razlika, ne strategija.
Trosak posudbe dionica nije uracunat. Izlaz: results/tintor_claude_us/long_short.csv i
results/tintor_claude/long_short.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import Market  # noqa: E402
from us import CACHE, CFG, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERIODS = [("cijelo", None, None), ("do_2018", None, "2018-12-31"), ("od_2019", "2019-01-01", None)]


def ls_stats(long: pd.Series, short: pd.Series, base: pd.Series, a, b) -> dict:
    x = pd.concat([long, short, base], axis=1, keys=["l", "s", "m"]).loc[a:b].pct_change().dropna()
    d = x.l - x.s
    cum = (1 + d).cumprod()
    m = (1 + x).resample("ME").prod() - 1
    dm = m.l - m.s
    beta = np.cov(dm, m.m)[0, 1] / m.m.var()
    return {"long": x.l.mean() * 252, "short": x.s.mean() * 252, "razlika_godisnje": d.mean() * 252,
            "volatilnost": d.std() * np.sqrt(252), "sharpe": d.mean() / d.std() * np.sqrt(252),
            "t": dm.mean() / dm.std() * np.sqrt(len(dm)), "max_pad": (cum / cum.cummax() - 1).min(),
            "beta_prema_bazi": beta, "alfa_godisnje": (dm.mean() - beta * m.m.mean()) * 12, "mjeseci": len(dm)}


def legs(sc: pd.DataFrame, col: str, key: str, n: int | None, q: int | None) -> tuple[dict, dict]:
    top, bot = {}, {}
    for s, x in sc.dropna(subset=[col]).groupby("signal_date"):
        x = x.sort_values([col, "adt"], ascending=False)
        k = n if n else len(x) // q
        if len(x) < 2 * k:
            continue
        top[s], bot[s] = list(x[key].head(k)), list(x[key].tail(k))
    return top, bot


def run(sc, cols, key, R, cal, cost, specs, base_targets) -> pd.DataFrame:
    base = simulate(base_targets, R, cal, cost)[0]
    rows = []
    for name, col in cols.items():
        for label, n, q in specs:
            top, bot = legs(sc, col, key, n, q)
            L, S = simulate(top, R, cal, cost)[0], simulate(bot, R, cal, cost)[0]
            for per, a, b in PERIODS:
                rows.append({"varijanta": name, "portfelj": label, "razdoblje": per, **ls_stats(L, S, base, a, b)})
    return pd.DataFrame(rows)


def us() -> pd.DataFrame:
    sc = pd.read_parquet(ROOT / "results/tintor_claude_us/scores.parquet")
    R = pd.read_parquet(CACHE / "returns.parquet")
    cols = {"kvaliteta": "s_kvaliteta", "kvaliteta+vrednovanje": "s_kvaliteta+vrednovanje",
            "altman": "s_altman", "kralicek": "s_kralicek"}
    base = {s: list(x.security_id) for s, x in sc.groupby("signal_date")}
    return run(sc, cols, "security_id", R, R.index, CFG["execution"]["cost_per_side"],
               [("petine", None, 5), ("50/50", 50, None)], base)


def us_tails() -> pd.DataFrame:
    """Gornji i donji rep po ocjeni (20, 10, 5 i 1 %): prinosi strana, razlika i asimetrija mjesecnih prinosa."""
    sc = pd.read_parquet(ROOT / "results/tintor_claude_us/scores.parquet")
    R = pd.read_parquet(CACHE / "returns.parquet")
    cost = CFG["execution"]["cost_per_side"]
    base = simulate({s: list(x.security_id) for s, x in sc.groupby("signal_date")}, R, R.index, cost)[0]
    rows = []
    for name in ["kvaliteta", "kvaliteta+vrednovanje", "altman", "kralicek"]:
        col = "s_" + name
        for frac in [0.20, 0.10, 0.05, 0.01]:
            top, bot, n = {}, {}, []
            for s, x in sc.dropna(subset=[col]).groupby("signal_date"):
                x = x.sort_values([col, "adt"], ascending=False)
                k = max(int(round(len(x) * frac)), 1)
                top[s], bot[s] = list(x.security_id.head(k)), list(x.security_id.tail(k))
                n.append(k)
            L, S = simulate(top, R, R.index, cost)[0], simulate(bot, R, R.index, cost)[0]
            for per, a, b in PERIODS:
                x = pd.concat([L, S, base], axis=1, keys=["l", "s", "m"]).loc[a:b]
                yrs = (x.index[-1] - x.index[0]).days / 365.25
                cagr = (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1
                m = x.resample("ME").last().pct_change().dropna()
                d = x.pct_change().dropna()
                rows.append({"varijanta": name, "rep": f"{frac:.0%}", "razdoblje": per, "dionica_po_strani": np.mean(n),
                             "long_cagr": cagr.l, "short_cagr": cagr.s, "baza_cagr": cagr.m,
                             "long_max_pad": (x.l / x.l.cummax() - 1).min(), "short_max_pad": (x.s / x.s.cummax() - 1).min(),
                             "long_vol": d.l.std() * np.sqrt(252), "short_vol": d.s.std() * np.sqrt(252),
                             "long_skew_mj": m.l.skew(), "short_skew_mj": m.s.skew(), "baza_skew_mj": m.m.skew(),
                             **{k: v for k, v in ls_stats(L, S, base, a, b).items()
                                if k in ("razlika_godisnje", "t", "max_pad", "beta_prema_bazi", "alfa_godisnje")},
                             "razlika_skew_mj": (m.l - m.s).skew()})
    return pd.DataFrame(rows)


def zse() -> pd.DataFrame:
    m = Market()
    close = m.close.ffill()
    div = m.div.unstack().reindex(index=close.index, columns=close.columns).fillna(0.0)
    R = ((close + div) / close.shift(1) - 1).astype("float64")
    R[R.abs() > 0.6] = np.nan   # neispravljeni skokovi cijene ne smiju stvoriti prividnu razliku
    rk = pd.read_csv(ROOT / "results/tintor_claude/final/ranks__kvaliteta.csv", parse_dates=["signal_date"])
    rv = pd.read_csv(ROOT / "results/tintor_claude/final/ranks__kvaliteta_vrednovanje.csv", parse_dates=["signal_date"])
    rk = rk.merge(rv[["signal_date", "isin", "score"]].rename(columns={"score": "score_v"}), on=["signal_date", "isin"])
    rk = rk[rk.signal_date.isin(R.index) & (rk.signal_date < R.index[-3])]
    cols = {"kvaliteta": "score", "kvaliteta+vrednovanje": "score_v", "altman": "altman_z", "kralicek": "kralicek_df"}
    base = {s: list(x["isin"]) for s, x in rk.groupby("signal_date")}
    return run(rk, cols, "isin", R, R.index, 0.008, [("8/8", 8, None)], base)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    u = us()
    u.to_csv(ROOT / "results/tintor_claude_us/long_short.csv", index=False)
    print(u.round(3).to_string())
    z = zse()
    z.to_csv(ROOT / "results/tintor_claude/long_short.csv", index=False)
    print(z.round(3).to_string())
