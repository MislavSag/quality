"""Inacice ocjene kvalitete iz Tintorovih pokazatelja (dnevna tablica characteristics projekta), SAD.

Ulaz: data/tintor_claude/us/book_features.parquet (us_book_extract.py). Skup dionica, datumi signala,
izvrsenje i trosak su isti kao u us.py. Smjer svakog pokazatelja zadan je unaprijed prema knjizi (SIGN),
osim u inacici ic_2010, gdje se smjer i izbor uce na signalima do kraja 2010. i ocjenjuju od 2011.
Izlaz: results/tintor_claude_us_book/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_short import legs, ls_stats  # noqa: E402
from us import CACHE, CFG, active_t, simulate, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "tintor_claude_us_book"
P = "FUND_TINTOR_"
LEARN_END = pd.Timestamp("2010-12-31")
PERIODS = [("cijelo", None, None), ("do_2010", None, "2010-12-31"), ("od_2011", "2011-01-01", None),
           ("do_2018", None, "2018-12-31"), ("od_2019", "2019-01-01", None)]

# Stilovi po poglavljima knjige; +1 = vise je bolje, -1 = manje je bolje.
STYLES = {
    "zarada": {  # 9. Analiza snage zaradivanja
        "ROI_TTM": 1, "ROE_TTM": 1, "OPERATING_RETURN_ON_ASSETS_TTM": 1, "QUICK_NET_RETURN_TTM": 1,
        "OPERATING_RETURN_INVESTMENT_TTM": 1, "EBIT_SALES_TTM_PROXY": 1, "EBDIT_SALES_TTM_PROXY": 1,
        "NET_MARGIN_TTM_PROXY": 1, "PROFIT_REVENUE_MARGIN_TTM_PROXY": 1, "REGULAR_MARGIN_TTM_PROXY": 1,
        "TOTAL_ECONOMY_TTM_PROXY": 1, "OPERATING_ECONOMY_TTM_PROXY": 1, "EBIT_GROSS_ASSETS_TTM_PROXY": 1,
        "EBDIT_GROSS_ASSETS_TTM_PROXY": 1, "LEVERAGE_SPREAD_TTM_PROXY": 1},
    "kredit": {  # 10. Analiza kreditne sposobnosti (novcani tokovi i otplata)
        "CFO_TO_LIABILITIES_TTM_PROXY": 1, "QUICK_CF_REVENUE_TTM_PROXY": 1, "CFO_SALES_TTM_PROXY": 1,
        "CFO_RETURN_ON_ASSETS_TTM_PROXY": 1, "CFO_GROSS_ASSETS_TTM_PROXY": 1, "CFO_INTEREST_COVERAGE_TTM_PROXY": 1,
        "CFO_CAPEX_COVERAGE_TTM_PROXY": 1, "CASH_REALIZATION_QUALITY_TTM_PROXY": 1,
        "CASH_GENERATION_RATE_TTM_PROXY": 1, "CASH_GAP_TTM_PROXY": -1, "TRADE_POSITION_TTM_PROXY": -1,
        "QUICK_DEBT_PAYBACK_TTM": -1},
    "financiranje": {  # 7. Analiza financijske situacije (investiranje i financiranje)
        "EQUITY_SHARE_TTM": 1, "LONG_DEBT_TOTAL_CAPITAL_TTM": -1, "SHORT_DEBT_TOTAL_CAPITAL_TTM": -1,
        "INTEREST_INTENSITY_REVENUE_TTM_PROXY": -1, "CURRENT_RATIO_TTM": 1, "QUICK_RATIO_TTM": 1,
        "CASH_RATIO_TTM": 1, "WORKING_CAPITAL_SHARE_TTM": 1, "COVERAGE_P1_TTM": 1, "COVERAGE_P2_TTM": 1,
        "COVERAGE_P3_CLOSING_INVENTORY_TTM_PROXY": 1, "PPE_FUNCTIONALITY_TTM": 1,
        "PPE_RENEWAL_RATE_BRIDGE_TTM_PROXY": 1, "PPE_INVESTMENT_RATE_BRIDGE_TTM_PROXY": 1},
    "proizvodnja": {  # 8. Analiza proizvodnje - samo zamjenski pokazatelji iskoristenja i ekonomicnosti
        "ASSET_TURNOVER_TTM_PROXY": 1, "CURRENT_ASSET_TURNOVER_TTM_PROXY": 1, "INVENTORY_DAYS_COGS_TTM_PROXY": -1,
        "RECEIVABLE_DAYS_AVERAGE_TTM_PROXY": -1, "COGS_INTENSITY_TTM_PROXY": -1,
        "OPERATING_EXPENSE_INTENSITY_TTM_PROXY": -1, "REVENUE_GROWTH_TTM_PROXY": 1,
        "INVENTORY_GROWTH_MINUS_SALES_TTM_PROXY": -1, "AR_GROWTH_MINUS_SALES_TTM_PROXY": -1},
}
SIGN = {k: v for s in STYLES.values() for k, v in s.items()}
NEG_EQUITY_WORST = ["ROE_TTM", "COVERAGE_P1_TTM"]


def pct_rank(b: pd.DataFrame, col: pd.Series) -> pd.Series:
    return col.groupby(b.signal_date).rank(pct=True)


def signed_ranks(b: pd.DataFrame, sign: dict, suffix: str = "") -> pd.DataFrame:
    """Percentilni rang unutar datuma, okrenut tako da je 1 uvijek najbolje."""
    out = {}
    neg_eq = b[P + "EQUITY_SHARE_TTM"] <= 0
    for f, s in sign.items():
        v = b[P + f + suffix].astype("float64")
        if f == "QUICK_DEBT_PAYBACK_TTM":   # godine otplate imaju smisla samo uz pozitivan novcani tok
            v = v.where(b[P + "SIMPLE_CF_TTM"] > 0)
        r = pct_rank(b, s * v)
        if f == "QUICK_DEBT_PAYBACK_TTM":
            r = r.where(~(b[P + "SIMPLE_CF_TTM"] <= 0), 0.0)
        if f in NEG_EQUITY_WORST:
            r = r.where(~neg_eq, 0.0)
        out[f] = r
    return pd.DataFrame(out, index=b.index)


def wmean(S: pd.DataFrame, w: pd.Series, min_share: float = 0.5) -> pd.Series:
    w = w.reindex(S.columns).to_numpy()
    A = S.to_numpy()
    ok = np.isfinite(A)
    have = (ok * w).sum(axis=1)
    val = np.where(ok, A * w, 0.0).sum(axis=1) / np.where(have > 0, have, np.nan)
    return pd.Series(np.where(have >= min_share * w.sum(), val, np.nan), index=S.index)


def learn_corr(b: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Korelacija rangova pokazatelja, samo na signalima do kraja 2010."""
    return S[b.signal_date <= LEARN_END].corr(min_periods=500)


def greedy_keep(C: pd.DataFrame, order: list[str], limit: float = 0.7) -> list[str]:
    keep: list[str] = []
    for f in order:
        if all(abs(C.loc[f, k]) < limit for k in keep):
            keep.append(f)
    return keep


def similarity_weights(C: pd.DataFrame) -> pd.Series:
    w = 1.0 / (C.fillna(0.0) ** 2).sum(axis=1)
    return w / w.sum()


def forward_returns(b: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    """Prinos od zatvaranja sjednice nakon signala do zatvaranja sjednice nakon iduceg signala."""
    cum = (1.0 + R.fillna(0.0)).cumprod()
    sig = sorted(b.signal_date.unique())
    out = pd.Series(np.nan, index=b.index)
    for a, nxt in zip(sig[:-1], sig[1:]):
        i, j = R.index.get_loc(a) + 1, R.index.get_loc(nxt) + 1
        m = b.signal_date == a
        ids = b.loc[m, "security_id"]
        out[m] = (cum.iloc[j].reindex(ids).to_numpy() / cum.iloc[i].reindex(ids).to_numpy()) - 1.0
    return out


def ic_table(b: pd.DataFrame, feats: list[str], fwd: pd.Series) -> pd.DataFrame:
    m = (b.signal_date <= LEARN_END) & fwd.notna()
    x = b.loc[m, [P + f for f in feats]].astype("float64")
    x.columns = feats
    d, y = b.loc[m, "signal_date"], fwd[m]
    yr = y.groupby(d).rank()
    ics = {}
    for s, idx in d.groupby(d).groups.items():
        xs = x.loc[idx].rank()
        ics[s] = xs.corrwith(yr.loc[idx])
    ic = pd.DataFrame(ics).T
    n = ic.notna().sum()
    return pd.DataFrame({"ic": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(n), "datuma": n})


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    b = pd.read_parquet(CACHE / "book_features.parquet")
    b = b[b.book_ok].reset_index(drop=True)
    R = pd.read_parquet(CACHE / "returns.parquet")
    cal, cost, n = R.index, CFG["execution"]["cost_per_side"], CFG["portfolio"]["top_n"]

    S = signed_ranks(b, SIGN)
    has3 = {f: s for f, s in SIGN.items() if P + f + "_MEAN_3Y" in b.columns}   # nemaju svi trogodisnji prosjek
    S3 = signed_ranks(b, has3, "_MEAN_3Y")
    sc = b[["signal_date", "security_id", "ticker", "adt"]].copy()

    # 1. Altman i Kralicek
    z, df = b[P + "Z_BOOK_TTM_PROXY"], b[P + "DF_TTM_PROXY"]
    sc["altman"], sc["kralicek"] = pct_rank(b, z), pct_rank(b, df)
    sc["altman+kralicek"] = sc[["altman", "kralicek"]].mean(axis=1, skipna=False)
    d3 = (b[[P + "Z_BOOK_TTM_PROXY_DIRECTION_3Y", P + "DF_TTM_PROXY_DIRECTION_3Y"]].mean(axis=1, skipna=False) + 1) / 2
    lvl3 = pd.concat([pct_rank(b, b[P + "Z_BOOK_TTM_PROXY_MEAN_3Y"]), pct_rank(b, b[P + "DF_TTM_PROXY_MEAN_3Y"])], axis=1).mean(axis=1, skipna=False)
    sc["altman+kralicek 3g i smjer"] = (2 * lvl3 + d3) / 3

    # 2. Stilovi po poglavljima
    for name, feats in STYLES.items():
        sc["stil " + name] = wmean(S[list(feats)], pd.Series(1.0, index=list(feats)))
    style_cols = ["stil " + k for k in STYLES]
    sc["4 stila"] = sc[style_cols].mean(axis=1, skipna=False)
    g3 = [[f for f in fs if f in has3] for fs in STYLES.values()]
    print("pokazatelja s trogodisnjim prosjekom po stilu:", [len(g) for g in g3], "od", [len(f) for f in STYLES.values()])
    s3 = pd.concat([wmean(S3[g], pd.Series(1.0, index=g)) for g in g3], axis=1).mean(axis=1, skipna=False)
    sc["4 stila, prosjek 3g"] = s3

    # 3. i 4. Bez koreliranih, odnosno manji ponder slicnima
    C = learn_corr(b, S)
    order = list(S.notna().mean().sort_values(ascending=False).index)
    kept = greedy_keep(C, order)
    sc["bez koreliranih"] = wmean(S[kept], pd.Series(1.0, index=kept))
    w = similarity_weights(C)
    sc["ponder slicnosti"] = wmean(S, w)

    # 5a. Tintorov postupak: prvo izbaciti financijski ugrozene, zatim rangirati po stilovima
    safe = (z >= 1.81) & (df > 0) & (b[P + "EQUITY_SHARE_TTM"] > 0)
    sc["filtar ugrozenih + 4 stila"] = sc["4 stila"].where(safe)

    # 5b. Izbor i smjer nauceni do 2010., ocjena od 2011.
    meta = pd.read_csv("H:/strategies/characteristics_tintor_book_ttm_daily_us_20261117/tintor_book_ttm_predictor_metadata.csv")
    cand = [f[len(P):] for f in meta[meta.role == "predictor"].feature
            if f in b.columns and "MARKET_BOOK" not in f and "EARNINGS_MULTIPLE" not in f]
    fwd = forward_returns(b, R)
    ic = ic_table(b, cand, fwd)
    ic.sort_values("t").to_csv(RES / "ic_do_2010.csv")
    sel = ic[(ic.t.abs() >= 3) & (ic.datuma >= 30)]
    Sic = signed_ranks(b, {f: int(np.sign(sel.ic[f])) for f in sel.index})
    wic = similarity_weights(learn_corr(b, Sic))
    sc["ic_2010"] = wmean(Sic, wic)

    variants = [c for c in sc.columns if c not in ("signal_date", "security_id", "ticker", "adt")]
    base_t = {s: list(x.security_id) for s, x in sc.groupby("signal_date")}
    base = simulate(base_t, R, cal, cost)[0]
    rows, qrows, navs = [], [], {"baza": base}
    for per, a, e in PERIODS:
        rows.append({"varijanta": "baza (svi, jednaki ponderi)", "razdoblje": per, **stats(base, a, e)})
    for v in variants:
        tg = {s: list(x.dropna(subset=[v]).sort_values([v, "adt"], ascending=False).security_id.head(n))
              for s, x in sc.groupby("signal_date")}
        nav, turn, _ = simulate(tg, R, cal, cost)
        navs[v] = nav
        top, bot = legs(sc, v, "security_id", None, 5)
        L, Sh = simulate(top, R, cal, cost)[0], simulate(bot, R, cal, cost)[0]
        for per, a, e in PERIODS:
            m, t, _ = active_t(nav, base, a, e)
            ls = ls_stats(L, Sh, base, a, e)
            x = pd.concat([nav, base], axis=1).loc[a:e].pct_change().dropna()
            act = x.iloc[:, 0] - x.iloc[:, 1]
            rows.append({"varijanta": v, "razdoblje": per, **stats(nav, a, e), "promet": turn,
                         "aktivni_prinos": act.mean() * 252, "info_omjer": act.mean() / act.std() * np.sqrt(252),
                         "petine_razlika": ls["razlika_godisnje"], "petine_sharpe": ls["sharpe"],
                         "petine_max_pad": ls["max_pad"], "petina_gornja": ls["long"], "petina_donja": ls["short"]})
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "summary.csv", index=False)
    pd.DataFrame(navs).to_parquet(RES / "nav.parquet")
    sc.to_parquet(RES / "scores.parquet")
    C.to_csv(RES / "korelacije_do_2010.csv")
    pd.DataFrame({"ponder_slicnosti": w, "zadrzan_bez_koreliranih": w.index.isin(kept),
                  "stil": [next(k for k, v in STYLES.items() if f in v) for f in w.index],
                  "smjer": [SIGN[f] for f in w.index]}).to_csv(RES / "pokazatelji.csv")
    sel.assign(ponder=wic).sort_values("ponder", ascending=False).to_csv(RES / "ic_2010_izbor.csv")
    sc[variants].corr(method="spearman").to_csv(RES / "korelacije_ocjena.csv")

    pd.set_option("display.width", 260)
    cols = ["godisnji_prinos", "prinos_kroz_volatilnost", "max_pad", "info_omjer", "petine_razlika", "petine_sharpe", "petine_max_pad"]
    for per in ["cijelo", "od_2011", "od_2019"]:
        print("\n", per)
        print(summ[summ.razdoblje == per].set_index("varijanta")[cols].round(3).to_string())
    print("\nzadrzano bez koreliranih:", len(kept), "od", len(SIGN), kept)
    print("ic_2010 odabrano:", len(sel), "od", len(cand))
    print(sel.assign(ponder=wic).sort_values("ponder", ascending=False).round(3).head(40).to_string())
    print(sc.groupby("signal_date")[variants].count().iloc[::16].T.to_string())


if __name__ == "__main__":
    main()
