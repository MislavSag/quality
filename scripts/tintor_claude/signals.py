"""Signalni presjeci: za svaki datum signala samo izvjestaji objavljeni do tog trenutka.

Izlaz:
  data/tintor_claude/signals.parquet     jedan redak po (datum signala, izdavatelj)
  data/tintor_claude/dividends.parquet   procijenjene dividende po dionici (iz novcanog toka)
  results/tintor_claude/coverage_by_signal.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tintor as T  # noqa: E402
from fundamentals import FLOW_CF, FLOW_IS, HRK_PER_EUR, STOCK  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
RES = ROOT / "results" / "tintor_claude"
CFG = json.loads((ROOT / "config" / "tintor_claude.json").read_text(encoding="utf-8"))
FLOWS = FLOW_IS + FLOW_CF


def load_filings() -> pd.DataFrame:
    f = pd.read_parquet(DATA / "filings.parquet")
    f = f[f.is_pod & f.bs_ok & f.ticker.notna() & f.published.notna() & f.period_end.notna()].copy()
    # nedostajuca stavka u popunjenom obrascu znaci nulu za ove retke
    for c in ["capex", "div_paid", "interest_exp", "tax", "da", "da_cf", "inventories", "lt_liab", "provisions"]:
        for s in ["", "_prior"]:
            if c + s in f:
                f[c + s] = f[c + s].fillna(0)
    for s in ["", "_prior"]:
        # skraceni obrazac nema zbirni redak dobiti: izvodi se iz ukupnih prihoda i rashoda
        f["ebt" + s] = f["ebt" + s].fillna(f["total_rev" + s] - f["total_exp" + s])
        f["net_income" + s] = f["net_income" + s].fillna(f["ebt" + s] - f["tax" + s])
    conv = np.where(f.currency.eq("HRK"), 1 / HRK_PER_EUR, 1.0)
    num = [c for c in f.columns if c in STOCK + FLOWS or c.endswith("_prior")]
    f[num] = f[num].mul(conv, axis=0)
    return f


def signal_dates(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    out = []
    for y in range(cal[0].year, cal[-1].year + 1):
        for m in CFG["signal"]["months"]:
            t = pd.Timestamp(y, m, CFG["signal"]["day_of_month"])
            i = cal.searchsorted(t)
            if i < len(cal) - 1:
                out.append(cal[i])
    return out


def pick_statements(f: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Po izdavatelju: najnoviji javno dostupan izvjestaj i TTM tokovi."""
    av = f[f.published <= cutoff]
    recent = av[av.period_end >= cutoff - pd.Timedelta(days=460)]
    cons = set(recent[recent.consolidated].ticker)
    av = av[av.consolidated == av.ticker.isin(cons)]
    rows = []
    for tk, g in av.groupby("ticker"):
        g = g.sort_values(["period_end", "published"])
        L = g.iloc[-1]
        rec = {"ticker": tk, "eho_id": L.eho_id, "period_end": L.period_end, "ptype": L.ptype,
               "published": L.published, "consolidated": bool(L.consolidated), "audited": bool(L.audited),
               "ttm_ok": True, "ttm_note": ""}
        for c in STOCK:
            rec[c] = L[c]
        if L.ptype in ("FY", "Q4"):
            for c in FLOWS:
                rec[c] = L[c]
        else:
            A = g[(g.fy == L.fy - 1) & g.ptype.isin(["FY", "Q4"])]
            if A.empty:
                rec["ttm_ok"], rec["ttm_note"] = False, "nema prethodnog godisnjeg izvjestaja"
                for c in FLOWS:
                    rec[c] = np.nan
            else:
                A = A.sort_values("published").iloc[-1]
                k = 1.0
                base, ref = A.total_assets, L.total_assets_prior
                if base and ref and base == base and ref == ref and base > 0 and ref > 0:
                    ratio = ref / base
                    if 0.75 <= ratio <= 1.33:
                        k = 1.0
                    elif 750 <= ratio <= 1333:
                        k, rec["ttm_note"] = 1000.0, "godisnji u tisucama"
                    elif 0.00075 <= ratio <= 0.001333:
                        k, rec["ttm_note"] = 0.001, "kvartalni u tisucama"
                    else:
                        rec["ttm_ok"], rec["ttm_note"] = False, f"nesklad aktive prethodne godine ({ratio:.3g})"
                for c in FLOWS:
                    rec[c] = A[c] * k + L[c] - L[c + "_prior"] if rec["ttm_ok"] else np.nan
                rec["eho_id_annual"] = A.eho_id
        rows.append(rec)
    return pd.DataFrame(rows)


def indicators(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["A"], x["VK"] = x.total_assets, x.equity
    x["TK"] = x.A - x.VK
    x["ND"] = x.net_income
    x["BD"] = T.bruto_dobit(x.ebt, x.interest_exp).values
    x["SCF"] = T.scf(x.cfo, x.capex).values
    x["neg_equity"] = x.VK <= 0
    x["r_vk"] = T.r_vk(x.ND, x.VK).values
    x["roi"] = T.roi(x.BD, x.A).values
    x["scf_a"] = T.roi(x.SCF, x.A).values
    x["fp"] = T.fp(x.TK, x.VK).values
    x["kl"] = T.kl(x.current_assets, x.st_liab).values
    x["p1"] = T.p1(x.VK, x.fixed_assets).values
    x["p2"] = T.p2(x.VK, x.lt_liab, x.fixed_assets).values
    x["altman_z"] = T.altman_z(x.current_assets - x.st_liab, x.retained, x.BD, x.VK, x.TK, x.total_rev, x.A).values
    x["kralicek_df"] = T.kralicek_df(x.ND + x.da, x.TK, x.A, x.BD, x.op_rev, x.inventories).values
    return x


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    f = load_filings()
    p = pd.read_parquet(DATA / "prices.parquet")
    cal = pd.DatetimeIndex(sorted(p.date.unique()))
    master = pd.DataFrame(json.loads((DATA / "raw" / "securities_master.json").read_text(encoding="utf-8"))["securities"])
    master = master[master.security_class == "EQTY"].sort_values("segment_listing_date").drop_duplicates("isin", keep="last")
    shares_now = master.set_index("isin").listed_quantity.astype(float)
    sector = master.set_index("isin").sector_id

    # izdavatelj <-> ISIN preko korijena oznake (ADRS, ADRS2 -> ADRS)
    tickers = set(f.ticker)
    roots = p[["isin", "root"]].drop_duplicates()
    roots["ticker"] = [r if r in tickers else (r.rstrip("0123456789") if r.rstrip("0123456789") in tickers else None)
                       for r in roots.root]
    isin2tk = roots.dropna(subset=["ticker"]).drop_duplicates("isin").set_index("isin").ticker
    p["ticker"] = p["isin"].map(isin2tk)
    p["traded"] = p.num_trades > 0
    p = p.sort_values(["isin", "date"])
    n = CFG["universe"]["liquidity_lookback_days"]
    g = p.groupby("isin")
    p["adt"] = g.turnover.transform(lambda s: s.rolling(n, min_periods=int(n * 0.8)).mean())
    p["tds"] = g.traded.transform(lambda s: s.rolling(n, min_periods=int(n * 0.8)).mean())
    p["shares"] = p["isin"].map(shares_now) / p.adj
    p["mcap_cls"] = p.close * p.shares

    u = CFG["universe"]
    snaps, cover, liq_all = [], [], []
    for S in signal_dates(cal):
        cutoff = S + pd.Timedelta(hours=CFG["signal"]["publication_cutoff_hour"])
        day = p[p.date == S]
        listed = len(day)
        nonfin = day[~day["isin"].map(sector).fillna("").str.startswith(tuple(u["exclude_sectors_prefix"]))]
        liq = nonfin[(nonfin.adt >= u["min_avg_daily_turnover_eur"]) & (nonfin.tds >= u["min_traded_day_share"])]
        mcap = day.groupby("ticker").mcap_cls.sum()
        # svi likvidni nefinancijski papiri (i bez izvjestaja na EHO-u), jedna klasa po korijenu oznake
        la = liq.assign(r=liq.root.str.rstrip("0123456789")).sort_values("adt").drop_duplicates("r", keep="last")
        liq_all.append(la[["isin", "symbol", "adt"]].assign(signal_date=S, has_issuer_link=la.ticker.notna().values))
        # jedna (najlikvidnija) klasa dionica po izdavatelju
        liq1 = liq.dropna(subset=["ticker"]).sort_values("adt").drop_duplicates("ticker", keep="last")
        st = pick_statements(f, cutoff)
        if st.empty:  # prije prvih objava na EHO-u nema izvjestaja
            continue
        x = liq1[["ticker", "isin", "symbol", "close", "close_adj", "adt", "tds"]].merge(st, on="ticker", how="left")
        x["mcap"] = x.ticker.map(mcap)
        x = indicators(x)
        x["age_days"] = (S - x.period_end).dt.days
        x["fresh"] = x.age_days <= CFG["signal"]["max_statement_age_days"]
        x["scf_mc"] = (x.SCF / x.mcap.where(x.mcap > 0))
        x["signal_date"] = S
        snaps.append(x)
        cover.append({"signal_date": S, "listed_equities": listed, "non_financial": len(nonfin),
                      "liquid_nonfin_isins": len(liq), "liquid_no_issuer_link": int(liq.ticker.isna().sum()),
                      "liquid_issuers": len(liq1), "with_statement": int(x.eho_id.notna().sum()),
                      "fresh_statement": int((x.eho_id.notna() & x.fresh).sum()),
                      "ttm_ok": int((x.fresh & x.ttm_ok.fillna(False).astype(bool)).sum()),
                      "neg_equity": int((x.fresh & x.neg_equity).sum())})
    s = pd.concat(snaps, ignore_index=True)
    s.to_parquet(DATA / "signals.parquet", index=False)
    pd.concat(liq_all, ignore_index=True).to_parquet(DATA / "liquid_universe.parquet", index=False)
    pd.DataFrame(cover).to_csv(RES / "coverage_by_signal.csv", index=False)

    # dividende: isplate iz novcanog toka (nekonsolidirano ima prednost), po tromjesecjima
    d = f[f.ptype.notna()].copy()
    d["q"] = d.ptype.map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4})
    d = d.sort_values(["consolidated", "published"]).drop_duplicates(["ticker", "fy", "q", "consolidated"], keep="last")
    d = d.sort_values("consolidated").drop_duplicates(["ticker", "fy", "q"], keep="first")
    w = d.pivot(index=["ticker", "fy"], columns="q", values="div_paid").reindex(columns=[1, 2, 3, 4])
    w = w.ffill(axis=1).fillna(0)
    inc = w.diff(axis=1)
    inc[1] = w[1]
    inc = inc.clip(lower=0).stack().rename("amount").reset_index()
    inc = inc[inc.amount > 0]
    inc["pay_date"] = [pd.Timestamp(int(y), {1: 2, 2: 5, 3: 8, 4: 11}[q], 15) for y, q in zip(inc.fy, inc.q)]
    inc["pay_date"] = [cal[min(cal.searchsorted(t), len(cal) - 1)] for t in inc.pay_date]
    tk_shares = (p.drop_duplicates("isin").assign(sh=lambda z: z["isin"].map(shares_now))
                 .groupby("ticker").sh.sum())
    inc["shares_adj"] = inc.ticker.map(tk_shares)
    inc["dps_adj"] = inc.amount / inc.shares_adj
    mc = p.groupby(["ticker", "date"]).mcap_cls.sum().rename("mcap").reset_index()
    inc = inc.merge(mc, left_on=["ticker", "pay_date"], right_on=["ticker", "date"], how="left").drop(columns="date")
    inc["yield"] = inc.amount / inc.mcap
    cap = CFG["dividends"]["max_yield_per_payment"]
    inc["capped"] = inc["yield"] > cap
    inc.loc[inc.capped, "dps_adj"] *= cap / inc.loc[inc.capped, "yield"]
    inc = inc.dropna(subset=["dps_adj", "mcap"])
    inc.to_parquet(DATA / "dividends.parquet", index=False)

    c = pd.DataFrame(cover)
    print(c.iloc[::4].to_string(index=False))
    print("dividend payments", len(inc), "capped", int(inc.capped.sum()),
          "median yield", round(float(inc["yield"].median()), 4))


if __name__ == "__main__":
    main()
