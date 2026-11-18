"""AOP redci -> stavke po izvjestaju (jedan redak po objavi) s vremenom objave.

Mapiranje stavki uci se iz hrvatskih naziva redaka po inacici obrasca
(potpis = vrsta lista + raspon AOP oznaka), a zatim se primjenjuje po AOP broju
na sve datoteke iste inacice (ukljucujuci engleske). Izlaz:
  data/tintor_claude/filings.parquet        stavke po objavi (izvorna valuta i EUR)
  results/tintor_claude/aop_map.csv         naucena mapa stavka -> AOP po inacici
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch  # noqa: E402
from parse_reports import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
RES = ROOT / "results" / "tintor_claude"
HRK_PER_EUR = 7.5345

# stavka -> (list, regex nad normaliziranim nazivom, nacin: first|sum|all)
CONCEPTS = {
    # bilanca
    "unpaid_capital": ("BS", r"^a\)? ?potrazivanja za upisani", "first"),
    "fixed_assets": ("BS", r"^b\)? ?dugotrajna imovina", "first"),
    "current_assets": ("BS", r"^c\)? ?kratkotrajna imovina", "first"),
    "inventories": ("BS", r"^i\. ?zalihe", "first"),
    "cash": ("BS", r"novac u banci i blagajni", "first"),
    "prepaid": ("BS", r"^d\)? ?placeni troskovi buduceg", "first"),
    "total_assets": ("BS", r"^[a-g]\)? ?ukupno ?[-–]? ?aktiva", "first"),
    "equity": ("BS", r"^a\)? ?kapital i rezerve", "first"),
    "retained": ("BS", r"^[ivx]+\. ?zadrzana dobit", "first"),
    "profit_year_bs": ("BS", r"dobit ili gubitak poslovne godine", "first"),
    "provisions": ("BS", r"^b\)? ?rezerviranja", "first"),
    "lt_liab": ("BS", r"^c\)? ?dugorocne obveze", "first"),
    "st_liab": ("BS", r"^d\)? ?kratkorocne obveze", "first"),
    "accruals": ("BS", r"^e\)? ?odgodeno placanje troskova", "first"),
    "total_liab_eq": ("BS", r"^[a-g]\)? ?ukupno ?[-–]? ?pasiva", "first"),
    # racun dobiti i gubitka
    "op_rev": ("IS", r"^i\. ?poslovni prihodi", "first"),
    "op_exp": ("IS", r"^ii\. ?poslovni rashodi", "first"),
    "da": ("IS", r"^\d\. ?amortizacija", "first"),
    "fin_exp": ("IS", r"^iv\. ?financijski rashodi", "first"),
    "interest_exp": ("IS", r"(kamate, tecajne razlike i drugi rashodi|rashodi s osnove kamata i slicni rashodi)", "sum"),
    "total_rev": ("IS", r"^[ivx]+\. ?ukupni prihodi", "first"),
    "total_exp": ("IS", r"^[ivx]+\. ?ukupni rashodi", "first"),
    "ebt": ("IS", r"^[ivx]+\. ?dobit ili gubitak prije oporezivanja", "first"),
    "tax": ("IS", r"^[ivx]+\. ?porez na dobit", "first"),
    "net_income": ("IS", r"^[ivx]+\. ?dobit ili gubitak razdoblja", "first"),
    # novcani tok (indirektna ili direktna metoda)
    "cfo_single": ("CF", r"^a\)? ?neto novcani tokovi od poslovnih aktivnosti", "first"),
    "cfo_up": ("CF", r"neto povecanje novcanog t(ije|o)ka od poslovnih", "first"),
    "cfo_down": ("CF", r"neto smanjenje novcanog t(ije|o)ka od poslovnih", "first"),
    "capex": ("CF", r"novcani izdaci za kupnju dugotrajne materijalne i nematerijalne", "first"),
    "ppe_sales": ("CF", r"novcani primici od prodaje dugotrajne materijalne i nematerijalne", "first"),
    "div_paid": ("CF", r"novcani izdaci za isplatu dividendi", "first"),
    "da_cf": ("CF", r"^(\d\.|[a-z]\)) ?amortizacija", "first"),
}
FLOW_IS = ["op_rev", "op_exp", "da", "fin_exp", "interest_exp", "total_rev", "total_exp", "ebt", "tax", "net_income"]
FLOW_CF = ["cfo", "capex", "ppe_sales", "div_paid", "da_cf"]
STOCK = ["unpaid_capital", "fixed_assets", "current_assets", "inventories", "cash", "prepaid", "total_assets",
         "equity", "retained", "profit_year_bs", "provisions", "lt_liab", "st_liab", "accruals", "total_liab_eq"]

PERIODS = {
    "Prvi kvartal": ("Q1", "03-31"), "Drugi kvartal": ("Q2", "06-30"), "Polugodišnji izvještaj": ("Q2", "06-30"),
    "Treći kvartal": ("Q3", "09-30"), "Četvrti kvartal": ("Q4", "12-31"), "Godišnji izvještaj": ("FY", "12-31"),
}


def load_meta() -> pd.DataFrame:
    m = fetch.all_meta()
    m = m[m.tip_dokumenta.isin(["XLS", "XLSX"])].copy()
    m["published"] = pd.to_datetime(m.vrijeme_objave, format="%d.%m.%Y. %H:%M", errors="coerce")
    m["fy"] = m.za_godinu.str.extract(r"(\d{4})")[0].astype(float)
    m["ticker"] = m.izdavatelj.str.extract(r"\(([A-Z0-9]+)\)\s*$")[0]
    m["ptype"] = m.razdoblje.map(lambda x: PERIODS.get(x, (None, None))[0])
    m["period_end"] = pd.to_datetime(
        [f"{int(y)}-{PERIODS[p][1]}" if p in PERIODS and y == y else None for y, p in zip(m.fy, m.razdoblje)])
    m["consolidated"] = m.status_konsolidacije.eq("Konsolidirano")
    m["audited"] = m.status_revizije.eq("Revidirano")
    return m


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(DATA / "aop_rows.parquet")
    rows["kind2"] = rows.kind.replace({"CF_I": "CF", "CF_D": "CF"})
    rows["ln"] = rows.label.map(norm)
    meta = load_meta()
    lang = meta.set_index("eho_id").na_jeziku

    # potpis inacice obrasca po listu
    sig = rows.groupby(["eho_id", "sheet", "kind"]).aop.agg(["min", "max", "size"]).reset_index()
    sig["sig"] = sig.kind + ":" + sig["min"].astype(str) + "-" + sig["max"].astype(str) + "/" + sig["size"].astype(str)
    rows = rows.merge(sig[["eho_id", "sheet", "sig"]], on=["eho_id", "sheet"])
    rows["hr"] = rows.eho_id.map(lang).eq("Hrvatski")

    # ucenje mape: po potpisu i stavci vecinski AOP (ili skup AOP-a) iz hrvatskih naziva
    learned = []
    for c, (kind, rx, mode) in CONCEPTS.items():
        q = rows[(rows.kind2 == kind) & rows.hr & rows.ln.str.contains(rx, regex=True)]
        if mode == "first":
            per = q.sort_values("aop").groupby(["sig", "eho_id"]).aop.first().map(lambda a: (a,))
        else:
            per = q.sort_values("aop").groupby(["sig", "eho_id"]).aop.agg(tuple)
        for s, g in per.groupby(level=0):
            vc = g.value_counts()
            learned.append({"sig": s, "concept": c, "aops": vc.index[0], "n_files": int(vc.iloc[0]),
                            "agreement": float(vc.iloc[0] / vc.sum())})
    amap = pd.DataFrame(learned)
    n_sig = rows[rows.hr].groupby("sig").eho_id.nunique().rename("files_with_sig")
    amap = amap.merge(n_sig, on="sig")
    amap = amap[(amap.n_files >= 3) & (amap.agreement >= 0.8)]
    amap.assign(aops=amap.aops.map(lambda t: "+".join(map(str, t)))).sort_values(["sig", "concept"]).to_csv(
        RES / "aop_map.csv", index=False)

    # primjena mape
    long = []
    for r in amap.itertuples():
        for a in r.aops:
            long.append((r.sig, r.concept, a))
    long = pd.DataFrame(long, columns=["sig", "concept", "aop"])
    x = rows.merge(long, on=["sig", "aop"])
    # stupci s brojevima: stupac se racuna ako je popunjen barem 40 % kao najpuniji (rijetki stupac = oznaka biljeske)
    cnt = rows.groupby(["eho_id", "sheet"])[["v0", "v1", "v2", "v3"]].count()
    use = cnt.ge(0.4 * cnt.max(axis=1), axis=0) & cnt.gt(3)
    order = use.apply(lambda r: [c for c in ["v0", "v1", "v2", "v3"] if r[c]], axis=1).rename("cols").reset_index()
    x = x.merge(order, on=["eho_id", "sheet"])
    V = x[["v0", "v1", "v2", "v3"]].to_numpy()
    ci = {"v0": 0, "v1": 1, "v2": 2, "v3": 3}
    cur, prior = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for j, (kind, cols) in enumerate(zip(x.kind.to_numpy(), x.cols.to_numpy())):
        n = len(cols)
        if n == 0:
            continue
        if n == 1:
            cur[j] = V[j, ci[cols[0]]]
            continue
        prior[j] = V[j, ci[cols[0]]]
        # RDG s cetiri stupca: prethodno kumulativ, prethodno tromjesecje, tekuce kumulativ, tekuce tromjesecje
        cur[j] = V[j, ci[cols[2]]] if (kind == "IS" and n == 4) else V[j, ci[cols[1]]]
    x["cur"], x["prior"] = cur, prior
    # indirektna metoda ima prednost pred direktnom
    x["pref"] = x.kind.map({"BS": 0, "IS": 0, "CF_I": 0, "CF_D": 1})
    # izdavatelji koji koriste direktnu metodu ostavljaju indirektni list popunjen nulama: takav list se ne koristi
    filled = (rows[rows.kind == "CF_I"].assign(a=lambda d: d[["v0", "v1"]].abs().sum(axis=1))
              .groupby("eho_id").a.sum())
    empty_ind = set(filled[filled == 0].index)
    x.loc[x.kind.eq("CF_I") & x.eho_id.isin(empty_ind), "pref"] = 2
    agg = (x.groupby(["eho_id", "kind", "pref", "concept"])[["cur", "prior"]].sum(min_count=1).reset_index()
           .sort_values("pref").drop_duplicates(["eho_id", "concept"]))
    cur = agg.pivot(index="eho_id", columns="concept", values="cur")
    pri = agg.pivot(index="eho_id", columns="concept", values="prior").add_suffix("_prior")
    f = cur.join(pri)
    for sfx in ["", "_prior"]:
        single = f.get("cfo_single" + sfx)
        up, down = f.get("cfo_up" + sfx), f.get("cfo_down" + sfx)
        cfo = single.copy() if single is not None else pd.Series(np.nan, index=f.index)
        if up is not None:
            alt = up.fillna(0).abs() - down.fillna(0).abs()
            alt[up.isna() & down.isna()] = np.nan
            cfo = cfo.fillna(alt)
        f["cfo" + sfx] = cfo
        for c in ["capex", "div_paid"]:
            if c + sfx in f:
                f[c + sfx] = f[c + sfx].abs()
    f = f.drop(columns=[c for c in f.columns if c.startswith(("cfo_single", "cfo_up", "cfo_down"))])

    f = f.join(meta.set_index("eho_id")[["ticker", "izdavatelj", "fy", "ptype", "period_end", "consolidated",
                                          "audited", "published", "na_jeziku", "link"]], how="left")
    f = f.reset_index()
    # razdoblje iz samog obrasca ima prednost kad je jasno (1.1. - kraj tromjesecja iste godine) i razlicito od EHO oznake
    hd = pd.read_parquet(DATA / "report_headers.parquet")[["eho_id", "period_start", "period_end"]]
    hd = hd.rename(columns={"period_start": "hdr_start", "period_end": "hdr_end"})
    f = f.merge(hd, on="eho_id", how="left")
    qend = f.hdr_end.dt.is_quarter_end & (f.hdr_start.dt.month == 1) & (f.hdr_start.dt.day == 1)         & (f.hdr_start.dt.year == f.hdr_end.dt.year) & (f.hdr_end.dt.year == f.fy)
    # zaglavlje vrijedi samo ako je objava nakon kraja tog razdoblja (inace je to nepopunjen predlozak)
    lag = (f.published - f.hdr_end).dt.days
    fix = qend.fillna(False) & (f.hdr_end != f.period_end) & lag.between(0, 150)
    f["period_fixed"] = fix
    new_pt = f.hdr_end.dt.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    f.loc[fix & ~(f.ptype.eq("FY") & new_pt.eq("Q4")), "ptype"] = new_pt
    f.loc[fix, "period_end"] = f.hdr_end
    # obrazac poduzetnika (TFI-POD/GFI-POD): ima dugotrajnu i kratkotrajnu imovinu te poslovne prihode
    f["is_pod"] = f.fixed_assets.notna() & f.current_assets.notna() & f.op_rev.notna()
    # kontrole
    f["bs_gap"] = (f.total_assets - f.total_liab_eq).abs() / f.total_assets.abs()
    f["bs_ok"] = f.bs_gap < 0.005
    # valuta izvjestaja: HRK za razdoblja do 31.12.2022., EUR nakon toga
    f["currency"] = np.where(f.period_end <= pd.Timestamp("2022-12-31"), "HRK", "EUR")
    f.to_parquet(DATA / "filings.parquet", index=False)

    print("filings", len(f), "pod", int(f.is_pod.sum()), "bs_ok among pod", round(float(f[f.is_pod].bs_ok.mean()), 4))
    print("unmapped (no total_assets):", int(f.total_assets.isna().sum()))
    print(f[f.is_pod][STOCK + FLOW_IS + FLOW_CF].notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
