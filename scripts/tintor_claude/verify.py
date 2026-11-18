"""Rucna provjera: od redaka izvornog izvjestaja (AOP, naziv, iznos) do pokazatelja i ranga.

Za zadani datum signala i izdavatelje ispisuje izvorne retke koristenih obrazaca,
ponovno racuna TTM i pokazatelje neovisno o signals.py te ih usporeduje s rangovima.
Izlaz: results/tintor_claude/manual_verification.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
RES = ROOT / "results" / "tintor_claude"
HRK = 7.5345

# stavka -> dio naziva retka (trazi se u izvornom nazivu, neovisno o naucenoj AOP mapi)
ROWS = {
    "BS": {"A": "UKUPNO AKTIVA|UKUPNO – AKTIVA|UKUPNO - AKTIVA", "VK": "KAPITAL I REZERVE", "DI": "DUGOTRAJNA IMOVINA (",
           "TI": "KRATKOTRAJNA IMOVINA (", "DO": "DUGOROČNE OBVEZE (", "KO": "KRATKOROČNE OBVEZE ("},
    "IS": {"UP": "UKUPNI PRIHODI", "EBT": "DOBIT ILI GUBITAK PRIJE OPOREZIVANJA", "ND": "DOBIT ILI GUBITAK RAZDOBLJA",
           "KAM": "Kamate, tečajne razlike i drugi rashodi|Rashodi s osnove kamata i slični rashodi"},
    "CF": {"CFO+": "NETO NOVČANI TOKOVI OD POSLOVNIH AKTIVNOSTI|NETO POVEĆANJE NOVČANOG TIJEKA OD POSLOVNIH",
           "CFO-": "NETO SMANJENJE NOVČANOG TIJEKA OD POSLOVNIH",
           "CAPEX": "Novčani izdaci za kupnju dugotrajne materijalne i nematerijalne"},
}


def grab(rows: pd.DataFrame, eid: str) -> tuple[dict, list[str]]:
    r = rows[rows.eho_id == eid]
    out, lines = {}, []
    for kind, items in ROWS.items():
        kk = ["CF_I", "CF_D"] if kind == "CF" else [kind]
        for key, pat in items.items():
            got = None
            for k in kk:
                q = r[(r.kind == k) & r.label.str.contains("|".join(map(lambda s: s.replace("(", r"\("), pat.split("|"))),
                                                           case=False, regex=True)]
                # indirektni list popunjen nulama (izdavatelj koristi direktnu metodu) se preskace
                if len(q) and (kind != "CF" or r[r.kind == k][["v0", "v1"]].abs().sum().sum() > 0):
                    got = q
                    break
            if got is None:
                continue
            if key == "KAM":
                rowsel = got
            else:
                rowsel = got.sort_values("aop").head(1)
            vals = rowsel[["v0", "v1", "v2", "v3"]].to_numpy(dtype=float)
            nn = [[v for v in row if v == v] for row in vals]
            four = kind == "IS" and any(len(x) == 4 for x in nn)
            cur = sum((x[2] if four and len(x) == 4 else (x[1] if len(x) > 1 else (x[0] if x else 0.0))) for x in nn)
            pri = sum((x[0] if len(x) > 1 else 0.0) for x in nn)
            out[key] = (cur, pri)
            for _, w in rowsel.iterrows():
                vv = ", ".join("" if v != v else f"{v:,.0f}" for v in [w.v0, w.v1, w.v2, w.v3])
                lines.append(f"| {w.sheet} | {int(w.aop):03d} | {w.label.strip()[:70]} | {vv} |")
    return out, lines


def main() -> None:
    S = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else None
    variant = "kvaliteta"
    folder = RES / "final" if (RES / "final" / f"ranks__{variant}.csv").exists() else RES / "dev"
    ranks = pd.read_csv(folder / f"ranks__{variant}.csv", parse_dates=["signal_date", "period_end", "published"])
    rows = pd.read_parquet(DATA / "aop_rows.parquet")
    fil = pd.read_parquet(DATA / "filings.parquet").set_index("eho_id")
    if S is None:
        S = ranks.signal_date.max()
    x = ranks[ranks.signal_date == S].sort_values("rank")
    want = sys.argv[2].split(",") if len(sys.argv) > 2 else list(x.ticker.iloc[[0, 3, len(x) // 2, len(x) - 1]])
    md = [f"# Ručna provjera: od izvornog izvještaja do ranga\n",
          f"Datum signala **{S.date()}**, varijanta `{variant}`, podobnih izdavatelja u presjeku: {len(x)}. "
          "Iznosi u tablicama redaka su u izvornoj valuti obrasca (stupci desno od AOP oznake, redom kako su u datoteci). "
          "Kontrolni izračun ispod tablice ne koristi naučenu AOP mapu nego traži retke po nazivu.\n"]
    worst = 0.0
    for tk in want:
        r = x[x.ticker == tk]
        if r.empty:
            md.append(f"## {tk}\nNije u podobnom presjeku na taj datum.\n")
            continue
        r = r.iloc[0]
        eid = str(int(r.eho_id)) if not isinstance(r.eho_id, str) else r.eho_id
        F = fil.loc[eid]
        cur, lines = grab(rows, eid)
        fx = HRK if F.currency == "HRK" else 1.0
        md.append(f"## {tk} ({r.symbol}), rang {int(r['rank'])} od {len(x)}{' — ODABRAN' if r.selected else ''}\n")
        md.append(f"Zadnji javno dostupan izvještaj: EHO {eid}, razdoblje {F.ptype} {int(F.fy)} (do {F.period_end.date()}), "
                  f"objavljen {F.published}, {'konsolidiran' if F.consolidated else 'nekonsolidiran'}, valuta {F.currency}. "
                  f"Izvor: https://eho.zse.hr{F.link}\n")
        md.append("| List | AOP | Naziv retka | Iznosi (stupci) |\n|---|---|---|---|")
        md += lines
        flows = {k: cur[k][0] for k in cur}
        ann = None
        if F.ptype not in ("FY", "Q4"):
            aid = r.get("eho_id_annual")
            aid = str(int(aid)) if aid == aid else None
            if aid:
                FA = fil.loc[aid]
                ann, lines2 = grab(rows, aid)
                fxa = HRK if FA.currency == "HRK" else 1.0
                md.append(f"\nPrethodni godišnji izvještaj za TTM: EHO {aid}, {FA.ptype} {int(FA.fy)}, objavljen {FA.published}, "
                          f"valuta {FA.currency}. Izvor: https://eho.zse.hr{FA.link}\n")
                md.append("| List | AOP | Naziv retka | Iznosi (stupci) |\n|---|---|---|---|")
                md += lines2
        def ttm(key: str) -> float:
            c, p = cur.get(key, (0.0, 0.0))
            if ann is None:
                return c / fx
            return ann.get(key, (0.0, 0.0))[0] / fxa + (c - p) / fx
        A, VK = cur["A"][0] / fx, cur["VK"][0] / fx
        ND, EBT, KAM, UP = ttm("ND"), ttm("EBT"), ttm("KAM"), ttm("UP")
        if "ND" not in cur:
            ND = np.nan
        CFO = ttm("CFO+") - (abs(ttm("CFO-")) if "CFO-" in cur else 0.0) if "CFO-" in cur else ttm("CFO+")
        CAPEX = abs(ttm("CAPEX"))
        BD, SCF = EBT + KAM, CFO - CAPEX
        calc = {"r_vk": ND / VK if VK > 0 else np.nan, "roi": BD / A, "scf_a": SCF / A,
                "fp": (A - VK) / VK if VK > 0 else np.nan, "kl": cur["TI"][0] / cur["KO"][0],
                "p2": (cur["VK"][0] + cur.get("DO", (0, 0))[0]) / cur["DI"][0]}
        md.append(f"\nTTM (EUR): ND = {ND:,.0f}; dobit prije oporezivanja = {EBT:,.0f}; kamate = {KAM:,.0f}; "
                  f"BD = {BD:,.0f}; CFO = {CFO:,.0f}; kapitalni izdaci = {CAPEX:,.0f}; SCF = {SCF:,.0f}. "
                  f"Stanje (EUR): A = {A:,.0f}; VK = {VK:,.0f}.\n")
        md.append("| Pokazatelj | Kontrolni izračun | Cjevovod | Razlika | Rang (0–1) |\n|---|---|---|---|---|")
        for k, lab in [("r_vk", "R_VK = ND/VK"), ("roi", "RoI = BD/A"), ("scf_a", "SCF/A"), ("fp", "Fp = TK/VK"),
                       ("kl", "KL = TI/KO"), ("p2", "P² = (VK+DO)/DA")]:
            pv = r[k]
            diff = abs(calc[k] - pv) if calc[k] == calc[k] and pv == pv else np.nan
            if diff == diff:
                worst = max(worst, diff / max(abs(pv), 1e-9))
            md.append(f"| {lab} | {calc[k]:.4f} | {pv:.4f} | {diff:.2e} | {r['rk_' + k]:.3f} |")
        md.append(f"\nSkupine: rentabilnost {r.g_rentabilnost:.3f}, novčani tok {r.g_novcani_tok:.3f}, "
                  f"otpornost {r.g_otpornost:.3f} → ocjena {r.score:.3f}, rang {int(r['rank'])}.\n")
    md.append(f"\nNajveće relativno odstupanje kontrolnog izračuna od cjevovoda: {worst:.2e}.\n")
    (RES / "manual_verification.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)[:6000])


if __name__ == "__main__":
    main()
