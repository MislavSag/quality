"""Kratak HTML izvjestaj + mapa formula iz rezultata backtesta.

Izlaz: results/tintor_claude/report.html, formula_map.md, nav_final.png
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
RES = ROOT / "results" / "tintor_claude"
CFG = json.loads((ROOT / "config" / "tintor_claude.json").read_text(encoding="utf-8"))
SERIES = {"kvaliteta": "#2a78d6", "kvaliteta+vrednovanje": "#eb6834", "ew_svi": "#1baf7a", "CROBEXtr": "#eda100"}
NAMES = {"kvaliteta": "Kvaliteta (8 dionica)", "kvaliteta+vrednovanje": "Kvaliteta + vrednovanje",
         "ew_svi": "Jednako ponderirani univerzum", "CROBEXtr": "CROBEXtr"}

PCT = ["godisnji_neto_prinos", "volatilnost", "max_pad", "godisnji_promet", "prosjecan_udio_gotovine",
       "udio_neizvrsenog", "troskovi_godisnje"]
HEAD = {"varijanta": "Varijanta", "godisnji_neto_prinos": "Godišnji neto prinos", "volatilnost": "Volatilnost",
        "max_pad": "Najveći pad", "godisnji_promet": "Godišnji promet", "prosjecan_broj_pozicija": "Pozicija (prosjek)",
        "prosjecan_udio_gotovine": "Gotovina (prosjek)", "udio_neizvrsenog": "Neizvršeni dio naloga",
        "od": "Od", "do": "Do"}


def fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    d = df[cols].copy()
    for c in cols:
        if c in PCT:
            d[c] = d[c].map(lambda v: "" if v != v else f"{100 * v:.1f} %".replace(".", ","))
        elif c == "prosjecan_broj_pozicija":
            d[c] = d[c].map(lambda v: "" if v != v else f"{v:.1f}".replace(".", ","))
    d = d.rename(columns=HEAD)
    return d.to_html(index=False, border=0, classes="t", escape=True)


def chart(nav: pd.DataFrame, period: str) -> str:
    a, b = CFG["periods"]["final" if period == "zavrsno" else "development"]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    idx = pd.read_csv(DATA / "raw" / "index_CROBEXtr.csv", parse_dates=["date"]).sort_values("date").set_index("date").last_value
    idx = idx[(idx.index >= a) & (idx.index <= b)]
    lines = {}
    for k in ["kvaliteta", "kvaliteta+vrednovanje", "ew_svi"]:
        s = nav[f"{k}|{period}"].dropna()
        lines[k] = 100 * s / s.iloc[0]
    if len(idx) > 10:
        start = lines["kvaliteta"].reindex(idx.index, method="ffill").iloc[0]
        lines["CROBEXtr"] = start * idx / idx.iloc[0]
    for k, s in lines.items():
        ax.plot(s.index, s.values, color=SERIES[k], lw=2, label=NAMES[k])
    # izravne oznake na kraju linija, razmaknute da se ne preklapaju
    ends = sorted(((s.iloc[-1], k) for k, s in lines.items()))
    gap = 0.045 * (max(s.max() for s in lines.values()) - min(s.min() for s in lines.values()))
    y_prev = -1e9
    for y, k in ends:
        y_lab = max(y, y_prev + gap)
        y_prev = y_lab
        ax.annotate(NAMES[k], (lines[k].index[-1], y_lab), xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color="#52514e", va="center")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    ax.set_ylabel("Vrijednost portfelja (početak = 100)", fontsize=8, color="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.78, left=0.07, top=0.96, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    (RES / f"nav_{period}.png").write_bytes(buf.getvalue())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def formula_map() -> str:
    am = pd.read_csv(RES / "aop_map.csv")
    top = am.groupby("sig").files_with_sig.first().sort_values(ascending=False)
    def aops(concept: str) -> str:
        q = am[am.concept == concept].sort_values("files_with_sig", ascending=False).head(3)
        return "; ".join(f"{r.sig.split('/')[0]} → {r.aops}" for r in q.itertuples())
    rows = [
        ("R_VK = 100 × ND / VK", "1032 (PDF 525)", "ND: dobit ili gubitak razdoblja (RDG), TTM. VK: A) Kapital i rezerve (bilanca), stanje",
         f"ND {aops('net_income')} | VK {aops('equity')}", "VK ≤ 0 → nedefinirano; rang 0 (najlošiji). ND nedostaje → izdavatelj bez ocjene"),
        ("RoI = 100 × BD / A", "1032 (PDF 525); BD definiran na 327 (PDF 164)",
         "BD = dobit prije oporezivanja + rashodi kamata (TTM). A: ukupno aktiva, stanje",
         f"EBT {aops('ebt')} | kamate {aops('interest_exp')} | A {aops('total_assets')}",
         "A ≤ 0 → nedefinirano. PRILAGODBA: obrasci do 2018. nemaju čiste kamate nego „kamate, tečajne razlike i drugi rashodi”"),
        ("SCF = CFO − kapitalni izdaci", "338–339 (PDF 170), formula 5.2",
         "CFO: neto novčani tok od poslovnih aktivnosti (TTM). Izdaci: kupnja dugotrajne materijalne i nematerijalne imovine (TTM)",
         f"CFO {aops('cfo_single')} / {aops('cfo_up')} − {aops('cfo_down')} | capex {aops('capex')}",
         "Prazan redak izdataka = 0. PRILAGODBA ZA STRATEGIJU: SCF / A (usporedivost) i SCF / tržišna kapitalizacija (vrednovanje)"),
        ("Fp = TK / VK", "1032 (PDF 525)", "TK = A − VK (tablica 6.1: A = VK + TK). Stanje",
         f"A {aops('total_assets')} | VK {aops('equity')}", "VK ≤ 0 → nedefinirano; rang 0. Niža poluga = bolji rang"),
        ("KL = TI / KO", "1032 (PDF 525)", "TI: C) Kratkotrajna imovina. KO: D) Kratkoročne obveze. Stanje",
         f"TI {aops('current_assets')} | KO {aops('st_liab')}", "KO ≤ 0 → nedefinirano. Za rang ograničeno na 3,0"),
        ("%P² = 100 × (VK + DO) / DA", "554–555 (PDF 279), formula 7.62", "DO: C) Dugoročne obveze. DA: B) Dugotrajna imovina. Stanje",
         f"DO {aops('lt_liab')} | DA {aops('fixed_assets')}", "DA ≤ 0 → nedefinirano. Za rang ograničeno na 2,0. KL i P² dijele jedan glas (isti rizik ročne usklađenosti)"),
        ("Z = 1,2 NRK/A + 1,4 ZD/A + 3,3 BD/A + 0,6 VK/TK + 1,0 UP/A", "394 (PDF 198), formula 6.1",
         "NRK = TI − KO. ZD: zadržana dobit ili preneseni gubitak (bilanca). UP: ukupni prihodi (TTM)",
         f"ZD {aops('retained')} | UP {aops('total_rev')}", "TK ≤ 0 ili A ≤ 0 → nedefinirano. Knjižni primjer (tabl. 6.2) u 2. članu koristi ND/A; test reproducira 1,88 / 1,64 / 1,89 / 2,24"),
        ("DF = 1,5x₁ + 0,08x₂ + 10x₃ + 5x₄ + 0,3x₅ + 0,1x₆", "395 (PDF 198), formula 6.2",
         "x₁ = (ND + amortizacija)/TK; x₂ = A/TK; x₃ = BD/A; x₄ = BD/poslovni prihodi; x₅ = zalihe/poslovni prihodi; x₆ = poslovni prihodi/A",
         f"amortizacija {aops('da')} | poslovni prihodi {aops('op_rev')} | zalihe {aops('inventories')}",
         "PRILAGODBA: čisti novčani tok = ND + amortizacija (CF = dobitak + amortizacija, PDF 166). Nazivnik ≤ 0 → nedefinirano"),
    ]
    md = ["# Mapa Tintorovih pokazatelja na stavke TFI-POD / GFI-POD obrazaca\n",
          "Izvor formula: J. Tintor, *Poslovna analiza* (otisnuta stranica; PDF stranica fotografija). "
          "Sve stavke su u valuti izvještaja (HRK za razdoblja do 31.12.2022., EUR nakon toga) i preračunate u EUR po 7,5345; "
          "jedinica je valuta bez tisuća (kad godišnji i tromjesečni obrazac nisu u istoj jedinici, omjer aktive to otkriva i ispravlja). "
          "Tokovi su zadnjih 12 mjeseci (TTM = prethodna godina + tekući kumulativ − lanjski kumulativ iz istog obrasca), stanja su s kraja zadnjeg objavljenog razdoblja. "
          "AOP oznake su naučene iz naziva redaka po inačici obrasca (potpis = list:raspon AOP-a); puna mapa je u `aop_map.csv`.\n",
          "**„Bruto dobit” (BD).** Tintor na str. 327 definira BD kao dodanu vrijednost umanjenu za zarade radnika, tj. EBIT: "
          "BD − kamate = dobit (EBT), EBT − porez = ND. To nije računovodstvena bruto marža niti dobit prije poreza. "
          "U obrascu takav redak ne postoji, pa je BD = dobit prije oporezivanja + rashodi kamata.\n",
          "| Pokazatelj | Stranica | Ulazi i razdoblje | AOP po inačici obrasca (list:raspon → AOP) | Nedostajući podaci, nulti/negativni nazivnik, prilagodbe |",
          "|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    md.append("\nInačice obrasca po učestalosti (hrvatske datoteke): " + ", ".join(f"`{k}` ({v})" for k, v in top.head(10).items()) + ".\n")
    (RES / "formula_map.md").write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md)


def main() -> None:
    fin = RES / "final"
    s = pd.read_csv(fin / "summary.csv")
    nav = pd.read_csv(fin / "nav.csv", index_col=0, parse_dates=True)
    cov = pd.read_csv(RES / "coverage_by_signal.csv", parse_dates=["signal_date"])
    formula_map()
    cols = ["varijanta", "godisnji_neto_prinos", "volatilnost", "max_pad", "godisnji_promet",
            "prosjecan_broj_pozicija", "prosjecan_udio_gotovine", "udio_neizvrsenog"]
    def block(group: list[str], period: str, names: list[str] | None = None) -> str:
        q = s[s.skupina.isin(group) & (s.razdoblje == period)]
        if names:
            q = q.set_index("varijanta").reindex(names).dropna(how="all").reset_index()
        return fmt_table(q, cols)
    main_names = ["kvaliteta", "kvaliteta+vrednovanje", "ew_svi", "CROBEXtr", "CROBEX"]
    abl = ["kvaliteta", "bez_rentabilnosti", "bez_novcanog_toka", "bez_otpornosti", "samo_vrednovanje", "altman", "kralicek"]
    nodiv = ["kvaliteta (bez dividendi)", "kvaliteta+vrednovanje (bez dividendi)", "ew_svi (bez dividendi)", "CROBEX"]
    img = chart(nav, "zavrsno")
    c2 = cov[cov.signal_date >= "2011-05-01"].copy()
    c2["godina"] = c2.signal_date.dt.year
    cy = c2.groupby("godina")[["listed_equities", "non_financial", "liquid_issuers", "liquid_no_issuer_link",
                               "fresh_statement", "ttm_ok", "neg_equity"]].mean().round(1).reset_index()
    cy.columns = ["Godina", "Uvrštene dionice", "Nefinancijske", "Likvidni izdavatelji", "Likvidni bez veze na EHO",
                  "Sa svježim izvještajem", "S potpunim TTM-om", "Negativan kapital"]
    extra = (RES / "report_notes.html").read_text(encoding="utf-8") if (RES / "report_notes.html").exists() else ""
    html = f"""<!doctype html><html lang="hr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tintor ZSE strategija</title><style>
body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#f9f9f7;line-height:1.5}}
h1{{font-size:1.5rem}}h2{{font-size:1.15rem;margin-top:2rem}}
.t{{border-collapse:collapse;font-size:.85rem;width:100%;margin:.5rem 0 1rem}}.t th,.t td{{padding:4px 8px;border-bottom:1px solid #e4e3df;text-align:right}}
.t th:first-child,.t td:first-child{{text-align:left}}.wrap{{overflow-x:auto}}img{{max-width:100%}}.n{{color:#52514e;font-size:.85rem}}
</style></head><body>
<h1>Fundamentalna strategija prema Tintorovoj <em>Poslovnoj analizi</em> na Zagrebačkoj burzi</h1>
{extra}
<h2>Završno razdoblje ({CFG['periods']['final'][0]} – {CFG['periods']['final'][1]}), neto, s procijenjenim dividendama</h2>
<div class="wrap">{block(['osnovno', 'indeks'], 'zavrsno', main_names)}</div>
<img alt="Vrijednost portfelja u završnom razdoblju" src="data:image/png;base64,{img}">
<p class="n">Svaka serija kreće sa 100 na početku završnog razdoblja. CROBEXtr je indeks ukupnog prinosa bez troškova trgovanja.</p>
<h2>Isto bez dividendi (usporedivo s cjenovnim CROBEX-om)</h2>
<div class="wrap">{block(['bez dividendi', 'indeks'], 'zavrsno', nodiv)}</div>
<h2>Doprinos skupina pokazatelja, Altman i Kralicek (završno razdoblje)</h2>
<div class="wrap">{block(['osnovno'], 'zavrsno', abl)}</div>
<h2>Osjetljivost na troškove, kapital i veličinu portfelja (završno razdoblje)</h2>
<div class="wrap">{block(['osjetljivost'], 'zavrsno')}</div>
<h2>Razvojno razdoblje ({CFG['periods']['development'][0]} – {CFG['periods']['development'][1]})</h2>
<div class="wrap">{block(['osnovno', 'indeks'], 'razvoj', main_names + abl[1:])}</div>
<h2>Pokrivenost podataka po godinama (prosjek četiri datuma signala)</h2>
<div class="wrap">{cy.to_html(index=False, border=0, classes='t')}</div>
</body></html>"""
    (RES / "report.html").write_text(html, encoding="utf-8")
    print("report written", RES / "report.html")


if __name__ == "__main__":
    main()
