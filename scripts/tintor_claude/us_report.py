"""HTML izvjestaj: ocjene kvalitete po Tintoru na americkom trzistu (long only, razvrstavanje, faktorski portfelji).

Cita gotove rezultate iz results/tintor_claude_us i results/tintor_claude_us_book; nista ne racuna iznova.
Izlaz: results/tintor_claude_us_book/izvjestaj.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "tintor_claude_us_book"
OLD = ROOT / "results" / "tintor_claude_us"

NAME = {
    "SPY": "SPY", "baza": "Baza (sve dionice, jednaki ponderi)",
    "kvaliteta": "Kvaliteta (prva verzija)", "kvaliteta+vrednovanje": "Kvaliteta + vrednovanje",
    "altman": "Altman Z", "kralicek": "Kralicek DF", "altman+kralicek": "Altman + Kralicek",
    "altman+kralicek 3g i smjer": "Altman + Kralicek, 3 godine i smjer",
    "stil zarada": "Stil: snaga zarađivanja", "stil kredit": "Stil: kreditna sposobnost",
    "stil financiranje": "Stil: investiranje i financiranje", "stil proizvodnja": "Stil: proizvodnja",
    "4 stila": "4 stila", "4 stila, prosjek 3g": "4 stila, prosjek 3 godine",
    "bez koreliranih": "Bez koreliranih", "ponder slicnosti": "Manji ponder sličnima",
    "filtar ugrozenih + 4 stila": "Filtar ugroženih + 4 stila", "zalihe (pad dana)": "Zalihe (pad dana vezivanja)",
}
SCORES = [k for k in NAME if k not in ("SPY", "baza")]
SHOW = ["kvaliteta", "kvaliteta+vrednovanje", "bez koreliranih", "stil kredit", "4 stila", "altman+kralicek", "zalihe (pad dana)"]
STYLE = {"zarada": "Snaga zarađivanja", "kredit": "Kreditna sposobnost", "financiranje": "Investiranje i financiranje", "proizvodnja": "Proizvodnja"}


def num(v: float, nd: int = 1, pct: bool = False, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    v = v * 100 if pct else v
    s = f"{abs(v):.{nd}f}".replace(".", ",")
    if round(abs(v), nd) == 0:
        return s + (" %" if pct else "")
    return ("−" if v < 0 else ("+" if sign else "")) + s + (" %" if pct else "")


def heat(v: float, scale: float) -> str:
    if v is None or not np.isfinite(v):
        return ""
    a = min(abs(v) / scale, 1.0) * 0.55
    return f' style="background:rgba({"42,120,214" if v > 0 else "227,73,72"},{a:.2f})"'


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(c if c.startswith("<t") else f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def th(s: str) -> str:
    return f'<th scope="row">{s}</th>'


def td(v: float, nd: int, pct: bool, scale: float | None = None) -> str:
    return f"<td{heat(v, scale) if scale else ''}>{num(v, nd, pct)}</td>"


def nav_stats(n: pd.Series, a=None, e=None) -> dict:
    n = n.dropna().loc[a:e]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    r = n.pct_change().dropna()
    cagr, vol = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1, r.std() * np.sqrt(252)
    return {"cagr": cagr, "vol": vol, "omjer": cagr / vol, "pad": (n / n.cummax() - 1).min()}


def main() -> None:
    old, new = pd.read_parquet(OLD / "nav.parquet"), pd.read_parquet(RES / "nav.parquet")
    nav = pd.concat([old[["SPY", "kvaliteta", "kvaliteta+vrednovanje"]], new.drop(columns=["ic_2010"])], axis=1).dropna()
    nav = nav / nav.iloc[0]
    B = pd.read_csv(RES / "barra_i_razvrstavanje.csv")
    D = pd.read_parquet(RES / "barra_dnevni_prinosi.parquet")
    ind = pd.read_csv(RES / "pokazatelji.csv", index_col=0)

    def cell(score, metoda, razdoblje, col):
        x = B[(B.ocjena == score) & (B.metoda == metoda) & (B.razdoblje == razdoblje)]
        return float(x[col].iloc[0]) if len(x) else np.nan

    # --- long only: 50 najboljih
    lo_order = ["SPY", "baza"] + [s for s in SCORES if s in nav.columns]
    t_long = []
    for k in lo_order:
        s, a, b_ = nav_stats(nav[k]), nav_stats(nav[k], None, "2010-12-31"), nav_stats(nav[k], "2011-01-01")
        t_long.append([th(NAME[k]), td(s["cagr"], 1, True), td(s["vol"], 1, True), td(s["omjer"], 2, False), td(s["pad"], 0, True),
                       td(a["cagr"], 1, True), td(b_["cagr"], 1, True)])
    base = nav_stats(nav["baza"])["cagr"]

    # --- long only po kvantilima
    t_q = []
    for k in SCORES[:-1]:
        r = [th(NAME[k])]
        for m in ["5 %", "desetine", "petine"]:
            v = cell(k, "razvrstavanje " + m, "cijelo", "long_prinos")
            r.append(f"<td{heat(v - base, 0.05)}>{num(v, 1, True)}</td>")
        r.append(td(cell(k, "razvrstavanje 5 %", "cijelo", "long_max_pad"), 0, True))
        r.append(td(cell(k, "razvrstavanje petine", "cijelo", "long_max_pad"), 0, True))
        t_q.append(r)

    # --- long-short Sharpe po metodi
    METH = ["razvrstavanje 5 %", "razvrstavanje desetine", "razvrstavanje petine", "regresija jednostavna", "regresija puna"]

    def sharpe_table(per):
        return [[th(NAME[k])] + [td(cell(k, m, per, "sharpe"), 2, False, 0.6) for m in METH] for k in SCORES]

    t_risk = []
    for m, lab in zip(METH, ["Gornjih / donjih 5 %", "Desetine", "Petine", "Regresija, samo ocjena", "Regresija, puna"]):
        x = B[(B.metoda == m) & (B.razdoblje == "cijelo")]
        t_risk.append([th(lab), td(x.sharpe.median(), 2, False), td(x.volatilnost.median(), 1, True),
                       f"<td>{num(x.max_pad.max(), 0, True)} do {num(x.max_pad.min(), 0, True)}</td>", td(x.beta.median(), 2, False)])

    # --- puna regresija, detalj
    t_full = []
    for k in sorted(SCORES, key=lambda s: -cell(s, "regresija puna", "cijelo", "sharpe")):
        c = lambda col, per="cijelo": cell(k, "regresija puna", per, col)  # noqa: E731
        t_full.append([th(NAME[k]), td(c("godisnje"), 1, True), td(c("volatilnost"), 1, True), td(c("sharpe"), 2, False, 0.6),
                       td(c("t"), 1, False), td(c("max_pad"), 0, True), td(c("beta"), 2, False), td(c("promet"), 1, False),
                       td(c("sharpe", "do_2010"), 2, False, 0.9), td(c("sharpe", "od_2011"), 2, False, 0.9)])

    # --- po godinama
    g = D.groupby(D.index.year)
    yret = g.apply(lambda z: (1 + z.fillna(0)).prod() - 1)
    ysh = g.mean() / g.std() * np.sqrt(252)
    hd_y = ["Godina"] + [NAME[k] for k in SHOW]
    t_yret = [[th(str(y))] + [td(yret.loc[y, "puna|" + k], 1, True, 0.15) for k in SHOW] for y in yret.index]
    t_ysh = [[th(str(y))] + [td(ysh.loc[y, "puna|" + k], 2, False, 2.5) for k in SHOW] for y in ysh.index]
    ny = nav.resample("YE").last()
    ny = pd.concat([nav.iloc[[0]], ny]).pct_change().dropna()
    LOY = ["SPY", "baza", "kvaliteta", "kvaliteta+vrednovanje", "4 stila", "bez koreliranih", "altman+kralicek 3g i smjer"]
    t_ylo = [[th(str(i.year)), td(r["SPY"], 1, True), td(r["baza"], 1, True)] +
             [f"<td{heat(r[k] - r['baza'], 0.12)}>{num(r[k], 1, True)}</td>" for k in LOY[2:]] for i, r in ny.iterrows()]

    # --- korelacije mjesecnih prinosa faktorskih portfelja
    M = ((1 + D[["puna|" + k for k in SHOW]].fillna(0)).resample("ME").prod() - 1).corr()
    t_corr = [[th(NAME[k])] + [td(M.loc["puna|" + k, "puna|" + j], 2, False, 1.0) for j in SHOW] for k in SHOW]

    # --- pokazatelji
    t_ind = [[th(f"<code>{i}</code>"), f"<td>{STYLE[r.stil]}</td>", f"<td>{'više je bolje' if r.smjer > 0 else 'manje je bolje'}</td>",
              f"<td>{'da' if r.zadrzan_bez_koreliranih else 'ne'}</td>", td(r.ponder_slicnosti * 100, 1, False)] for i, r in ind.iterrows()]

    # --- podaci za grafove (tjedno)
    w = nav.resample("W-FRI").last().dropna()
    c1 = {"dates": [d.strftime("%Y-%m-%d") for d in w.index], "log": True, "unit": "×", "series": [
        {"name": NAME[k].split(" (")[0] if k == "baza" else NAME[k], "color": c, "dash": dsh, "values": [round(float(v), 3) for v in w[k]]}
        for k, c, dsh in [("kvaliteta", "--s1", ""), ("kvaliteta+vrednovanje", "--s2", ""), ("4 stila", "--s3", ""),
                          ("baza", "--ink2", ""), ("SPY", "--muted", "5 4")]]}
    cum = ((1 + D.fillna(0)).cumprod() - 1).resample("W-FRI").last() * 100
    c2 = {"dates": [d.strftime("%Y-%m-%d") for d in cum.index], "log": False, "unit": " %", "vline": "2011-01-01", "series": [
        {"name": NAME[k], "color": c, "dash": "", "values": [round(float(v), 1) for v in cum["puna|" + k]]}
        for k, c in [("kvaliteta", "--s1"), ("kvaliteta+vrednovanje", "--s2"), ("4 stila", "--s3"), ("zalihe (pad dana)", "--s4")]]}

    kv = lambda col, per="cijelo", k="kvaliteta": cell(k, "regresija puna", per, col)  # noqa: E731
    html = TEMPLATE
    rep = {
        "@@DATE@@": f"{nav.index[0]:%d. %m. %Y.} – {nav.index[-1]:%d. %m. %Y.}",
        "@@T_LONG@@": table(["Portfelj", "Godišnji prinos", "Volatilnost", "Prinos / volatilnost", "Najveći pad", "Prinos do 2010.", "Prinos od 2011."], t_long),
        "@@T_Q@@": table(["Ocjena", "Gornjih 5 %", "Gornja desetina", "Gornja petina", "Najveći pad, 5 %", "Najveći pad, petina"], t_q),
        "@@T_SH@@": table(["Ocjena", "5 %", "Desetine", "Petine", "Regresija, samo ocjena", "Regresija, puna"], sharpe_table("cijelo")),
        "@@T_SH11@@": table(["Ocjena", "5 %", "Desetine", "Petine", "Regresija, samo ocjena", "Regresija, puna"], sharpe_table("od_2011")),
        "@@T_RISK@@": table(["Metoda", "Medijan Sharpea", "Medijan volatilnosti", "Najveći pad (raspon)", "Medijan bete"], t_risk),
        "@@T_FULL@@": table(["Ocjena", "Godišnje", "Volatilnost", "Sharpe", "t", "Najveći pad", "Beta", "Promet", "Sharpe do 2010.", "Sharpe od 2011."], t_full),
        "@@T_YRET@@": table(hd_y, t_yret), "@@T_YSH@@": table(hd_y, t_ysh),
        "@@T_YLO@@": table(["Godina"] + [NAME[k].split(" (")[0] if k == "baza" else NAME[k] for k in LOY], t_ylo),
        "@@T_CORR@@": table([""] + [NAME[k] for k in SHOW], t_corr),
        "@@T_IND@@": table(["Pokazatelj", "Stil", "Smjer", "Zadržan bez koreliranih", "Ponder sličnosti, %"], t_ind),
        "@@C1@@": json.dumps(c1, ensure_ascii=False), "@@C2@@": json.dumps(c2, ensure_ascii=False),
        "@@BASE@@": num(base, 1, True), "@@SPY@@": num(nav_stats(nav["SPY"])["cagr"], 1, True),
        "@@KV_G@@": num(kv("godisnje"), 1, True), "@@KV_S@@": num(kv("sharpe"), 2), "@@KV_T@@": num(kv("t"), 1),
        "@@KV_S10@@": num(kv("sharpe", "do_2010"), 2), "@@KV_S11@@": num(kv("sharpe", "od_2011"), 2),
        "@@KVV_S11@@": num(kv("sharpe", "od_2011", "kvaliteta+vrednovanje"), 2),
    }
    for k, v in rep.items():
        html = html.replace(k, v)
    assert "@@" not in html
    (RES / "izvjestaj.html").write_text(html, encoding="utf-8")
    # samostalna stranica za GitHub Pages
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text('<!doctype html>\n<html lang="hr">\n<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                     "<style>body{margin:0}</style>\n</head>\n<body>\n" + html + "\n</body>\n</html>\n", encoding="utf-8")
    print("zapisano", RES / "izvjestaj.html", len(html) // 1024, "KB")


TEMPLATE = (Path(__file__).resolve().parent / "us_report_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
