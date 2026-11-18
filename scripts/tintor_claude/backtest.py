"""Dnevni simulator portfelja za ZSE s ogranicenjem naloga prema prometu.

Tijek: signal na zatvaranju dana S (izvjestaji objavljeni do 16:00) -> nalozi od sljedece
sjednice -> izvrsenje po dnevnom VWAP-u uz proklizavanje, najvise `participation_cap`
dnevnog prometa, kroz najvise `max_fill_days` sjednica; ostatak se otkazuje i biljezi.

Upotreba:
  python -P backtest.py dev      # samo razvojno razdoblje
  python -P backtest.py final    # zavrsno razdoblje + cijelo razdoblje
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
RES = ROOT / "results" / "tintor_claude"
CFG = json.loads((ROOT / "config" / "tintor_claude.json").read_text(encoding="utf-8"))

GROUPS = {
    "rentabilnost": ["r_vk", "roi"],
    "novcani_tok": ["scf_a"],
    "otpornost": ["fp", "rocnost"],  # rocnost = prosjek rangova KL i P2 (isti rizik, jedan glas)
    "vrednovanje": ["scf_mc"],
}


# ----------------------------------------------------------------------------- bodovanje
def pct_rank(s: pd.Series, higher_better: bool = True) -> pd.Series:
    r = s.rank(pct=True, ascending=higher_better, method="average")
    return r


def score_snapshot(x: pd.DataFrame, cap: bool = True) -> pd.DataFrame:
    """Rangovi 0-1 (vise = bolje) unutar podobnog presjeka jednog datuma signala."""
    x = x.copy()
    kl, p2 = x.kl, x.p2
    if cap:  # visak likvidnosti/pokrica iznad praga ne donosi dodatne bodove
        kl = kl.clip(upper=CFG["scoring"]["liquidity_ratio_cap"])
        p2 = p2.clip(upper=CFG["scoring"]["coverage_ratio_cap"])
    r = pd.DataFrame(index=x.index)
    r["r_vk"] = pct_rank(x.r_vk)
    r["roi"] = pct_rank(x.roi)
    r["scf_a"] = pct_rank(x.scf_a)
    r["fp"] = pct_rank(x.fp, higher_better=False)
    r["kl"] = pct_rank(kl)
    r["p2"] = pct_rank(p2)
    # negativan ili nulti kapital: najlosiji rang za pokazatelje s VK u nazivniku
    neg = x.neg_equity.fillna(False).astype(bool)
    r.loc[neg, ["r_vk", "fp"]] = 0.0
    r["rocnost"] = r[["kl", "p2"]].mean(axis=1)
    r["scf_mc"] = pct_rank(x.scf_mc)
    for g, cols in GROUPS.items():
        x["g_" + g] = r[cols].mean(axis=1, skipna=False)
    for c in r.columns:
        x["rk_" + c] = r[c]
    x["rk_altman"] = pct_rank(x.altman_z)
    x["rk_kralicek"] = pct_rank(x.kralicek_df)
    return x


VARIANTS = {
    "kvaliteta": ["rentabilnost", "novcani_tok", "otpornost"],
    "kvaliteta+vrednovanje": ["rentabilnost", "novcani_tok", "otpornost", "vrednovanje"],
    "bez_rentabilnosti": ["novcani_tok", "otpornost"],
    "bez_novcanog_toka": ["rentabilnost", "otpornost"],
    "bez_otpornosti": ["rentabilnost", "novcani_tok"],
    "samo_vrednovanje": ["vrednovanje"],
}


def eligible(sig: pd.DataFrame) -> pd.DataFrame:
    e = sig[sig.eho_id.notna() & sig.fresh.fillna(False).astype(bool) & sig.ttm_ok.fillna(False).astype(bool)]
    need = ["A", "VK", "ND", "BD", "SCF", "kl", "p2"]
    return e.dropna(subset=need)


def build_targets(sig: pd.DataFrame, variant: str, n: int, cap: bool = True) -> dict:
    """Po datumu signala: {isin: ciljana tezina} i tablica rangova."""
    out, tabs = {}, []
    for S, x in eligible(sig).groupby("signal_date"):
        x = score_snapshot(x, cap=cap)
        if variant == "ew_svi":
            sel = x
            w = 1.0 / len(sel) if len(sel) else 0.0
        else:
            if variant == "altman":
                x["score"] = x.rk_altman
            elif variant == "kralicek":
                x["score"] = x.rk_kralicek
            else:
                x["score"] = x[["g_" + g for g in VARIANTS[variant]]].mean(axis=1, skipna=False)
            x = x.dropna(subset=["score"]).sort_values(["score", "adt"], ascending=False)
            x["rank"] = np.arange(1, len(x) + 1)
            sel = x.head(n)
            w = 1.0 / n if CFG["portfolio"]["hold_cash_when_short"] else (1.0 / len(sel) if len(sel) else 0.0)
        if len(x) < CFG["signal"]["min_universe"]:
            sel = sel.iloc[0:0]
        out[S] = {r["isin"]: w for _, r in sel.iterrows()}
        x["selected"] = x["isin"].isin(sel["isin"])
        tabs.append(x)
    return out, (pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame())


# ----------------------------------------------------------------------------- simulator
class Market:
    def __init__(self) -> None:
        p = pd.read_parquet(DATA / "prices.parquet")
        self.cal = pd.DatetimeIndex(sorted(p.date.unique()))
        self.close = p.pivot(index="date", columns="isin", values="close_adj")
        self.vwap = p.pivot(index="date", columns="isin", values="vwap_adj")
        self.turn = p.pivot(index="date", columns="isin", values="turnover").fillna(0.0)
        self.sym = p.sort_values("date").drop_duplicates("isin", keep="last").set_index("isin").symbol
        d = pd.read_parquet(DATA / "dividends.parquet")
        sig = pd.read_parquet(DATA / "signals.parquet")
        pr = p[["isin", "root"]].drop_duplicates()
        tk = set(d.ticker)
        pr["ticker"] = [r if r in tk else r.rstrip("0123456789") for r in pr.root]
        dd = d.merge(pr[["isin", "ticker"]], on="ticker")
        self.div = dd.groupby(["pay_date", "isin"]).dps_adj.sum()
        self.div_dates = set(dd.pay_date)
        self.sig = sig


def run(mkt: Market, targets: dict, start: str, end: str, *, capital: float | None = None,
        commission: float | None = None, slippage: float | None = None, cap: float | None = None,
        fill_days: int | None = None, dividends: bool = True, label: str = "") -> dict:
    ex = CFG["execution"]
    commission = ex["commission"] if commission is None else commission
    slippage = ex["slippage"] if slippage is None else slippage
    cap = ex["participation_cap"] if cap is None else cap
    fill_days = ex["max_fill_days"] if fill_days is None else fill_days
    cash = CFG["portfolio"]["initial_capital_eur"] if capital is None else capital
    min_trade = CFG["portfolio"]["min_trade_share_of_nav"]
    days = mkt.cal[(mkt.cal >= start) & (mkt.cal <= end)]
    pos: dict[str, float] = {}
    last_px: dict[str, float] = {}
    orders: list[dict] = []
    fills, skipped, navs, events = [], [], [], []
    for t in days:
        c, v, tv = mkt.close.loc[t], mkt.vwap.loc[t], mkt.turn.loc[t]
        # dividende (procjena iz novcanog toka)
        if dividends and t in mkt.div_dates:
            for i, q in pos.items():
                dps = mkt.div.get((t, i), 0.0)
                if dps and q > 0:
                    cash += q * dps
                    events.append({"date": t, "isin": i, "type": "dividenda", "amount": q * dps})
        # delistiranje: papir vise nije na cjeniku
        for i in list(pos):
            px = c.get(i, np.nan)
            if px != px:
                val = pos[i] * last_px.get(i, 0.0) * (1 - ex["delisting_haircut"])
                cash += val
                events.append({"date": t, "isin": i, "type": "delistiranje", "amount": val})
                del pos[i]
                orders = [o for o in orders if o["isin"] != i]
            else:
                last_px[i] = px
        # izvrsenje naloga: prvo prodaje, zatim kupnje
        for side in ("S", "B"):
            for o in [o for o in orders if o["side"] == side]:
                i = o["isin"]
                o["days"] += 1
                day_turn, px = tv.get(i, 0.0), v.get(i, np.nan)
                if day_turn > 0 and px == px and px > 0:
                    max_sh = cap * day_turn / px
                    q = min(o["rem"], max_sh)
                    if side == "B":
                        fp_ = px * (1 + slippage)
                        q = min(q, max(cash, 0.0) / (fp_ * (1 + commission)))
                        if q > 1e-9:
                            cash -= q * fp_ * (1 + commission)
                            pos[i] = pos.get(i, 0.0) + q
                            last_px[i] = c.get(i, px)
                    else:
                        fp_ = px * (1 - slippage)
                        q = min(q, pos.get(i, 0.0))
                        if q > 1e-9:
                            cash += q * fp_ * (1 - commission)
                            pos[i] -= q
                            if pos[i] <= 1e-9:
                                del pos[i]
                    if q > 1e-9:
                        o["rem"] -= q
                        fills.append({"signal_date": o["signal"], "exec_date": t, "isin": i, "symbol": mkt.sym.get(i),
                                      "side": side, "shares_adj": q, "vwap": px, "fill_price": fp_,
                                      "value": q * fp_, "commission": q * fp_ * commission,
                                      "day_turnover": day_turn, "participation": q * px / day_turn, "day_no": o["days"]})
                if o["rem"] <= 1e-9 * max(o["qty"], 1) or o["days"] >= fill_days:
                    if o["rem"] > 1e-6 * o["qty"]:
                        skipped.append({"signal_date": o["signal"], "cancel_date": t, "isin": i,
                                        "symbol": mkt.sym.get(i), "side": side, "ordered_shares": o["qty"],
                                        "unfilled_shares": o["rem"], "unfilled_share": o["rem"] / o["qty"],
                                        "unfilled_value": o["rem"] * o["ref_px"],
                                        "reason": "bez prometa u roku" if o["rem"] >= o["qty"] * 0.999999
                                        else "ogranicenje udjela u prometu / gotovina"})
                    o["done"] = True
            orders = [o for o in orders if not o.get("done")]
        nav = cash + sum(q * last_px.get(i, 0.0) for i, q in pos.items())
        navs.append({"date": t, "nav": nav, "cash": cash, "n_pos": len(pos)})
        # novi signal: nalozi vrijede od sljedece sjednice
        if t in targets:
            tg = targets[t]
            for o in orders:
                skipped.append({"signal_date": o["signal"], "cancel_date": t, "isin": o["isin"],
                                "symbol": mkt.sym.get(o["isin"]), "side": o["side"], "ordered_shares": o["qty"],
                                "unfilled_shares": o["rem"], "unfilled_share": o["rem"] / o["qty"],
                                "unfilled_value": o["rem"] * o["ref_px"], "reason": "zamijenjen novim signalom"})
            orders = []
            for i in sorted(set(pos) | set(tg)):
                px = c.get(i, np.nan)
                if px != px or px <= 0:
                    continue
                d_sh = tg.get(i, 0.0) * nav / px - pos.get(i, 0.0)
                full_exit = i not in tg
                if abs(d_sh) * px < min_trade * nav and not full_exit:
                    continue
                if abs(d_sh) < 1e-9:
                    continue
                orders.append({"signal": t, "isin": i, "side": "B" if d_sh > 0 else "S", "qty": abs(d_sh),
                               "rem": abs(d_sh), "days": 0, "ref_px": px})
    return {"label": label, "nav": pd.DataFrame(navs).set_index("date"), "fills": pd.DataFrame(fills),
            "skipped": pd.DataFrame(skipped), "events": pd.DataFrame(events), "positions": dict(pos)}


def stats(res: dict, start: str | None = None, end: str | None = None) -> dict:
    nav = res["nav"]
    if start:
        nav = nav[(nav.index >= start) & (nav.index <= end)]
    if len(nav) < 2:
        return {}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    r = nav.nav.pct_change().dropna()
    f = res["fills"]
    if len(f):
        f = f[(f.exec_date >= nav.index[0]) & (f.exec_date <= nav.index[-1])]
    sk = res["skipped"]
    if len(sk):
        sk = sk[(sk.signal_date >= nav.index[0]) & (sk.signal_date <= nav.index[-1])]
    buys = f[f.side == "B"].value.sum() if len(f) else 0.0
    sells = f[f.side == "S"].value.sum() if len(f) else 0.0
    ordered = (f.value.sum() if len(f) else 0.0) + (sk.unfilled_value.sum() if len(sk) else 0.0)
    return {
        "od": nav.index[0].date().isoformat(), "do": nav.index[-1].date().isoformat(),
        "godisnji_neto_prinos": (nav.nav.iloc[-1] / nav.nav.iloc[0]) ** (1 / yrs) - 1,
        "volatilnost": r.std() * np.sqrt(252),
        "max_pad": (nav.nav / nav.nav.cummax() - 1).min(),
        "godisnji_promet": min(buys, sells) / nav.nav.mean() / yrs,
        "prosjecan_broj_pozicija": nav.n_pos.mean(),
        "prosjecan_udio_gotovine": (nav.cash / nav.nav).mean(),
        "udio_neizvrsenog": (sk.unfilled_value.sum() / ordered) if len(sk) and ordered > 0 else 0.0,
        "broj_izvrsenja": len(f), "broj_otkazanih": len(sk),
        "troskovi_godisnje": (f.commission.sum() / nav.nav.mean() / yrs) if len(f) else 0.0,
    }


def index_stats(name: str, start: str, end: str) -> dict:
    x = pd.read_csv(DATA / "raw" / f"index_{name}.csv", parse_dates=["date"]).sort_values("date").set_index("date")
    s = x.last_value[(x.index >= start) & (x.index <= end)]
    if len(s) < 2:
        return {}
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return {"od": s.index[0].date().isoformat(), "do": s.index[-1].date().isoformat(),
            "godisnji_neto_prinos": (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1,
            "volatilnost": s.pct_change().std() * np.sqrt(252), "max_pad": (s / s.cummax() - 1).min()}


def main(mode: str) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    mkt = Market()
    n = CFG["portfolio"]["max_positions"]
    dev, fin = CFG["periods"]["development"], CFG["periods"]["final"]
    windows = {"razvoj": dev} if mode == "dev" else {"razvoj": dev, "zavrsno": fin}
    end_all = dev[1] if mode == "dev" else fin[1]
    sig = mkt.sig[mkt.sig.signal_date <= end_all]
    out_dir = RES / ("dev" if mode == "dev" else "final")
    out_dir.mkdir(exist_ok=True)

    rows, navs = [], {}

    def add(name: str, res: dict, group: str) -> None:
        navs[name] = res["nav"].nav
        for w, (a, b) in windows.items():
            s = stats(res, a, b)
            if s:
                rows.append({"skupina": group, "varijanta": name, "razdoblje": w, **s})

    # svaka varijanta krece iznova na pocetku svakog razdoblja (bez prijenosa pozicija)
    def run_windows(targets: dict, **kw) -> dict:
        parts = {w: run(mkt, targets, a, b, **kw) for w, (a, b) in windows.items()}
        return parts

    def add_parts(name: str, parts: dict, group: str) -> None:
        for w, res in parts.items():
            s = stats(res)
            if s:
                rows.append({"skupina": group, "varijanta": name, "razdoblje": w, **s})
            navs[f"{name}|{w}"] = res["nav"].nav

    main_parts, tabs = {}, {}
    tge0, _ = build_targets(sig, "ew_svi", n)
    for v in ["kvaliteta", "kvaliteta+vrednovanje", "bez_rentabilnosti", "bez_novcanog_toka", "bez_otpornosti",
              "samo_vrednovanje", "altman", "kralicek", "ew_svi"]:
        tg, tab = build_targets(sig, v, n)
        parts = run_windows(tg)
        add_parts(v, parts, "osnovno")
        main_parts[v], tabs[v] = parts, tab
        if v in ("kvaliteta", "kvaliteta+vrednovanje", "ew_svi"):
            add_parts(v + " (bez dividendi)", run_windows(tg, dividends=False), "bez dividendi")

    # baza bez uvjeta dostupnosti izvjestaja: mjeri pristranost prezivljavanja EHO arhive
    lu = pd.read_parquet(DATA / "liquid_universe.parquet")
    lu = lu[lu.signal_date <= end_all]
    tg_all = {S: {i: 1.0 / len(g) for i in g["isin"]} for S, g in lu.groupby("signal_date") if S in tge0}
    add_parts("ew_svi_likvidni (i bez izvjestaja)", run_windows(tg_all), "osnovno")
    add_parts("ew_svi_likvidni (i bez izvjestaja) (bez dividendi)", run_windows(tg_all, dividends=False), "bez dividendi")

    tgq, _ = build_targets(sig, "kvaliteta", n)
    tge, _ = build_targets(sig, "ew_svi", n)
    ex = CFG["execution"]
    for nm, kw in {"troskovi 0": dict(commission=0.0, slippage=0.0),
                   "troskovi x2": dict(commission=2 * ex["commission"], slippage=2 * ex["slippage"]),
                   "troskovi x3": dict(commission=3 * ex["commission"], slippage=3 * ex["slippage"]),
                   "kapital 25k": dict(capital=25_000), "kapital 500k": dict(capital=500_000),
                   "kapital 2M": dict(capital=2_000_000),
                   "udio u prometu 10%": dict(cap=0.10), "rok izvrsenja 3 dana": dict(fill_days=3)}.items():
        add_parts(f"kvaliteta | {nm}", run_windows(tgq, **kw), "osjetljivost")
        add_parts(f"ew_svi | {nm}", run_windows(tge, **kw), "osjetljivost")
    for k in (5, 12):
        tg, _ = build_targets(sig, "kvaliteta", k)
        add_parts(f"kvaliteta | {k} pozicija", run_windows(tg), "osjetljivost")
    tg, _ = build_targets(sig, "kvaliteta", n, cap=False)
    add_parts("kvaliteta | bez ogranicenja KL i P2", run_windows(tg), "osjetljivost")
    CFG["portfolio"]["hold_cash_when_short"] = False
    tg, _ = build_targets(sig, "kvaliteta", n)
    add_parts("kvaliteta | bez gotovine (1/broj odabranih)", run_windows(tg), "osjetljivost")
    CFG["portfolio"]["hold_cash_when_short"] = True

    for idx in ("CROBEX", "CROBEXtr"):
        for w, (a, b) in windows.items():
            s = index_stats(idx, a, b)
            if s:
                rows.append({"skupina": "indeks", "varijanta": idx, "razdoblje": w, **s})

    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(navs).to_csv(out_dir / "nav.csv")
    for v in ("kvaliteta", "kvaliteta+vrednovanje", "ew_svi"):
        for w, res in main_parts[v].items():
            tag = f"{v.replace('+', '_')}__{w}"
            res["fills"].to_csv(out_dir / f"fills__{tag}.csv", index=False)
            res["skipped"].to_csv(out_dir / f"skipped__{tag}.csv", index=False)
            res["events"].to_csv(out_dir / f"events__{tag}.csv", index=False)
    for v in ("kvaliteta", "kvaliteta+vrednovanje"):
        t = tabs[v]
        cols = ["signal_date", "ticker", "isin", "symbol", "rank", "score", "selected", "eho_id", "eho_id_annual",
                "period_end", "ptype", "published", "consolidated", "A", "VK", "TK", "ND", "BD", "SCF", "mcap",
                "r_vk", "roi", "scf_a", "fp", "kl", "p2", "scf_mc", "altman_z", "kralicek_df", "neg_equity",
                "rk_r_vk", "rk_roi", "rk_scf_a", "rk_fp", "rk_kl", "rk_p2", "rk_rocnost", "rk_scf_mc",
                "g_rentabilnost", "g_novcani_tok", "g_otpornost", "g_vrednovanje", "adt", "tds"]
        t[[c for c in cols if c in t.columns]].to_csv(out_dir / f"ranks__{v.replace('+', '_')}.csv", index=False)

    pd.set_option("display.width", 250)
    show = summ[summ.skupina.isin(["osnovno", "indeks", "bez dividendi"])][
        ["varijanta", "razdoblje", "godisnji_neto_prinos", "volatilnost", "max_pad", "godisnji_promet",
         "prosjecan_broj_pozicija", "prosjecan_udio_gotovine", "udio_neizvrsenog"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
