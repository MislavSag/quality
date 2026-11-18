"""Tintorove strategije na americkom trzistu (EODHD izvjestaji + QC dnevne cijene).

Formule (tintor.py) i bodovanje (backtest.score_snapshot) su isti kao za ZSE. Razlike:
tokovi su zbroj zadnja cetiri tromjesecja, portfelj je 50 dionica, izvrsenje je na
zatvaranju sjednice nakon signala uz trosak po strani, bez modela djelomicnog izvrsenja.
Izlaz: results/tintor_claude_us/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tintor as T  # noqa: E402
from backtest import VARIANTS, score_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "config" / "tintor_claude_us.json").read_text(encoding="utf-8"))
ZSE = json.loads((ROOT / "config" / "tintor_claude.json").read_text(encoding="utf-8"))
SRC = Path(CFG["source"])
RES = ROOT / "results" / "tintor_claude_us"
CACHE = ROOT / "data" / "tintor_claude" / "us"
STRATS = {"kvaliteta": "kvaliteta", "kvaliteta+vrednovanje": "kvaliteta+vrednovanje",
          "altman": "altman", "kralicek": "kralicek"}
FLOW = {"net_income": "FUND_IS_NET_INCOME", "ebt": "FUND_IS_INCOME_BEFORE_TAX", "ebit": "FUND_IS_EBIT",
        "interest_exp": "FUND_IS_INTEREST_EXPENSE", "total_rev": "FUND_IS_TOTAL_REVENUE",
        "cfo": "FUND_CF_OPERATING_CASH_FLOW", "capex": "FUND_CF_CAPITAL_EXPENDITURES",
        "da_cf": "FUND_CF_DEPRECIATION", "da_is": "FUND_IS_DEPRECIATION_AND_AMORTIZATION"}
STOCK = {"total_assets": "FUND_BS_TOTAL_ASSETS", "equity": "FUND_BS_TOTAL_STOCKHOLDER_EQUITY",
         "total_liab": "FUND_BS_TOTAL_LIABILITIES", "current_assets": "FUND_BS_TOTAL_CURRENT_ASSETS",
         "st_liab": "FUND_BS_TOTAL_CURRENT_LIABILITIES", "retained": "FUND_BS_RETAINED_EARNINGS",
         "inventories": "FUND_BS_INVENTORY"}


def ttm_table() -> pd.DataFrame:
    """Jedan redak po izdavatelju i tromjesecju: stanja zadnjeg tromjesecja i zbroj tokova 4 tromjesecja."""
    cols = ["security_id", "FUND_FISCAL_DATE", "FUND_AVAILABLE_DATE", "FUND_CURRENCY", *FLOW.values(), *STOCK.values()]
    q = (pl.scan_parquet(SRC / "fundamental_statements_base.parquet")
         .filter(pl.col("FUND_STATEMENT_PERIOD") == "quarterly").select(cols).collect().to_pandas())
    q = q.rename(columns={v: k for k, v in {**FLOW, **STOCK}.items()})
    q = q.rename(columns={"FUND_FISCAL_DATE": "fiscal", "FUND_AVAILABLE_DATE": "avail"})
    q["fiscal"], q["avail"] = pd.to_datetime(q.fiscal), pd.to_datetime(q.avail)
    q = q.sort_values(["security_id", "fiscal", "avail"]).drop_duplicates(["security_id", "fiscal"], keep="last")
    q["capex"] = q.capex.abs().fillna(0.0)          # izvor ga daje s oba predznaka
    q["interest_exp"] = q.interest_exp.abs()
    q["ebt"] = q.ebt.fillna(q.ebit - q.interest_exp.fillna(0.0))
    q["interest_exp"] = q.interest_exp.fillna((q.ebit - q.ebt).clip(lower=0)).fillna(0.0)
    q["da"] = q.da_cf.fillna(q.da_is).abs()
    g = q.groupby("security_id", sort=False)
    span = (q.fiscal - g.fiscal.shift(3)).dt.days
    ok = span.between(250, 290)                      # cetiri uzastopna tromjesecja
    for c in ["net_income", "ebt", "interest_exp", "total_rev", "cfo", "capex", "da"]:
        q[c] = g[c].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True).where(ok)
    return q.drop(columns=["da_cf", "da_is", "ebit"])


def indicators(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["A"], x["VK"] = x.total_assets, x.equity
    x["TK"] = x.A - x.VK
    x["ND"] = x.net_income
    x["BD"] = T.bruto_dobit(x.ebt, x.interest_exp).values
    x["SCF"] = T.scf(x.cfo, x.capex).values
    x["neg_equity"] = x.VK <= 0
    fixed = x.A - x.current_assets
    lt = x.total_liab.fillna(x.TK) - x.st_liab
    x["r_vk"] = T.r_vk(x.ND, x.VK).values
    x["roi"] = T.roi(x.BD, x.A).values
    x["scf_a"] = T.roi(x.SCF, x.A).values
    x["fp"] = T.fp(x.TK, x.VK).values
    x["kl"] = T.kl(x.current_assets, x.st_liab).values
    x["p2"] = T.p2(x.VK, lt, fixed).values
    x["scf_mc"] = (x.SCF / x.mcap.where(x.mcap > 0)).values
    x["altman_z"] = T.altman_z(x.current_assets - x.st_liab, x.retained, x.BD, x.VK, x.TK, x.total_rev, x.A).values
    x["kralicek_df"] = T.kralicek_df(x.ND + x.da, x.TK, x.A, x.BD, x.total_rev, x.inventories).values
    return x


def signal_dates(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    out = []
    for y in range(cal[0].year, cal[-1].year + 1):
        for m in ZSE["signal"]["months"]:
            d = pd.Timestamp(y, m, ZSE["signal"]["day_of_month"])
            i = cal.searchsorted(d)
            if i < len(cal) - 2 and cal[i] >= pd.Timestamp(CFG["first_signal"]):
                out.append(cal[i])
    return out


def build_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    CACHE.mkdir(parents=True, exist_ok=True)
    pv = pl.scan_parquet(SRC / "price_volume_characteristics.parquet")
    cal = pd.DatetimeIndex(pv.filter(pl.col("ticker") == "SPY").select("trade_date").collect().to_series().sort().to_list())
    S = signal_dates(cal)
    dates = [d.date() for d in S]
    px = (pv.filter(pl.col("trade_date").is_in(dates))
          .select("security_id", "ticker", "trade_date", "PV_CLOSE_RAW", "PV_AVG_DOLLAR_VOLUME_126D").collect().to_pandas())
    fc = (pl.scan_parquet(SRC / "fundamental_characteristics.parquet").filter(pl.col("trade_date").is_in(dates))
          .select("security_id", "trade_date", "FUND_ISSUER_ID", "FUND_IS_COMMON_STOCK", "FUND_MCAP_MAPPING_OK",
                  "FUND_MARKET_CAP_USD_COMPAT", "FUND_CURRENCY_USD_COMPATIBLE").collect().to_pandas())
    meta = pl.read_parquet(SRC / "fundamental_security_meta.parquet").select(
        "security_id", "FUND_EODHD_GIC_SECTOR", "FUND_EODHD_SECTOR", "FUND_EODHD_NAME").to_pandas().drop_duplicates("security_id")
    u = px.merge(fc, on=["security_id", "trade_date"], how="left").merge(meta, on="security_id", how="left")
    u["signal_date"] = pd.to_datetime(u.trade_date)
    u = u.rename(columns={"PV_CLOSE_RAW": "price", "PV_AVG_DOLLAR_VOLUME_126D": "adt", "FUND_MARKET_CAP_USD_COMPAT": "mcap"})
    U = CFG["universe"]
    sector = u.FUND_EODHD_GIC_SECTOR.fillna(u.FUND_EODHD_SECTOR)
    u["liquid"] = ((u.FUND_IS_COMMON_STOCK == True) & (u.FUND_CURRENCY_USD_COMPATIBLE == True)  # noqa: E712
                   & (u.mcap >= U["min_market_cap_usd"]) & (u.price >= U["min_raw_price"])
                   & (u.adt >= U["min_avg_dollar_volume_126d"]))
    u["financial"] = sector.isin(U["exclude_sectors"])
    u = u[u.liquid]
    t = ttm_table()
    rows = []
    max_age = pd.Timedelta(days=ZSE["signal"]["max_statement_age_days"])
    for s in S:
        cur = t[t.avail <= s].groupby("security_id", sort=False).tail(1)
        x = u[u.signal_date == s].merge(cur, on="security_id", how="left")
        x["fresh"] = (s - x.fiscal) <= max_age
        rows.append(x)
    p = indicators(pd.concat(rows, ignore_index=True))
    need = ["A", "VK", "ND", "BD", "SCF", "kl", "p2"]
    p["has_statement"] = p.fiscal.notna()
    p["eligible"] = p.fresh.fillna(False) & ~p.financial & p[need].notna().all(axis=1) & (p.FUND_CURRENCY == "USD")
    # jedna klasa dionica po izdavatelju: najlikvidnija
    p["issuer"] = p.FUND_ISSUER_ID.fillna(p.security_id)
    p = p.sort_values(["signal_date", "issuer", "adt"], ascending=[True, True, False])
    p.loc[p.duplicated(["signal_date", "issuer"]) & p.eligible, "eligible"] = False
    p.to_parquet(CACHE / "panel.parquet")
    return p, cal


def score_all(p: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, x in p[p.eligible].groupby("signal_date"):
        x = score_snapshot(x, cap=True)
        for name in STRATS:
            if name == "altman":
                x["s_" + name] = x.rk_altman
            elif name == "kralicek":
                x["s_" + name] = x.rk_kralicek
            else:
                x["s_" + name] = x[["g_" + g for g in VARIANTS[name]]].mean(axis=1, skipna=False)
        out.append(x)
    return pd.concat(out, ignore_index=True)


def returns_matrix(ids: list[str], cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    f = CACHE / "returns.parquet"
    if f.exists():
        r = pd.read_parquet(f)
        if set(ids) <= set(r.columns):
            return r, pd.read_parquet(CACHE / "terminal_unknown.parquet").flag
    d = (pl.scan_parquet(SRC / "price_volume_characteristics.parquet").filter(pl.col("security_id").is_in(ids))
         .select("security_id", "trade_date", "PV_RET_CC_1D", "PV_TERMINAL_RETURN_UNKNOWN").collect())
    flag = d.group_by("security_id").agg(pl.col("PV_TERMINAL_RETURN_UNKNOWN").fill_null(False).any().alias("flag")).to_pandas().set_index("security_id").flag
    r = d.select("security_id", "trade_date", "PV_RET_CC_1D").to_pandas().pivot(index="trade_date", columns="security_id", values="PV_RET_CC_1D")
    r.index = pd.to_datetime(r.index)
    r = r.reindex(cal).astype("float32")
    r.to_parquet(f)
    flag.to_frame().to_parquet(CACHE / "terminal_unknown.parquet")
    return r, flag


def simulate(targets: dict, R: pd.DataFrame, cal: pd.DatetimeIndex, cost: float, haircut: float = 0.0,
             flag: pd.Series | None = None) -> tuple[pd.Series, float, int]:
    """Jednaki ponderi; trgovanje na zatvaranju sjednice nakon signala; trosak na promet po strani."""
    col = {c: i for i, c in enumerate(R.columns)}
    A = R.to_numpy()
    last_valid = np.where(np.isfinite(A), np.arange(len(A))[:, None], -1).max(axis=0)
    A = np.nan_to_num(A, nan=0.0).astype(np.float64)
    fl = np.array([bool(flag.get(c, False)) for c in R.columns]) if flag is not None else None
    sig = sorted(targets)
    tix = [cal.get_loc(s) + 1 for s in sig]
    nav = np.full(len(cal), np.nan)
    value = np.zeros(A.shape[1])
    cash, turn, lost = 1.0, 0.0, 0
    for k, t in enumerate(tix):
        end = tix[k + 1] if k + 1 < len(tix) else len(cal) - 1
        total = value.sum() + cash
        w = np.zeros(A.shape[1])
        ids = [col[i] for i in targets[sig[k]]]
        w[ids] = 1.0 / len(ids)
        trade = np.abs(w * total - value).sum()
        turn += trade / total
        total -= trade * cost
        value, cash = w * total, 0.0
        nav[t] = total
        for d in range(t + 1, end + 1):
            value = value * (1.0 + A[d])
            if haircut and flag is not None:
                gone = (last_valid == d - 1) & (value > 0) & (d - 1 < len(cal) - 6)
                if gone.any():
                    hit = gone & fl
                    lost += int(hit.sum())
                    value[hit] *= 1.0 + haircut
            nav[d] = value.sum() + cash
    s = pd.Series(nav, index=cal).dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    return s / s.iloc[0], turn / years, lost


def stats(nav: pd.Series, a: str | None = None, b: str | None = None) -> dict:
    s = nav.loc[a:b]
    r = s.pct_change().dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    return {"godisnji_prinos": cagr, "volatilnost": vol, "prinos_kroz_volatilnost": cagr / vol,
            "max_pad": (s / s.cummax() - 1).min()}


def active_t(nav: pd.Series, base: pd.Series, a=None, b=None) -> tuple[float, float, int]:
    m = pd.concat([nav, base], axis=1).loc[a:b].resample("ME").last().pct_change().dropna()
    d = m.iloc[:, 0] - m.iloc[:, 1]
    return d.mean(), d.mean() / d.std() * np.sqrt(len(d)), len(d)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    T.self_test()
    p, cal = build_panel()
    sc = score_all(p)
    R, flag = returns_matrix(sorted(sc.security_id.unique()), cal)
    n, split, c0 = CFG["portfolio"]["top_n"], CFG["split"], CFG["execution"]["cost_per_side"]
    targets = {"ew_svi": {s: list(x.security_id) for s, x in sc.groupby("signal_date")}}
    for name in STRATS:
        targets[name] = {s: list(x.dropna(subset=["s_" + name]).sort_values(["s_" + name, "adt"], ascending=False).security_id.head(n))
                         for s, x in sc.groupby("signal_date")}
    navs, rows = {}, []
    for name, tg in targets.items():
        navs[name], turnover, _ = simulate(tg, R, cal, c0)
        for label, a, b in [("cijelo", None, None), ("do_2018", None, "2018-12-31"), ("od_2019", split, None)]:
            rows.append({"varijanta": name, "razdoblje": label, **stats(navs[name], a, b), "godisnji_promet": turnover})
    spy = (1 + R.get("SPY R735QTJ8XC9X", pd.Series(dtype=float)).fillna(0)).cumprod() if "SPY R735QTJ8XC9X" in R else None
    if spy is None:
        s = (pl.scan_parquet(SRC / "price_volume_characteristics.parquet").filter(pl.col("ticker") == "SPY")
             .select("trade_date", "PV_CLOSE").collect().to_pandas())
        spy = pd.Series(s.PV_CLOSE.values, index=pd.to_datetime(s.trade_date)).sort_index()
    spy = spy.reindex(navs["ew_svi"].index)
    navs["SPY"] = spy / spy.iloc[0]
    for label, a, b in [("cijelo", None, None), ("do_2018", None, "2018-12-31"), ("od_2019", split, None)]:
        rows.append({"varijanta": "SPY", "razdoblje": label, **stats(navs["SPY"], a, b), "godisnji_promet": 0.0})
    summ = pd.DataFrame(rows)
    for i, r in summ.iterrows():
        if r.varijanta in STRATS:
            a, b = {"cijelo": (None, None), "do_2018": (None, "2018-12-31"), "od_2019": (split, None)}[r.razdoblje]
            m, t, k = active_t(navs[r.varijanta], navs["ew_svi"], a, b)
            summ.loc[i, ["mj_razlika_prema_ew", "t_prema_ew", "mjeseci"]] = [m, t, k]
    summ.to_csv(RES / "summary.csv", index=False)
    pd.DataFrame(navs).to_parquet(RES / "nav.parquet")

    # osjetljivosti: trosak i delistiranja
    sens = []
    for name in STRATS:
        for c in CFG["execution"]["cost_sensitivity"]:
            sens.append({"varijanta": name, "test": f"trosak {c:.2%} po strani", **stats(simulate(targets[name], R, cal, c)[0])})
        nv, _, lost = simulate(targets[name], R, cal, c0, CFG["execution"]["delisting_haircut_sensitivity"], flag)
        sens.append({"varijanta": name, "test": f"delistiranje -30 % ({lost} slucajeva)", **stats(nv)})
    pd.DataFrame(sens).to_csv(RES / "sensitivity.csv", index=False)

    # kvintili po ocjeni (bruto, jednaki ponderi) - pokazuje je li veza monotona
    qrows = []
    for name in STRATS:
        x = sc.dropna(subset=["s_" + name]).copy()
        x["q"] = x.groupby("signal_date")["s_" + name].transform(
            lambda v: pd.qcut(v.rank(method="first"), CFG["portfolio"]["quantiles"], labels=False) + 1)
        for q, y in x.groupby("q"):
            nv = simulate({s: list(z.security_id) for s, z in y.groupby("signal_date")}, R, cal, 0.0)[0]
            for label, a, b in [("cijelo", None, None), ("do_2018", None, "2018-12-31"), ("od_2019", split, None)]:
                qrows.append({"varijanta": name, "kvintil": int(q), "razdoblje": label, **stats(nv, a, b)})
    pd.DataFrame(qrows).to_csv(RES / "quintiles.csv", index=False)

    cov = p.groupby("signal_date").agg(likvidnih=("security_id", "size"), s_izvjestajem=("has_statement", "sum"),
                                       financijskih=("financial", "sum"), podobnih=("eligible", "sum")).reset_index()
    cov.to_csv(RES / "coverage.csv", index=False)
    last = sc[sc.signal_date == sc.signal_date.max()]
    keep = ["ticker", "FUND_EODHD_NAME", "fiscal", "avail", "mcap", "r_vk", "roi", "scf_a", "fp", "kl", "p2", "scf_mc",
            "altman_z", "kralicek_df"] + ["s_" + k for k in STRATS]
    for name in STRATS:
        last.sort_values("s_" + name, ascending=False).head(n)[keep].to_csv(RES / f"holdings_last__{name}.csv", index=False)
    sc[["signal_date", "security_id", "ticker", "fiscal", "avail", "mcap", "adt", "r_vk", "roi", "scf_a", "fp", "kl", "p2",
        "scf_mc", "altman_z", "kralicek_df"] + ["s_" + k for k in STRATS]].to_parquet(RES / "scores.parquet")
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string())
    print(pd.DataFrame(sens).round(4).to_string())
    print(pd.DataFrame(qrows).pivot_table(index=["varijanta", "razdoblje"], columns="kvintil", values="godisnji_prinos").round(4).to_string())
    print(cov.iloc[::8].to_string())


if __name__ == "__main__":
    main()
