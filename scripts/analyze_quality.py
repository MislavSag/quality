"""Monthly, gross-return diagnostics of the existing local profitability panel.

This is signal research, not a simulated executable short strategy. Inputs are
read-only. Formation uses no future outcome or future universe membership.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("H:/strategies/characteristics")
EXT = Path("H:/strategies/characteristics_profitability_20261110")
OUT = ROOT / "results" / "initial_analysis"
CACHE = ROOT / "data" / "derived"
BASE = {
    "roa_ttm": "FUND_RETURN_ON_ASSETS_TTM",
    "roe_ttm": "FUND_RETURN_ON_EQUITY_TTM",
    "gp_assets_ttm": "FUND_GROSS_PROFIT_TO_ASSETS_TTM",
    "op_assets_ttm": "FUND_OPERATING_PROFIT_TO_ASSETS_TTM",
    "ebit_assets_ttm": "FUND_EBIT_TO_ASSETS_TTM",
    "cfo_assets_ttm": "FUND_CFO_TO_ASSETS_TTM",
    "fcf_assets_ttm": "FUND_FCF_TO_ASSETS_TTM",
    "gross_margin_ttm": "FUND_GROSS_MARGIN_TTM",
    "operating_margin_ttm": "FUND_OPERATING_MARGIN_TTM",
    "ebit_margin_ttm": "FUND_EBIT_MARGIN_TTM",
    "ebitda_margin_ttm": "FUND_EBITDA_MARGIN_TTM",
    "net_margin_ttm": "FUND_NET_MARGIN_TTM",
}
EXTRA = {
    "ff2015_fy": "FUND_FF2015_OP_BE_FY_EODHD",
    "gp_assets_fy": "FUND_NM_GP_ASSETS_FY_EODHD",
    "ball2015_fy": "FUND_BALL2015_OP_ASSETS_FY_PROXY",
    "ball2016_fy": "FUND_BALL2016_OP_LAG_ASSETS_FY_PROXY",
    "cbop_bs_fy": "FUND_BALL_CBOP_BS_FY_PROXY",
    "cbop_cf_fy": "FUND_BALL_CBOP_CF_FY_PROXY",
    "roic_fy": "FUND_ROIC_FY_EODHD",
    "hxz_roe_q": "FUND_HXZ_ROE_Q_PROXY",
    "qmj_like_fy": "FUND_QMJ_LIKE_PROFITABILITY_FY",
}
SIGNALS = {**BASE, **EXTRA}
CORE = ["gp_assets_ttm", "roa_ttm", "roe_ttm", "cfo_assets_ttm", "gp_assets_fy", "qmj_like_fy"]
CONTROLS = {
    "value": "FUND_BOOK_TO_MARKET",
    "sales_growth": "FUND_SALES_GROWTH_YOY",
    "asset_growth": "FUND_ASSET_GROWTH_YOY",
}
PRICES = SOURCE / "price_volume_characteristics.parquet"
FUND = SOURCE / "fundamental_characteristics.parquet"
EXTFILE = EXT / "fundamental_profitability_extensions.parquet"


def log(message):
    print(message, flush=True)


def fingerprint(path):
    stat = path.stat()
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        footer_bytes = int.from_bytes(stream.read(4), "little")
        stream.seek(-(8 + footer_bytes), 2)
        footer_hash = hashlib.sha256(stream.read(footer_bytes + 8)).hexdigest()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "parquet_footer_sha256": footer_hash}


def session_calendar():
    spy = ds.dataset(str(PRICES)).to_table(
        columns=["trade_date", "PV_CLOSE"], filter=ds.field("ticker") == "SPY"
    ).to_pandas(date_as_object=False).dropna().sort_values("trade_date")
    assert not spy.trade_date.duplicated().any()
    sessions = spy.trade_date.to_numpy()
    ends = spy.groupby(spy.trade_date.dt.to_period("M")).trade_date.max().tolist()
    rows = []
    for start, end in zip(ends[:-1], ends[1:]):
        entry_i = sessions.searchsorted(np.datetime64(start), side="right")
        exit_i = sessions.searchsorted(np.datetime64(end), side="right")
        if entry_i < len(sessions) and exit_i < len(sessions):
            rows.append((start, pd.Timestamp(sessions[entry_i]), pd.Timestamp(sessions[exit_i])))
    calendar = pd.DataFrame(rows, columns=["signal_date", "entry_date", "exit_date"])
    assert (calendar.signal_date < calendar.entry_date).all()
    assert (calendar.entry_date < calendar.exit_date).all()
    return calendar, spy


def selected_scan(path, columns, dates):
    return (pl.scan_parquet(path).select(columns)
            .filter(pl.col("trade_date").is_in(dates)).collect(streaming=True))


def build_monthly(calendar, inputs):
    cache = CACHE / "monthly_panel.parquet"
    contract = {"inputs": inputs, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = CACHE / "contract.json"
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text()) == contract:
        log("Reusing the matching monthly cache")
        return pd.read_parquet(cache)
    signal_dates = [x.date() for x in calendar.signal_date]
    all_dates = sorted({d.date() for col in calendar for d in calendar[col]})
    log("Reading monthly price endpoints and formation liquidity")
    prices = selected_scan(PRICES, ["security_id", "ticker", "trade_date", "PV_CLOSE", "PV_CLOSE_RAW",
        "PV_AVG_DOLLAR_VOLUME_21D", "PV_MOM_12_1", "PV_QC_FACTOR_CONTRACT_OK",
        "PV_TERMINAL_RETURN_UNKNOWN"], all_dates).to_pandas(date_as_object=False)
    assert not prices.duplicated(["security_id", "trade_date"]).any()
    log(f"Read {len(prices):,} price endpoints")
    formation = prices[prices.trade_date.isin(calendar.signal_date)].rename(columns={"trade_date": "signal_date"})
    panel = formation.merge(calendar, on="signal_date", validate="many_to_one")
    endpoint = prices[["security_id", "trade_date", "PV_CLOSE", "PV_QC_FACTOR_CONTRACT_OK"]]
    for label in ["entry", "exit"]:
        ep = endpoint.rename(columns={"trade_date": f"{label}_date", "PV_CLOSE": f"{label}_price",
                                      "PV_QC_FACTOR_CONTRACT_OK": f"{label}_contract_ok"})
        panel = panel.merge(ep, on=["security_id", f"{label}_date"], how="left", validate="many_to_one")
    log("Reading month-end fundamental characteristics")
    fundamental_columns = ["security_id", "trade_date", "FUND_MARKET_CAP_USD_COMPAT",
        "FUND_CURRENCY_USD_COMPATIBLE", "FUND_HAS_STATEMENT", "FUND_STATEMENT_AGE_DAYS",
        "FUND_STATEMENT_AVAILABLE_DATE", *BASE.values(), *CONTROLS.values()]
    fund = selected_scan(FUND, fundamental_columns, signal_dates).to_pandas(date_as_object=False)
    ext = selected_scan(EXTFILE, ["security_id", "trade_date", *EXTRA.values(),
        "FUND_QMJ_LIKE_COMPONENT_COUNT"], signal_dates).to_pandas(date_as_object=False)
    for data in [fund, ext]:
        data.rename(columns={"trade_date": "signal_date"}, inplace=True)
        assert not data.duplicated(["security_id", "signal_date"]).any()
        panel = panel.merge(data, on=["security_id", "signal_date"], how="left", validate="one_to_one")
    panel.rename(columns={v: k for k, v in {**SIGNALS, **CONTROLS}.items()}, inplace=True)
    eligible = (
        panel.FUND_HAS_STATEMENT.fillna(False).astype(bool)
        & panel.FUND_CURRENCY_USD_COMPATIBLE.fillna(False).astype(bool)
        & (panel.FUND_MARKET_CAP_USD_COMPAT >= 100_000_000)
        & (panel.PV_CLOSE_RAW >= 5)
        & (panel.PV_AVG_DOLLAR_VOLUME_21D >= 1_000_000)
        & panel.PV_QC_FACTOR_CONTRACT_OK.fillna(False).astype(bool)
        & ~panel.PV_TERMINAL_RETURN_UNKNOWN.fillna(True).astype(bool)
    )
    panel["eligible"] = eligible
    panel["return_observed"] = ((panel.entry_price > 0) & (panel.exit_price > 0)
        & panel.entry_contract_ok.fillna(False).astype(bool)
        & panel.exit_contract_ok.fillna(False).astype(bool))
    panel["forward_return"] = (panel.exit_price / panel.entry_price - 1).where(panel.return_observed)
    panel["log_size"] = np.log(panel.FUND_MARKET_CAP_USD_COMPAT.where(panel.FUND_MARKET_CAP_USD_COMPAT > 0))
    panel["momentum"] = panel.PV_MOM_12_1
    assert (panel.loc[eligible, "FUND_STATEMENT_AVAILABLE_DATE"] <= panel.loc[eligible, "signal_date"]).all()
    assert not panel.duplicated(["security_id", "signal_date"]).any()
    CACHE.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache, index=False)
    stamp.write_text(json.dumps(contract, indent=2))
    log(f"Saved {len(panel):,} month-end rows; {eligible.sum():,} ex-ante eligible")
    return panel


def hac_mean(values, lags=3):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 12 or np.std(values) == 0:
        return np.nan, np.nan
    fit = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


def cosine_similarity(signal, returns):
    """Uncentered cosine; finite, matched vectors only; undefined for zero norm."""
    x, y = np.asarray(signal, dtype=float), np.asarray(returns, dtype=float)
    if x.shape != y.shape or not len(x) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    sx, sy = np.max(np.abs(x)), np.max(np.abs(y))
    if sx == 0 or sy == 0:
        return np.nan
    x, y = x/sx, y/sy
    return float(np.clip(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)), -1, 1))


def cosine_metrics(sample, signal):
    observed = sample.return_observed & np.isfinite(sample.forward_return)
    if observed.sum() < 100:
        return {"ic_cosine_raw": np.nan, "ic_cosine_rank_centered": np.nan}
    # Rank and center on the formation set, before observing missing outcomes.
    ranks = sample[signal].rank(method="average")
    rank_signal = ranks - ranks.mean()
    returns = sample.loc[observed, "forward_return"]
    return {"ic_cosine_raw": cosine_similarity(sample.loc[observed, signal], returns),
            "ic_cosine_rank_centered": cosine_similarity(rank_signal.loc[observed], returns)}


def diagnostics(data, label):
    records = []
    for month, group in data.groupby("signal_date", sort=True):
        for signal in SIGNALS:
            sample = group[np.isfinite(group[signal])].copy()
            if len(sample) < 100 or sample[signal].nunique() < 5:
                continue
            observed = sample[sample.return_observed]
            ic = observed[signal].corr(observed.forward_return, method="spearman") if len(observed) >= 100 else np.nan
            bins = np.floor((sample[signal].rank(method="average") - .5) / len(sample) * 5).clip(0, 4).astype(int)
            sample["quintile"] = bins
            row = {"sample": label, "signal_date": month, "return_month": sample.entry_date.iloc[0].strftime("%Y-%m"),
                "signal": signal, "n": len(sample), "n_observed": len(observed),
                "missing_return_fraction": 1-len(observed)/len(sample), "ic": ic,
                "ic_spearman": ic, **cosine_metrics(sample, signal)}
            for q in range(5):
                sleeve = sample[sample.quintile == q]
                known = sleeve.forward_return.notna()
                miss = 1-known.mean() if len(sleeve) else np.nan
                row[f"q{q+1}_n"] = len(sleeve)
                row[f"q{q+1}_missing"] = miss
                # Preserve the formation set; accept at most 1% unobserved equal weight.
                row[f"q{q+1}_return"] = sleeve.forward_return.mean() if len(sleeve) >= 10 and miss <= .01+1e-12 else np.nan
                row[f"q{q+1}_missing_minus100"] = sleeve.forward_return.fillna(-1).mean() if len(sleeve) >= 10 else np.nan
            row["spread"] = row["q5_return"] - row["q1_return"]
            row["spread_missing_minus100"] = row["q5_missing_minus100"] - row["q1_missing_minus100"]
            records.append(row)
    return pd.DataFrame(records)


def summarize(monthly):
    rows = []
    for (sample, signal), group in monthly.groupby(["sample", "signal"]):
        for period, part in [("full", group), ("2007-2012", group[group.signal_date.dt.year <= 2012]),
            ("2013-2019", group[group.signal_date.dt.year.between(2013, 2019)]),
            ("2020-2026", group[group.signal_date.dt.year >= 2020])]:
            ic, ic_t = hac_mean(part.ic)
            cos_raw, cos_raw_t = hac_mean(part.ic_cosine_raw)
            cos_rank, cos_rank_t = hac_mean(part.ic_cosine_rank_centered)
            spread, spread_t = hac_mean(part.spread)
            sd = part.spread.std()
            rows.append({"sample": sample, "signal": signal, "period": period,
                "months_ic": part.ic.notna().sum(), "months_spread": part.spread.notna().sum(),
                "mean_n": part.n.mean(), "mean_missing_return_pct": part.missing_return_fraction.mean()*100,
                "mean_ic": ic, "ic_hac_t": ic_t,
                "mean_ic_spearman": ic, "ic_spearman_hac_t": ic_t,
                "mean_ic_cosine_raw": cos_raw, "ic_cosine_raw_hac_t": cos_raw_t,
                "months_ic_cosine_raw": part.ic_cosine_raw.notna().sum(),
                "mean_ic_cosine_rank_centered": cos_rank, "ic_cosine_rank_centered_hac_t": cos_rank_t,
                "months_ic_cosine_rank_centered": part.ic_cosine_rank_centered.notna().sum(),
                "mean_spread_monthly": spread,
                "spread_arithmetic_annualized": spread*12, "spread_hac_t": spread_t,
                "spread_gross_sharpe": spread/sd*np.sqrt(12) if sd > 0 else np.nan,
                "mean_top_return": part.q5_return.mean(), "mean_bottom_return": part.q1_return.mean()})
    return pd.DataFrame(rows)


def ic_columns_for_export():
    return ["signal", "period", "months_ic", "mean_n", "mean_ic_spearman", "ic_spearman_hac_t",
            "mean_ic_cosine_raw", "ic_cosine_raw_hac_t", "months_ic_cosine_raw",
            "mean_ic_cosine_rank_centered", "ic_cosine_rank_centered_hac_t",
            "months_ic_cosine_rank_centered"]


def regressions(data):
    signals = ["gp_assets_ttm", "roa_ttm", "cfo_assets_ttm", "qmj_like_fy", "gp_assets_fy",
               "ff2015_fy", "ball2016_fy", "hxz_roe_q", "roic_fy"]
    rows = []
    controls = ["value", "log_size", "momentum", "sales_growth", "asset_growth"]
    for month, group in data.groupby("signal_date"):
        for signal in signals:
            sample = group[[signal, *controls, "forward_return"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 100:
                continue
            ranks = sample[[signal, *controls]].rank(method="average")
            z = (ranks-ranks.mean())/ranks.std()
            if not np.isfinite(z.to_numpy()).all():
                continue
            z = z.rename(columns={signal: "profitability"})
            z["profitability_x_sales_growth"] = z.profitability*z.sales_growth
            x = np.column_stack([np.ones(len(z)), z.to_numpy()])
            if np.linalg.matrix_rank(x) < x.shape[1]:
                continue
            coef = np.linalg.lstsq(x, sample.forward_return.to_numpy(), rcond=None)[0]
            rows.extend({"signal_date": month, "signal": signal, "term": term, "coefficient": value, "n": len(sample)}
                        for term, value in zip(["intercept", *z.columns], coef))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT/"monthly_cross_section_regressions.csv", index=False)
    results = []
    for (signal, term), group in raw.groupby(["signal", "term"]):
        mean, t = hac_mean(group.coefficient)
        results.append({"signal": signal, "term": term, "mean_coefficient": mean,
                        "hac_t": t, "months": len(group), "mean_n": group.n.mean()})
    result = pd.DataFrame(results)
    result.to_csv(OUT/"regression_summary.csv", index=False)
    return result


def make_report(panel, monthly, summary, regressions_table, calendar, manifest):
    plt.style.use("seaborn-v0_8-whitegrid")
    own = monthly[monthly["sample"] == "own"]
    focus = ["gp_assets_ttm", "roa_ttm", "cfo_assets_ttm", "gp_assets_fy", "qmj_like_fy"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for signal in focus:
        s = own[own.signal == signal].sort_values("signal_date")
        axes[0].plot(s.signal_date, s.ic.rolling(36, min_periods=24).mean(), label=signal)
        axes[1].plot(s.signal_date, s.n, label=signal)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_title("Rank IC: 36-month trailing mean (own samples)")
    axes[0].set_ylabel("Spearman correlation with next-month return")
    axes[1].set_title("Number of securities at formation")
    axes[1].set_ylabel("Securities")
    axes[0].legend(fontsize=9, ncol=3)
    fig.savefig(OUT/"ic_and_coverage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for color, signal in zip(["#1764ab", "#d16619", "#208956"],
                             ["fcf_assets_ttm", "cfo_assets_ttm", "gp_assets_ttm"]):
        s = own[own.signal == signal].sort_values("signal_date")
        for metric, style, label in [("ic_cosine_raw", "-", "raw"),
                                      ("ic_cosine_rank_centered", "--", "rank-centered")]:
            ax.plot(s.signal_date, s[metric].rolling(36, min_periods=24).mean(),
                    color=color, linestyle=style, label=f"{signal}: {label}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_title("Cosine IC: 36-month trailing mean (own samples)")
    ax.set_ylabel("Cosine similarity with next-month raw return")
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(OUT/"cosine_comparison.png", dpi=160)
    plt.close(fig)
    main = summary[(summary["sample"] == "own") & (summary.period == "full")].copy()
    main = main.sort_values("mean_ic", ascending=False)
    main["signal"] = main.signal.map(lambda x: x + (" [proxy]" if x in ["ball2015_fy", "ball2016_fy", "cbop_bs_fy", "cbop_cf_fy", "hxz_roe_q", "qmj_like_fy"] else ""))
    display_cols = ["signal", "months_ic", "mean_n", "mean_ic", "ic_hac_t", "months_spread",
                    "spread_arithmetic_annualized", "spread_hac_t", "mean_missing_return_pct"]
    common = summary[(summary["sample"] == "common_core") & (summary.period == "full") & summary.signal.isin(CORE)]
    coef = regressions_table[regressions_table.term.isin(["profitability", "profitability_x_sales_growth"])]
    eligible = panel[panel.eligible]
    unknown = eligible.forward_return.isna().mean()
    extremes = eligible[eligible.return_observed & ((eligible.forward_return > 1) | (eligible.forward_return < -.9))]
    full = summary[(summary["sample"] == "own") & (summary.period == "full")].set_index("signal")
    best_name = full[full.months_ic == len(calendar)].mean_ic.idxmax()
    best = full.loc[best_name]
    coefficients = regressions_table.set_index(["signal", "term"])
    recent_gp = summary[(summary["sample"] == "own") & (summary.period == "2020-2026")
                        & (summary.signal == "gp_assets_fy")].mean_ic.iloc[0]
    table = lambda d: d.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    def results_table(d):
        d = d[display_cols].copy()
        d["spread_arithmetic_annualized"] *= 100
        d["mean_n"] = d["mean_n"].round().astype("Int64")
        d = d.rename(columns={"signal": "Mjera", "months_ic": "Mjeseci IC",
            "mean_n": "Prosječni N", "mean_ic": "Prosječni IC", "ic_hac_t": "IC t(HAC)",
            "months_spread": "Mjeseci Q5−Q1", "spread_arithmetic_annualized": "Q5−Q1 godišnje (%)",
            "spread_hac_t": "Q5−Q1 t(HAC)", "mean_missing_return_pct": "Nepoznati ishodi (%)"})
        return table(d)
    ic_columns = ["signal", "period", "mean_ic_spearman", "ic_spearman_hac_t",
                  "mean_ic_cosine_raw", "ic_cosine_raw_hac_t",
                  "mean_ic_cosine_rank_centered", "ic_cosine_rank_centered_hac_t"]
    ic_comparison = summary[(summary["sample"] == "own") & (summary.period == "full")][ic_columns]
    comparison_table = table(ic_comparison.rename(columns={"mean_ic_spearman": "Spearman",
        "ic_spearman_hac_t": "Spearman t", "mean_ic_cosine_raw": "Cosine raw",
        "ic_cosine_raw_hac_t": "Cosine raw t", "mean_ic_cosine_rank_centered": "Cosine rank-centered",
        "ic_cosine_rank_centered_hac_t": "Cosine rank-centered t"}))
    body = f"""<!doctype html><html lang='hr'><meta charset='utf-8'><title>Quality: početna analiza profitabilne komponente</title>
    <style>body{{font:16px system-ui;margin:40px auto;max-width:1250px;color:#182536;line-height:1.5}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#edf1f5}}.note{{background:#fff3d6;padding:18px}}h1,h2{{color:#153651}}</style>
    <h1>Quality: početna analiza profitabilne komponente</h1>
    <p>Signali: {calendar.signal_date.min():%d.%m.%Y.}–{calendar.signal_date.max():%d.%m.%Y.};
    {len(calendar)} mjesečnih presjeka. Ukupno {len(eligible):,} prihvatljivih opažanja vrijednosnica.
    Analizirana je 21 unaprijed navedena mjera; rezultati su eksplorativni.</p>
    <div class='note'>Ovo je statistička analiza bruto signala na postojećem lokalnom uzorku.
    Budući prinos je omjer vendorovih prilagođenih cijena, a ne verificiran izvršivi ukupni prinos.
    Nema troškova trgovanja, ograničenja shortanja ni troška posudbe. Razlika Q5−Q1 opisna je razlika
    dvaju portfelja. Nepoznati terminalni/delisting ishodi nisu rekonstruirani. Povijesne revizije
    izvještaja, procijenjeni datumi objave i izbor izvorno pokrivenih vrijednosnica ograničavaju zaključke.</div>
    <h2>Što pokazuje prvi presjek</h2>
    <p>{html.escape(best_name)} ima prosječni rangovni IC {best.mean_ic:.4f} (HAC t={best.ic_hac_t:.2f}) i najjači rezultat među mjerama
    koje pokrivaju svih {len(calendar)} mjeseci. Njegova godišnje izražena aritmetička razlika Q5−Q1 iznosi
    {best.spread_arithmetic_annualized*100:.2f} postotnih bodova, uz t={best.spread_hac_t:.2f}. Pozitivno rangiranje samo po sebi ne znači da je potvrđena
    robusna strategija trgovanja krajnjim kvintilima.</p>
    <p>CFO/assets TTM ima IC {full.loc['cfo_assets_ttm','mean_ic']:.4f}, bruto GP/assets TTM {full.loc['gp_assets_ttm','mean_ic']:.4f}, a QMJ-like {full.loc['qmj_like_fy','mean_ic']:.4f}.
    Pozitivni IC prisutan je i u zajedničkom uzorku šest osnovnih mjera.
    QMJ-like ima gotovo nultu prosječnu razliku krajnjih kvintila, što pokazuje da rangovna
    povezanost i ekonomski rezultat krajnjih portfelja mogu dati različite zaključke.</p>
    <p>Uz value, veličinu, momentum i growth kontrole, ROIC ima pozitivan koeficijent profitabilnosti
    (t={coefficients.loc[('roic_fy','profitability'),'hac_t']:.2f}), na manjem i posebno filtriranom uzorku. GP/assets TTM ima t={coefficients.loc[('gp_assets_ttm','profitability'),'hac_t']:.2f}; njegova interakcija
    s rastom prihoda t={coefficients.loc[('gp_assets_ttm','profitability_x_sales_growth'),'hac_t']:.2f}. To su eksplorativni nalazi bez korekcije za višestruke testove.
    Nije potvrđeno da kombinacija visoke profitabilnosti i visokog rasta općenito nosi dodatnu premiju.</p>
    <p>Rezultati ovise o razdoblju: godišnji GP/assets u 2020.–2026. ima prosječni IC {recent_gp:.4f}.
    Pokrivenost se naglo smanjuje početkom 2026., pa završetak uzorka zahtijeva poseban oprez.
    Prioritet sljedećeg koraka: FCF/assets, CFO/assets i GP/assets uz zajedničke uzorke,
    provjeru nepoznatih ishoda, troškova i svježine izvora.</p>
    <h2>Metoda</h2><p>Signal se očitava na kraju mjeseca. Ulaz je zatvaranje sljedeće trgovinske sjednice,
    a izlaz zatvaranje sjednice nakon sljedećeg kraja mjeseca. Formiranje ne zahtijeva poznat budući ishod.
    Univerzum: raspoloživ fundamentalni izvještaj u USD, kompatibilna tržišna kapitalizacija najmanje
    100 mil. USD, sirova cijena najmanje 5 USD i prosječni dnevni promet u 21 sjednici najmanje
    1 mil. USD. Statične sektorske i današnje klasifikacije ne koriste se kao povijesni filtri.
    Identitet je security_id; više klasa istog izdavatelja može biti prisutno.</p>
    <p>Izvorni IC je mjesečna Spearmanova korelacija s budućim mjesečnim prinosom.
    Dodatno se računa cosine raw = s'r / (||s|| ||r||), bez centriranja, rangiranja,
    winsorizacije ili prilagodbe riziku. Cosine rank-centered koristi centrirane rangove signala
    iz formacijskog uzorka i izvorne buduće prinose; prinosi se ne centriraju i ne rangiraju.
    To nisu deklarirane replike Paleologove specifikacije. Cosine raw može sadržavati tržišnu
    komponentu zbog nenulte sredine signala. Sve mjere koriste iste poznate buduće ishode.
    Nulta norma daje nedostajuću vrijednost. Kvintili se određuju prije
    izostavljanja nepoznatih ishoda. Prinos kvintila zahtijeva najmanje 10 članova i najviše 1% nepoznate
    jednake težine; preostale se težine renormaliziraju. IC zahtijeva najmanje 100 poznatih ishoda.
    t-statistike koriste Newey–West/HAC s tri mjesečna laga. Godišnja razlika znači 12 × prosječna
    mjesečna razlika; nije CAGR. Nema korekcije za višestruko testiranje.</p>
    <h2>Spearman i cosine IC</h2>{comparison_table}
    <img src='cosine_comparison.png' alt='Cosine on raw signals and centered signal ranks'>
    <p>Sve tri mjere su mjesečni presjeci, a prikazani rezultat njihov vremenski prosjek.
    Stupci ic i mean_ic ostaju kompatibilni nazivi za Spearman.
    <a href='ic_comparison.csv'>Usporedba za sve uzorke i podrazdoblja</a>.</p>
    <h2>Mjere na vlastitim dostupnim uzorcima</h2>{results_table(main)}
    <p>Stopa nepoznatih budućih prinosa u čitavom prihvatljivom uzorku: {unknown:.2%}.
    Opažanja s mjesečnim prinosom iznad 100% ili ispod −90%: {len(extremes)}; nisu rezana.
    Sva su dostupna u <a href='extreme_returns.csv'>extreme_returns.csv</a>.</p>
    <h2>Zajednički uzorak šest osnovnih mjera</h2>{results_table(common)}
    <p>Za svaku mjeru postoji i usporedba s GP/assets TTM na identičnim vrijednosnicama i mjesecima
    u <a href='summary.csv'>summary.csv</a>, oznaka pair_SIGNAL.</p>
    <h2>Profitabilnost uz value i growth kontrole</h2>{table(coef)}
    <p>Mjesečna presječna regresija objašnjava sirovi budući prinos rangovno standardiziranom
    profitabilnošću, book-to-marketom, veličinom, momentumom 12−1, rastom prihoda i rastom imovine,
    uz interakciju profitabilnosti i rasta prihoda. Najmanje 100 potpunih opažanja po regresiji.
    Koeficijenti nisu dokaz uzročnosti. Sektorska neutralizacija nije provedena.</p>
    <h2>Stabilnost i pokrivenost</h2><img src='ic_and_coverage.png' alt='Rolling IC and coverage'>
    <p><a href='summary.csv'>Sažetak i podrazdoblja</a> · <a href='monthly_diagnostics.csv'>Mjesečni rezultati</a> ·
    <a href='signal_correlations.csv'>Korelacije signala</a> · <a href='coverage_by_year.csv'>Godišnja pokrivenost</a> ·
    <a href='regression_summary.csv'>Regresije</a> · <a href='manifest.json'>Ulazi i parametri</a></p></html>"""
    (OUT/"report.html").write_text(body, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calendar, spy = session_calendar()
    inputs = [fingerprint(p) for p in [PRICES, FUND, EXTFILE]]
    calendar.to_csv(OUT/"formation_calendar.csv", index=False)
    panel = build_monthly(calendar, inputs)
    eligible = panel[panel.eligible].copy()
    log("Calculating rank IC and quintile diagnostics")
    parts = [diagnostics(eligible, "own")]
    common = eligible[np.isfinite(eligible[CORE]).all(axis=1)]
    parts.append(diagnostics(common, "common_core"))
    # Each measure and the anchor are compared on exactly the same formation set.
    global SIGNALS
    all_signals = SIGNALS.copy()
    for signal in all_signals:
        if signal == "gp_assets_ttm":
            continue
        subset = eligible[np.isfinite(eligible[[signal, "gp_assets_ttm"]]).all(axis=1)]
        SIGNALS = {k: all_signals[k] for k in ["gp_assets_ttm", signal]}
        parts.append(diagnostics(subset, f"pair_{signal}"))
    SIGNALS = all_signals
    monthly = pd.concat(parts, ignore_index=True)
    monthly.to_csv(OUT/"monthly_diagnostics.csv", index=False)
    summary = summarize(monthly)
    summary.to_csv(OUT/"summary.csv", index=False)
    summary[["sample", *ic_columns_for_export()]].to_csv(OUT/"ic_comparison.csv", index=False)
    log("Estimating cross-sectional regressions with value and growth controls")
    reg = regressions(eligible)
    coverage = eligible.groupby(eligible.signal_date.dt.year)[list(SIGNALS)].agg(["count", "size"])
    coverage.columns = [f"{a}_{b}" for a,b in coverage.columns]
    coverage.to_csv(OUT/"coverage_by_year.csv")
    correlations = [g[list(SIGNALS)].corr(method="spearman").stack().rename("rho").reset_index()
                    for _, g in eligible.groupby("signal_date")]
    cor = pd.concat(correlations).groupby(["level_0", "level_1"]).rho.agg(["mean", "count"])
    cor.to_csv(OUT/"signal_correlations.csv")
    extremes = eligible[eligible.return_observed & ((eligible.forward_return > 1) | (eligible.forward_return < -.9))]
    extremes[["security_id", "ticker", "signal_date", "entry_date", "exit_date", "entry_price",
              "exit_price", "forward_return", "FUND_MARKET_CAP_USD_COMPAT"]].sort_values(
        "forward_return", ascending=False).to_csv(OUT/"extreme_returns.csv", index=False)
    mapping = pd.DataFrame([{"signal": k, "source_column": v} for k,v in SIGNALS.items()])
    mapping.to_csv(OUT/"signal_dictionary.csv", index=False)
    manifest = {"inputs": inputs, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {"min_raw_price": 5, "min_adv21_usd": 1_000_000, "min_cap_usd": 100_000_000,
        "min_cross_section": 100, "max_missing_sleeve_weight": .01, "hac_lags": 3,
        "core_common_signals": CORE, "formation": "month-end close", "entry": "next session close",
        "exit": "session after next month-end close", "costs": "not modeled",
        "return_basis": "parent adjusted-close proxy; unknown delisting outcomes remain missing"},
        "monthly_panel_rows": len(panel), "eligible_rows": len(eligible),
        "first_signal_date": str(calendar.signal_date.min().date()),
        "last_signal_date": str(calendar.signal_date.max().date()), "months": len(calendar),
        "packages": {"python": sys.version, "polars": pl.__version__, "pandas": pd.__version__}}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_report(panel, monthly, summary, reg, calendar, manifest)
    log(summary[(summary["sample"]=="own") & (summary.period=="full")][[
        "signal", "months_ic", "mean_n", "mean_ic", "ic_hac_t", "spread_arithmetic_annualized", "spread_hac_t"]].to_string(index=False))
    log(f"Report: {OUT/'report.html'}")


if __name__ == "__main__":
    main()
