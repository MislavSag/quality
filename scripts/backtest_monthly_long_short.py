"""Monthly portfolio accounting using read-only local monthly research inputs.

Costs and short availability are explicit assumptions, not historical IBKR data.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.api as sm

import analyze_quality as analysis

ROOT = analysis.ROOT
OUT = ROOT/"results"/"monthly_long_short"
CONFIG = ROOT/"config"/"monthly_long_short.json"


def post_cost_nav(nav, old_positions, weights, cost):
    """Solve NAV_after_cost = NAV_before_cost - c * traded dollar notional."""
    ids = old_positions.index.union(weights.index)
    old = old_positions.reindex(ids, fill_value=0).to_numpy()
    target = weights.reindex(ids, fill_value=0).to_numpy()
    after = nav
    for _ in range(100):
        traded = np.abs(target*after-old).sum()
        updated = nav-cost*traded
        if abs(updated-after) <= max(1e-8, abs(nav)*1e-12):
            after = updated
            break
        after = updated
    traded = np.abs(target*after-old).sum()
    assert np.isclose(after, nav-cost*traded, atol=1e-6, rtol=1e-11)
    if after <= 0:
        raise RuntimeError("Portfolio exhausted its capital")
    return after, traded


def availability_uniform(sid, date, seed):
    text = f"{seed}|{sid}|{date:%Y-%m}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")/2**64


def formation_books(data, signal, config):
    books = []
    for month, group in data[data.eligible].groupby("signal_date", sort=True):
        group = group[np.isfinite(group[signal])].copy()
        if len(group) < 100:
            raise RuntimeError(f"Insufficient formation sample: {month} {signal}")
        # Average ranks put identical signals in the same quintile.
        percentile = (group[signal].rank(method="average")-.5)/len(group)
        tail = config["tail_fraction"]
        book = group[(percentile < tail) | (percentile >= 1-tail)].copy()
        book["side"] = np.where(percentile.loc[book.index] >= 1-tail, 1, -1)
        book["requested_weight"] = 0.0
        for side in [1,-1]:
            mask = book.side == side
            assert mask.sum() >= 10
            desired = config["long_weight"] if side == 1 else config["short_weight"]
            book.loc[mask,"requested_weight"] = desired/mask.sum()
        book["entry_executable"] = (np.isfinite(book.entry_price) & (book.entry_price > 0)
                                      & book.entry_contract_ok.eq(True))
        book["availability_u"] = [availability_uniform(sid,month,config["availability_seed"])
                                   for sid in book.security_id]
        assert (book.signal_date < book.entry_date).all()
        assert (book.entry_date < book.exit_date).all()
        books.append(book.set_index("security_id", drop=False))
    return books


def simulate(books, scenario, config, adverse_missing=False, record_holdings=True):
    nav = float(config["initial_nav"])
    old = pd.Series(dtype=float)
    records, holdings = [], []
    cost = scenario["trading_bps"]/10000
    for i, book in enumerate(books):
        before = nav
        target = book.requested_weight.where(book.entry_executable, 0.0).copy()
        is_short = book.side == -1
        blocked = is_short & (book.availability_u < scenario["unavailable_short_fraction"])
        target[blocked] = 0
        cap = scenario["short_adv_fraction_cap"]
        if cap is not None:
            max_weight = book.PV_AVG_DOLLAR_VOLUME_21D*cap/config["initial_nav"]
            target.loc[is_short] = -np.minimum(-target.loc[is_short], max_weight.loc[is_short])
        # Single-side research books opt out; existing long-short calls retain
        # their original balancing and rejection policy.
        if config.get('balance_sides', True):
            accepted = min(target[target > 0].sum(), -target[target < 0].sum())
            for side in [1,-1]:
                mask = target*side > 0
                gross = abs(target[mask]).sum()
                if gross:
                    target[mask] *= accepted/gross
            assert abs(target.sum()) < 1e-10
        after, traded = post_cost_nav(nav, old, target, cost)
        positions = target*after
        known = book.return_observed & np.isfinite(book.forward_return)
        returns = book.forward_return.where(known, 0).copy()
        if adverse_missing:
            returns.loc[~known & (book.side == 1)] = -1
            returns.loc[~known & (book.side == -1)] = 1
        days = (book.exit_date.iloc[0]-book.entry_date.iloc[0]).days
        end_positions = positions*(1+returns)
        short_start = -positions.clip(upper=0).sum()
        short_end = -end_positions.clip(upper=0).sum()
        borrow = (short_start+short_end)/2*scenario["borrow_rate"]*days/365
        pnl = (positions*returns).sum()
        nav = after+pnl-borrow
        final_close_cost = 0.0
        if i == len(books)-1:
            final_close_cost = abs(end_positions).sum()*cost
            nav -= final_close_cost
        if nav <= 0:
            raise RuntimeError("Portfolio NAV became nonpositive")
        unknown_long_weight = target[(book.side == 1) & ~known].sum()
        unknown_short_weight = -target[(book.side == -1) & ~known].sum()
        records.append({"signal_date": book.signal_date.iloc[0], "entry_date": book.entry_date.iloc[0],
            "exit_date": book.exit_date.iloc[0], "nav_start": before, "nav_after_entry_cost": after,
            "nav_end": nav, "return": nav/before-1, "gross_pnl": pnl,
            "long_pnl": (positions[positions > 0]*returns[positions > 0]).sum(),
            "short_pnl": (positions[positions < 0]*returns[positions < 0]).sum(),
            "entry_trading_cost": cost*traded, "final_liquidation_cost": final_close_cost,
            "borrow_cost": borrow, "traded_notional": traded, "turnover_full_notional": traded/before,
            "long_gross_weight": target.clip(lower=0).sum(), "short_gross_weight": -target.clip(upper=0).sum(),
            "long_n": int((target>0).sum()), "short_n": int((target<0).sum()),
            "short_requested_n": int(is_short.sum()), "short_blocked_n": int(blocked.sum()),
            "entry_price_rejected_n": int((~book.entry_executable).sum()),
            "short_reduced_n": int((is_short & (abs(target)+1e-12 < abs(book.requested_weight))).sum()),
            "unknown_long_weight": unknown_long_weight, "unknown_short_weight": unknown_short_weight})
        if not adverse_missing and record_holdings:
            h = book[["security_id","ticker","signal_date","entry_date","exit_date","side",
                      "requested_weight","entry_executable","forward_return","return_observed"]].copy()
            h["weight"] = target
            h["short_blocked_assumption"] = blocked
            h["return_used"] = returns
            holdings.append(h.reset_index(drop=True))
        old = end_positions
    return pd.DataFrame(records), pd.concat(holdings, ignore_index=True) if holdings else None


def performance(rows, initial_nav, benchmark):
    r = rows["return"]
    elapsed = (rows.exit_date.iloc[-1]-rows.entry_date.iloc[0]).days/365.25
    curve = np.r_[1, rows.nav_end.to_numpy()/initial_nav]
    dd = curve/np.maximum.accumulate(curve)-1
    fit = sm.OLS(r.to_numpy(), sm.add_constant(benchmark)).fit(cov_type="HAC",cov_kwds={"maxlags":3})
    return {"months":len(r),"cagr":curve[-1]**(1/elapsed)-1,"annual_vol":r.std(ddof=1)*np.sqrt(12),
        "sharpe_cash0":r.mean()/r.std(ddof=1)*np.sqrt(12),"monthly_max_drawdown":dd.min(),
        "mean_monthly_return":r.mean(),"positive_month_fraction":(r>0).mean(),
        "spy_beta":fit.params[1],"spy_intercept_annualized":fit.params[0]*12,
        "spy_intercept_hac_t":fit.tvalues[0],"mean_turnover_full_notional":rows.turnover_full_notional.mean(),
        "trading_cost_sum":(rows.entry_trading_cost+rows.final_liquidation_cost).sum(),
        "borrow_cost_sum":rows.borrow_cost.sum(),"mean_short_gross":rows.short_gross_weight.mean(),
        "unknown_long_weight_mean":rows.unknown_long_weight.mean(),
        "unknown_short_weight_mean":rows.unknown_short_weight.mean(),
        "short_blocks":rows.short_blocked_n.sum(),"entry_price_rejections":rows.entry_price_rejected_n.sum()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config = json.loads(CONFIG.read_text())
    assert config["cash_interest_rate"] == 0, "A cash-interest model must be implemented before changing this parameter"
    assert abs(config["long_weight"]+config["short_weight"]) < 1e-12
    initial_manifest = json.loads((analysis.OUT/"manifest.json").read_text())
    assert initial_manifest["inputs"] == [analysis.fingerprint(p) for p in [analysis.PRICES,analysis.FUND,analysis.EXTFILE]]
    columns = ["security_id","ticker","signal_date","entry_date","exit_date","eligible","entry_price",
        "entry_contract_ok","return_observed","forward_return","PV_AVG_DOLLAR_VOLUME_21D",*config["signals"]]
    data = pd.read_parquet(analysis.CACHE/"monthly_panel.parquet",columns=columns)
    calendar = pd.read_csv(analysis.OUT/"formation_calendar.csv",parse_dates=["signal_date","entry_date","exit_date"])
    spy = ds.dataset(str(analysis.PRICES)).to_table(columns=["trade_date","PV_CLOSE"],
        filter=ds.field("ticker")=="SPY").to_pandas(date_as_object=False).set_index("trade_date").PV_CLOSE
    benchmark = np.array([spy.loc[row.exit_date]/spy.loc[row.entry_date]-1 for row in calendar.itertuples()])
    months, stats, all_holdings = [], [], []
    for signal in config["signals"]:
        books = formation_books(data,signal,config)
        for scenario in config["scenarios"]:
            for adverse in [False,True]:
                label = "adverse_unknown" if adverse else "flat_unknown"
                rows, holdings = simulate(books,scenario,config,adverse)
                assert len(rows)==len(calendar)
                assert np.array_equal(rows.entry_date.to_numpy(),calendar.entry_date.to_numpy())
                rows["signal"],rows["scenario"],rows["missing_policy"] = signal,scenario["name"],label
                rows["spy_return"] = benchmark
                stats.append({"signal":signal,"scenario":scenario["name"],"missing_policy":label,
                              **performance(rows,config["initial_nav"],benchmark)})
                months.append(rows)
                if holdings is not None:
                    holdings["signal"],holdings["scenario"] = signal,scenario["name"]
                    all_holdings.append(holdings)
            analysis.log(f"Completed {signal} / {scenario['name']}")
    monthly = pd.concat(months,ignore_index=True)
    summary = pd.DataFrame(stats)
    monthly.to_csv(OUT/"monthly_returns.csv",index=False)
    summary.to_csv(OUT/"summary.csv",index=False)
    pd.concat(all_holdings,ignore_index=True).to_parquet(OUT/"holdings.parquet",index=False)
    (OUT/"config.json").write_text(CONFIG.read_text())
    receipt = {"analysis":"monthly_long_short_v1","config":config,
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_inputs":initial_manifest["inputs"],"monthly_cache":str(analysis.CACHE/"monthly_panel.parquet"),
        "availability_data":"not observed; deterministic 10% rejection and ADV capacity scenario",
        "missing_outcomes":"flat-return mark/liquidation assumption plus adverse directional sensitivity",
        "return_basis":"vendor adjusted-price proxy; not verified delisting or executable fills",
        "borrow_accrual":"annual rate * mean(start,end short market value) * actual days / 365",
        "financing":"cash interest, collateral rebate and margin funding assumed zero",
        "trade_cost":"one-way bps per full absolute traded notional; fixed-point post-cost targets; includes final liquidation",
        "drawdown_frequency":"monthly endpoints only"}
    (OUT/"manifest.json").write_text(json.dumps(receipt,indent=2))
    report(monthly,summary,config,calendar)
    analysis.log(summary[summary.missing_policy=="flat_unknown"][["signal","scenario","cagr","sharpe_cash0",
        "monthly_max_drawdown","mean_turnover_full_notional","spy_beta"]].to_string(index=False))


def report(monthly,summary,config,calendar):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    colors=["#1764ab","#d16619","#208956"]
    for color,signal in zip(colors,config["signals"]):
        for scenario,style in [("gross_unconstrained","--"),("cost10_borrow3","-")]:
            x=monthly[(monthly.signal==signal)&(monthly.scenario==scenario)&(monthly.missing_policy=="flat_unknown")]
            equity=x.nav_end/config["initial_nav"]
            axes[0].plot(x.exit_date,equity,color=color,linestyle=style,label=f"{signal}: {scenario}")
            if style=="-":
                curve=np.r_[1,equity.to_numpy()]
                axes[1].plot(x.exit_date,(curve/np.maximum.accumulate(curve)-1)[1:],color=color,label=signal)
    axes[0].set_title("Monthly long-short: +100% / -100% target, gross and assumed costs")
    axes[0].set_ylabel("NAV / initial NAV")
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].set_title("Monthly drawdown: 10 bps trading + 3% annual borrow assumption")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=9)
    fig.savefig(OUT/"equity_and_drawdown.png",dpi=160)
    plt.close(fig)
    show=summary[["signal","scenario","missing_policy","cagr","annual_vol","sharpe_cash0",
        "monthly_max_drawdown","mean_turnover_full_notional","mean_short_gross","spy_beta"]].copy()
    for c in ["cagr","annual_vol","monthly_max_drawdown"]:
        show[c]*=100
    display=show.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)
    baseline=summary[summary.missing_policy=="flat_unknown"]
    brief=(baseline[baseline.scenario.isin(["gross_unconstrained","cost5_borrow0p5","cost10_borrow3"])]
        .pivot(index="signal",columns="scenario",values="cagr")*100)
    brief=brief[["gross_unconstrained","cost5_borrow0p5","cost10_borrow3"]].rename(columns={
        "gross_unconstrained":"Bruto CAGR (%)","cost5_borrow0p5":"5 bps + 0,5% posudba: CAGR (%)",
        "cost10_borrow3":"10 bps + 3% posudba: CAGR (%)"})
    brief_html=brief.to_html(float_format=lambda x:f"{x:.2f}",border=0)
    gross_sharpe=baseline[baseline.scenario=="gross_unconstrained"].sharpe_cash0
    document=f"""<!doctype html><html lang='hr'><meta charset='utf-8'><title>Quality: mjesečni long-short profitabilne komponente</title>
    <style>body{{font:16px system-ui;max-width:1350px;margin:40px auto;color:#182536;line-height:1.5}}
    table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
    img{{max-width:100%}}.note{{padding:18px;background:#fff3d6}}</style>
    <h1>Quality: mjesečni long-short profitabilne komponente</h1>
    <p>Tri unaprijed odabrane mjere, {len(calendar)} mjeseci. Formiranje od {calendar.signal_date.min():%d.%m.%Y.}
    do {calendar.signal_date.max():%d.%m.%Y.}. Zadnji izlaz {calendar.exit_date.max():%d.%m.%Y.}.</p>
    <div class='note'>Lokalni backtest uz eksplicitne pretpostavke. Cijene su vendorov prilagođeni proxy.
    Stvarna dostupnost shorta i povijesne naknade nisu potvrđene. Nepoznati izlazi u osnovnoj verziji
    imaju pretpostavljen prinos nula i pretpostavljeno zatvaranje; ne rekonstruira se delisting naknada.
    Drugi scenarij za takve ishode pretpostavlja pad 100% za long i rast 100% za short.
    Rezultati nisu verificirani izvršivi neto prinosi.</div>
    <h2>Prvi nalaz</h2>{brief_html}
    <p>Bruto Sharpe za tri mjere kreće se od {gross_sharpe.min():.2f} do {gross_sharpe.max():.2f}.
    Rezultat je osjetljiv na pretpostavljenu cijenu posudbe i obračun nepoznatih izlaza.
    Različite razine troškova su analiza osjetljivosti, a ne optimizacija ili procjena stvarnih
    povijesnih naknada. Pozitivan bruto rezultat samostalno ne potvrđuje robusnu strategiju.</p>
    <h2>Pravila</h2><p>Na kraju mjeseca odabire se gornjih 20% i donjih 20% signala,
    uz jednake težine unutar svake strane. Cilj je long +100% i short −100% NAV-a (bruto 200%).
    Trgovanje je na zatvaranju sljedeće sjednice, a držanje do odgovarajućeg sljedećeg rebalansa.
    Nema dnevnog vraćanja težina; težine između rebalansa slobodno driftaju. Filtri su isti kao
    u početnoj analizi: USD fundamenti, kapitalizacija ≥100 mil. USD, cijena ≥5 USD i ADV21 ≥1 mil. USD.
    Budući prinos ne sudjeluje u odabiru. Nedostupna ulazna cijena odbija nalog.</p>
    <p>Početni NAV je 10 mil. USD. Trošak se primjenjuje na stvarnu apsolutnu promjenu pozicija,
    uz prethodne pozicije driftane tržišnim prinosom. Trošak ulaza rješava se zajedno s ciljnim
    pozicijama nakon troška; završno zatvaranje također se naplaćuje. Posudba se računa na prosječni
    početni i završni short notional kroz stvarni broj dana. Kolateralna kamata, cash kamata i
    financiranje margine pretpostavljeni su nula.</p>
    <p>Scenariji: bruto; 5 bps po prometu + 0,5% godišnja posudba; 10 bps + 3%; 25 bps + 5%; te dodatno
    10% deterministički nedostupnih shortova i ograničenje short pozicije na 1% ADV21 u odnosu
    na referentni NAV od 10 mil. USD. To ograničenje nije opažena brokerova shortable količina.
    Odbijeni shortovi se ne zamjenjuju drugima; long strana smanjuje se na prihvaćenu short izloženost.</p>
    <h2>Rezultati</h2><p>CAGR, volatilnost i drawdown u tablici su postoci. Sharpe koristi cash stopu nula.
    Drawdown je mjesečni i ne obuhvaća gubitke unutar mjeseca. Promet je puni trgovani notional/NAV,
    bez dijeljenja s dva. SPY beta je opisna regresija na istom razdoblju.</p>{display}
    <img src='equity_and_drawdown.png' alt='Monthly equity and drawdown under stated assumptions'>
    <p><a href='summary.csv'>Sažetak</a> · <a href='monthly_returns.csv'>Mjesečni obračun</a> ·
    <a href='holdings.parquet'>Pozicije</a> · <a href='config.json'>Parametri</a> ·
    <a href='manifest.json'>Izvori i pretpostavke</a></p></html>"""
    (OUT/"report.html").write_text(document,encoding="utf-8")


if __name__=="__main__":
    main()
