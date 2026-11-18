"""Monthly stock-level profitability bundle, with controlled comparisons.

Public RMW is kept separate as a return-series basket, never a stock score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_quality as a
import backtest_monthly_long_short as bt
import compare_public_rmw as public_stats
import analyze_regimes as regimes

ROOT = a.ROOT
OUT = ROOT/'results'/'quality_profitability_bundle'
CONFIG_PATH = ROOT/'config'/'quality_profitability_bundle.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
COMPONENTS = list(CONFIG['components'])
SIGNALS = ['bundle4','bundle3_common4','fcf_common4','cfo_common4','gp_common4','ff_common4','bundle3']
DISPLAY = {'bundle4':'Bundle 4 (sva četiri podatka)',
           'bundle3_common4':'Bundle 3 na istim dionicama',
           'fcf_common4':'FCF/imovina na istim dionicama',
           'cfo_common4':'CFO/imovina na istim dionicama',
           'gp_common4':'Bruto dobit/imovina na istim dionicama',
           'ff_common4':'FF OP/kapital na istim dionicama',
           'bundle3':'Bundle 3 (širi univerzum)',
           'basket4_public_rmw':'Košarica 4 prinosa s javnim RMW-om'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_scores(data):
    """Only formation fields affect scores; unknown future outcomes are irrelevant."""
    scores = data[['security_id','ticker','signal_date','eligible']].copy()
    finite = np.isfinite(data[COMPONENTS])
    scores['component_count'] = finite.sum(axis=1).astype('int8')
    complete4 = data.eligible & finite.all(axis=1)
    complete3 = data.eligible & finite[COMPONENTS[:3]].all(axis=1)
    for name in COMPONENTS:
        scores['rank4_'+name] = np.nan
    for signal in ['bundle4','bundle3_common4','bundle3']:
        scores[signal] = np.nan
    groups = data.groupby('signal_date',sort=True).groups
    for date,index in groups.items():
        idx = index[complete4.loc[index]]
        if len(idx) >= CONFIG['minimum_cross_section']:
            ranks = (data.loc[idx,COMPONENTS].rank(method='average')-.5)/len(idx)
            scores.loc[idx,['rank4_'+s for s in COMPONENTS]] = ranks.to_numpy()
            weights = np.array(list(CONFIG['components'].values()))
            scores.loc[idx,'bundle4'] = 2*(ranks.to_numpy()@weights)-1
            scores.loc[idx,'bundle3_common4'] = 2*ranks[COMPONENTS[:3]].mean(axis=1)-1
        idx3 = index[complete3.loc[index]]
        if len(idx3) >= CONFIG['minimum_cross_section']:
            ranks3 = (data.loc[idx3,COMPONENTS[:3]].rank(method='average')-.5)/len(idx3)
            scores.loc[idx3,'bundle3'] = 2*ranks3.mean(axis=1)-1
    for signal in ['bundle4','bundle3_common4','bundle3']:
        scores[signal+'_score'] = 50*(scores[signal]+1)
    return scores


def coverage(data,scores):
    records = []
    for date,index in data.groupby('signal_date').groups.items():
        g = scores.loc[index]
        records.append(dict(signal_date=date,eligible_n=int(g.eligible.sum()),
            complete4_n=int(g.bundle4.notna().sum()),complete3_n=int(g.bundle3.notna().sum()),
            **{s+'_n':int((data.loc[index,'eligible'] & np.isfinite(data.loc[index,s])).sum()) for s in COMPONENTS}))
    return pd.DataFrame(records)


def diagnostic_rows(data):
    rows = []
    for signal in SIGNALS:
        for date,g in data[data.eligible & np.isfinite(data[signal])].groupby('signal_date'):
            known = g[g.return_observed & np.isfinite(g.forward_return)]
            rows.append(dict(sample='bundle_research',signal=signal,signal_date=date,n=len(g),
                n_observed=len(known),ic_spearman=known[signal].corr(known.forward_return,method='spearman'),
                **a.cosine_metrics(g,signal)))
    raw = pd.DataFrame(rows)
    summary = []
    for (signal,period),group in (( (s,p),g[regimes.period_mask(pd.DatetimeIndex(g.signal_date),p)])
            for s,g in raw.groupby('signal') for p in ['full','early','late']):
        row = dict(signal=signal,period=period,mean_n=group.n.mean())
        for metric in ['ic_spearman','ic_cosine_raw','ic_cosine_rank_centered']:
            mean,t = a.hac_mean(group[metric])
            row['mean_'+metric],row[metric+'_hac_t'] = mean,t
        summary.append(row)
    return raw,pd.DataFrame(summary)


def simulate_all(data,benchmark):
    config = json.loads(bt.CONFIG.read_text())
    config['signals'] = SIGNALS
    all_rows,all_stats,all_holdings = [],[],[]
    for signal in SIGNALS:
        books = bt.formation_books(data,signal,config)
        for scenario in config['scenarios']:
            for adverse in [False,True]:
                policy = 'adverse_unknown' if adverse else 'flat_unknown'
                rows,holdings = bt.simulate(books,scenario,config,adverse)
                rows['signal'],rows['scenario'],rows['missing_policy'] = signal,scenario['name'],policy
                rows['spy_return'] = benchmark
                all_rows.append(rows)
                for period in ['full','early','late']:
                    selection = regimes.period_mask(pd.DatetimeIndex(rows.signal_date),period)
                    part = rows.loc[selection].copy()
                    # Reset only the performance base for contiguous subperiods.
                    stats = bt.performance(part,float(part.nav_start.iloc[0]),benchmark[selection])
                    all_stats.append(dict(signal=signal,scenario=scenario['name'],missing_policy=policy,period=period,**stats))
                if holdings is not None and signal in ['bundle4','bundle3']:
                    holdings['signal'],holdings['scenario'] = signal,scenario['name']
                    all_holdings.append(holdings)
        a.log('Completed '+signal)
    return pd.concat(all_rows,ignore_index=True),pd.DataFrame(all_stats),pd.concat(all_holdings,ignore_index=True)


def return_basket():
    old = pd.read_csv(ROOT/'results/monthly_long_short/monthly_returns.csv',
                       parse_dates=['signal_date','entry_date','exit_date'])
    old = old[(old.scenario=='gross_unconstrained') & (old.missing_policy=='flat_unknown')]
    matrix = old.pivot(index='signal_date',columns='signal',values='return')
    public = pd.read_csv(ROOT/'results/public_comparison/rmw_aligned_windows.csv',parse_dates=['signal_date'])
    public = public.set_index('signal_date')
    matrix['public_rmw'] = public.RMW_daily_compounded.reindex(matrix.index)
    assert matrix.notna().all().all() and len(matrix.columns)==4
    matrix['basket4_public_rmw'] = matrix.mean(axis=1)
    dates = old.drop_duplicates('signal_date').set_index('signal_date')[['entry_date','exit_date']]
    matrix = dates.join(matrix)
    rows = []
    for period in ['full','early','late']:
        g = matrix[regimes.period_mask(matrix.index,period)]
        stats = public_stats.stats('basket4_public_rmw',g.basket4_public_rmw,period)
        # Same actual-day CAGR convention as local portfolio summaries.
        years = (g.exit_date.iloc[-1]-g.entry_date.iloc[0]).days/365.25
        stats['cagr'] = (1+g.basket4_public_rmw).prod()**(1/years)-1
        stats['period'] = period
        rows.append(stats)
    return matrix.reset_index(),pd.DataFrame(rows),matrix.drop(columns=['entry_date','exit_date']).corr()


def compare_increments(monthly,diagnostics):
    rows = []
    scenario_names = ['gross_unconstrained','cost5_borrow0p5','cost10_borrow3']
    for scenario in scenario_names:
        for policy in ['flat_unknown','adverse_unknown']:
            path = monthly[(monthly.scenario==scenario)&(monthly.missing_policy==policy)]
            wide = path.pivot(index='signal_date',columns='signal',values='return')
            for comparator in ['bundle3_common4','fcf_common4','cfo_common4','gp_common4','ff_common4']:
                for period in ['full','early','late']:
                    part = wide[regimes.period_mask(wide.index,period)]
                    y = part.bundle4-part[comparator]
                    fit = regimes.calendar_fit(y,np.ones((len(y),1)))
                    rows.append(dict(metric='return',scenario=scenario,missing_policy=policy,
                        comparator=comparator,period=period,months=len(y),mean_difference=fit['beta'][0],
                        annual_mean_difference=12*fit['beta'][0],hac_t=fit['t'][0],p=fit['p'][0]))
    for metric in ['ic_spearman','ic_cosine_raw','ic_cosine_rank_centered']:
        wide = diagnostics.pivot(index='signal_date',columns='signal',values=metric)
        for comparator in ['bundle3_common4','fcf_common4','cfo_common4','gp_common4','ff_common4']:
            for period in ['full','early','late']:
                part = wide[regimes.period_mask(wide.index,period)]
                y = part.bundle4-part[comparator]
                fit = regimes.calendar_fit(y,np.ones((len(y),1)))
                rows.append(dict(metric=metric,scenario='not_applicable',missing_policy='known_outcomes_only',
                    comparator=comparator,period=period,months=fit['n'],mean_difference=fit['beta'][0],
                    annual_mean_difference=np.nan,hac_t=fit['t'][0],p=fit['p'][0]))
    result = pd.DataFrame(rows)
    result['q_bh'] = np.nan
    from statsmodels.stats.multitest import multipletests
    for _,group in result.groupby(['metric','scenario','missing_policy','period']):
        result.loc[group.index,'q_bh'] = multipletests(group.p,method='fdr_bh')[1]
    return result


def conditional_results(monthly):
    features = pd.read_csv(ROOT/'results/regimes/regime_features.csv',parse_dates=['signal_date']).set_index('signal_date')
    records = []
    for key,path in monthly[monthly.signal.isin(['bundle4','bundle3'])].groupby(['signal','scenario','missing_policy']):
        y = path.set_index('signal_date')['return'].reindex(features.index)
        for regime in regimes.REGIMES:
            state = features[regime+'_state'].where(features.common_sample)
            for level in [0,1]:
                values = y[state.eq(level)]
                records.append(dict(signal=key[0],scenario=key[1],missing_policy=key[2],
                    regime=regime,state=level,**regimes.annual_stats(values)))
    return pd.DataFrame(records)


def make_plot(monthly,basket):
    fig,axes = plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    colors = {'bundle4':'#1c5d99','bundle3_common4':'#b96217','bundle3':'#2b8a66'}
    for signal,color in colors.items():
        for scenario,style in [('gross_unconstrained','-'),('cost10_borrow3','--')]:
            g = monthly[(monthly.signal==signal)&(monthly.scenario==scenario)&(monthly.missing_policy=='flat_unknown')]
            axes[0].plot(g.exit_date,g.nav_end/g.nav_start.iloc[0],color=color,ls=style,label=f'{signal}: {scenario}')
            if scenario=='gross_unconstrained':
                curve = np.r_[1,g.nav_end.to_numpy()/g.nav_start.iloc[0]]
                axes[1].plot(g.exit_date,(curve/np.maximum.accumulate(curve)-1)[1:],color=color,label=signal)
    equity = (1+basket.basket4_public_rmw).cumprod()
    axes[0].plot(basket.exit_date,equity,color='#7464a1',ls=':',label='4-return basket incl. public RMW (gross diagnostic)')
    axes[0].set_title('Profitability bundle: monthly equal-weight tails (+100% / −100%)')
    axes[0].set_ylabel('NAV / initial NAV')
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].set_title('Gross monthly endpoint drawdown (stock-level bundles)')
    axes[1].set_ylabel('Drawdown')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT/'bundle_equity.png',dpi=150)
    plt.close(fig)


def report(summary,ic,cover,increments,basket_stats,correlations):
    main = summary[(summary.period=='full') & summary.missing_policy.eq('flat_unknown')
                   & summary.scenario.isin(['gross_unconstrained','cost5_borrow0p5','cost10_borrow3'])]
    display = main[['signal','scenario','cagr','annual_vol','sharpe_cash0','monthly_max_drawdown','spy_beta']].copy()
    display['signal'] = display.signal.map(DISPLAY)
    for name in ['cagr','annual_vol','monthly_max_drawdown']:
        display[name] *= 100
    first = summary[(summary.signal=='bundle4') & (summary.period=='full')
                    & (summary.scenario=='gross_unconstrained') & (summary.missing_policy=='flat_unknown')].iloc[0]
    primary_diff = increments[(increments.metric=='return')&(increments.comparator=='bundle3_common4')
        &(increments.period=='full')&(increments.scenario=='gross_unconstrained')
        &(increments.missing_policy=='flat_unknown')].iloc[0]
    warning = ('Nema uvjerljivog dokaza da četvrta komponenta poboljšava prinos na istom univerzumu.'
               if primary_diff.p>=.05 else 'Razlika je eksplorativna; provjeriti q i stabilnost podrazdoblja.')
    document = f'''<!doctype html><html lang="hr"><meta charset="utf-8"><title>Quality: profitability bundle</title>
    <style>body{{font:15px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;line-height:1.55;color:#203044}}
    table{{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
    td:first-child,th:first-child{{text-align:left}}img{{width:100%}}.note{{background:#fff3d6;padding:16px}}</style>
    <h1>Quality: PROFIT_BUNDLE4 — profitabilna komponenta</h1>
    <p>Četiri komponente s jednakim težinama: FCF/imovina, CFO/imovina, bruto dobit/imovina (TTM)
    i godišnja FF2015 operativna profitabilnost/kapital. Nema optimizacije težina prema prinosima.</p>
    <p>Score = 100 × prosjek četiri percentilna ranga. Svaki rang je (prosječni rang − 0,5)/N,
    izračunat na kraju mjeseca samo među tada podobnim dionicama sa sva četiri podatka.
    Više znači profitabilnije u odnosu na druge dionice tog mjeseca. Interni signal bundle4 = score/50 − 1;
    ta verzija ima sredinu nula na formacijskom uzorku i koristi se za IC i long–short rangiranje.</p>
    <div class="note">Javni RMW je prinos portfelja, a ne podatak pojedine dionice. Četvrta komponenta
    indikatora zato je naš lokalni FF2015 OP/kapital. To nije replika javnog RMW-a: lokalna formula,
    računovodstveni izvori, tretman R&amp;D-a i nazivnik razlikuju se od aktualne Frenchove biblioteke,
    koja uključuje manjinski udio u nazivniku. Izvorne restrikcije podataka nisu popuštene.
    <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html">Definicija OP-a</a>;
    <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html">Konstrukcija RMW-a</a>.</div>
    <h2>Rezultat i pokrivenost</h2><p>Strogi bundle 4 pokriva prosječno {cover.complete4_n.mean():.0f} dionice mjesečno
    (min. {cover.complete4_n.min()}, max. {cover.complete4_n.max()}, zadnji mjesec {cover.complete4_n.iloc[-1]}).
    Bundle 3 na širem univerzumu pokriva prosječno {cover.complete3_n.mean():.0f}.
    Nedostajući četvrti podatak daje nedostajući score, bez imputacije i bez promjene težina.</p>
    <p>Bundle 4: bruto CAGR {first.cagr:.2%}, Sharpe {first.sharpe_cash0:.2f}, mjesečni maksimalni drawdown
    {first.monthly_max_drawdown:.2%}. Dodavanje četvrte komponente u odnosu na bundle 3 na istim dionicama
    mijenja godišnji aritmetički prosjek za {primary_diff.annual_mean_difference:.2%}
    (HAC t={primary_diff.hac_t:.2f}, p={primary_diff.p:.3f}). {warning}</p>
    <p>FCF i CFO znatno su povezani (prosječna mjesečna Spearman korelacija oko 0,83 na parovima dostupnih
    podataka). Jednake težine daju novčanom toku ukupno 50% nominalne težine; četiri sastavnice
    nisu četiri neovisna izvora informacije.</p>
    {display.to_html(index=False,float_format=lambda x:f'{x:.3f}')}
    <p>CAGR, annual_vol i monthly_max_drawdown u gornjoj tablici su postoci. Svi rezultati:
    gornjih/donjih 20%, jednake težine, long +100% i short −100%, rebalans svaki mjesec,
    ulaz sljedeće sesije na zatvaranju. Filtri i izvršni model isti su kao u ranijem backtestu.</p>
    <img src="bundle_equity.png" alt="Bundle portfolios and a separate return-basket diagnostic">
    <h2>IC na kontroliranom uzorku</h2>
    {ic[ic.period.eq('full')].to_html(index=False,float_format=lambda x:f'{x:.4f}')}
    <p>Cosine raw računa se na internom centriranom bundle signalu; pojedinačni računovodstveni
    omjeri ostaju izvorni. Razlike sirovog cosine-a nisu dokaz dodane vrijednosti jer komponente
    imaju različite sredine. Spearman i cosine centriranih rangova primjereniji su za ovu usporedbu.</p>
    <h2>Zasebna košarica četiriju prinosa</h2><p>Za doslovnu kombinaciju prethodna četiri faktora
    računa se i prosjek triju ranijih bruto long–short prinosa i javnog RMW-a, po 25% kapitala
    svakom na početku mjeseca, bez dnevnog vraćanja težina. To je dijagnostička serija prinosa,
    a ne indikator za dionice ni provjeren izvršivi portfelj. Ne preuzima se aktualno novija
    Frenchova povijest; koristi se ista prethodno spremljena verzija i isti prozori držanja.</p>
    {basket_stats.to_html(index=False,float_format=lambda x:f'{x:.4f}')}
    <p>Brojevi u tablici košarice su decimalne vrijednosti. Korelacije komponentnih prinosa:</p>
    {correlations.to_html(float_format=lambda x:f'{x:.3f}')}
    <h2>Podaci i provjere</h2><ul>{''.join(f'<li><a href="{name}">{name}</a></li>' for name in [
        'monthly_bundle_scores.parquet','latest_bundle_scores.csv','coverage.csv','summary.csv','monthly_returns.csv',
        'monthly_ic.csv','ic_summary.csv','incremental_comparisons.csv','conditional_returns.csv',
        'four_factor_basket.csv','basket_summary.csv','holdings.parquet','manifest.json','validation.json'])}</ul>
    <p>Kontrole istog univerzuma uključuju svaku komponentu i bundle 3 među istim dionicama kao bundle 4.
    Poseban bundle 3 na širem univerzumu pokazuje posljedicu pokrivenosti. Podrazdoblja prije i od 2017.
    provjeravaju stabilnost; nisu netaknut out-of-sample test. P vrijednosti i BH q su eksplorativni:
    pet usporedbi za svaki metric/scenarij/podrazdoblje, bez korekcije za raniju povijest istraživanja.</p>
    <p>Nepoznat izlaz u osnovnoj verziji znači pretpostavljen prinos nula i zatvaranje; spremljen je
    i nepovoljni scenarij (−100% underlying za long, +100% za short). Troškovi su pretpostavljeni,
    dostupnost shorta nije povijesno potvrđena. Cash kamata, kolateral i financiranje su nula.
    Prilagođene cijene, revizije izvještaja i procijenjeni datumi objave ostaju ograničenja.
    Režimske tablice su uvjetni 12 × mjesečni prosjeci, nisu CAGR režimske strategije.
    Indikator je mjesečni presjek; nije napravljen dnevni panel niti promijenjen upstream paket.</p></html>'''
    (OUT/'report.html').write_text(document,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    initial = json.loads((a.OUT/'manifest.json').read_text())
    assert initial['inputs'] == [a.fingerprint(p) for p in [a.PRICES,a.FUND,a.EXTFILE]]
    data = pd.read_parquet(a.CACHE/'monthly_panel.parquet')
    scores = build_scores(data)
    scores.to_parquet(OUT/'monthly_bundle_scores.parquet',index=False)
    latest = scores[scores.signal_date.eq(scores.signal_date.max()) & scores.bundle4.notna()]
    latest.sort_values('bundle4_score',ascending=False).to_csv(OUT/'latest_bundle_scores.csv',index=False)
    cover = coverage(data,scores)
    cover.to_csv(OUT/'coverage.csv',index=False)
    data = data.join(scores[['bundle4','bundle3_common4','bundle3']])
    common4 = data.bundle4.notna()
    for new,old in zip(['fcf_common4','cfo_common4','gp_common4','ff_common4'],COMPONENTS):
        data[new] = data[old].where(common4)
    diagnostics,ic = diagnostic_rows(data)
    diagnostics.to_csv(OUT/'monthly_ic.csv',index=False)
    ic.to_csv(OUT/'ic_summary.csv',index=False)
    prior = pd.read_csv(ROOT/'results/monthly_long_short/monthly_returns.csv')
    benchmark = prior.drop_duplicates('signal_date').sort_values('signal_date').spy_return.to_numpy()
    monthly,summary,holdings = simulate_all(data,benchmark)
    monthly.to_csv(OUT/'monthly_returns.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)
    holdings.to_parquet(OUT/'holdings.parquet',index=False)
    increments = compare_increments(monthly,diagnostics)
    increments.to_csv(OUT/'incremental_comparisons.csv',index=False)
    conditional_results(monthly).to_csv(OUT/'conditional_returns.csv',index=False)
    basket,basket_stats,correlations = return_basket()
    basket.to_csv(OUT/'four_factor_basket.csv',index=False)
    basket_stats.to_csv(OUT/'basket_summary.csv',index=False)
    correlations.to_csv(OUT/'factor_return_correlations.csv')
    make_plot(monthly,basket)
    report(summary,ic,cover,increments,basket_stats,correlations)
    files = [CONFIG_PATH,a.CACHE/'monthly_panel.parquet',bt.CONFIG,
             ROOT/'scripts/backtest_monthly_long_short.py',ROOT/'scripts/analyze_quality.py',
             ROOT/'scripts/analyze_regimes.py',ROOT/'scripts/compare_public_rmw.py',
             ROOT/'results/monthly_long_short/monthly_returns.csv',
             ROOT/'results/public_comparison/rmw_aligned_windows.csv',
             ROOT/'results/regimes/regime_features.csv']
    manifest = dict(config=CONFIG,code_sha256=sha(__file__),
        inputs=[dict(path=str(p),sha256=sha(p)) for p in files],source_inputs=initial['inputs'],
        output_rows=len(scores),eligible_bundle4_rows=int(scores.bundle4.notna().sum()),
        portfolio_paths=monthly.groupby(['signal','scenario','missing_policy']).ngroups,
        semantics='Stock score uses local OP/book-equity; public RMW is only in separate return basket',
        optimization='none; 25% each fixed in advance, no regime gate')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    a.log(summary[(summary.period=='full')&(summary.missing_policy=='flat_unknown')
        &(summary.scenario=='gross_unconstrained')][['signal','cagr','sharpe_cash0','monthly_max_drawdown']].to_string(index=False))


if __name__=='__main__':
    main()
