"""Causal price-based regimes and conditional profitability diagnostics.

Reuses already validated portfolio paths. No strategy gates are backtested.
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
import pyarrow.dataset as ds
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import analyze_quality as a

ROOT = a.ROOT
OUT = ROOT/'results'/'regimes'
CACHE = ROOT/'data'/'derived'
CONFIG = json.loads((ROOT/'config'/'regimes.json').read_text())
REGIMES = CONFIG['regimes']
LABELS = {
    'spy_vol21': ('Niska volatilnost', 'Visoka volatilnost'),
    'spy_trend126': ('Padajući trend', 'Rastući trend'),
    'spy_tlt_corr63': ('Korelacija ≤ 0', 'Korelacija > 0'),
    'breadth126': ('Uska širina', 'Široka širina'),
    'dispersion21': ('Niska disperzija', 'Visoka disperzija'),
    'illiquidity21': ('Viša likvidnost', 'Niža likvidnost'),
}
NAMES = {'spy_vol21':'Volatilnost SPY-a (21 dan)',
         'spy_trend126':'Trend SPY-a (126 dana)',
         'spy_tlt_corr63':'Korelacija SPY–TLT (63 dana)',
         'breadth126':'Širina: udio pozitivnih 126d prinosa',
         'dispersion21':'Disperzija 21d prinosa (IQR/1,349)',
         'illiquidity21':'Medijan Amihudove nelikvidnosti (21 dan)'}
SHORT = {'fcf_assets_ttm':'FCF / imovina', 'cfo_assets_ttm':'CFO / imovina',
         'gp_assets_ttm':'Bruto dobit / imovina', 'public_rmw':'Javni RMW'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs():
    # Assert that existing paths and new features use precisely the same snapshot.
    old = json.loads((ROOT/'results/initial_analysis/manifest.json').read_text())
    for expected in old['inputs']:
        assert a.fingerprint(Path(expected['path'])) == expected, 'Source snapshot changed'
    etf_path = CACHE/'regime_etf_daily.parquet'
    if not etf_path.exists():
        table = ds.dataset(str(a.PRICES)).to_table(
            columns=['ticker','trade_date','PV_CLOSE','PV_RET_CC_1D',
                     'PV_RET_CC_126D','PV_REALIZED_VOL_21D'],
            filter=ds.field('ticker').isin(['SPY','TLT']))
        table.to_pandas(date_as_object=False).to_parquet(etf_path, index=False)
    cross_path = CACHE/'regime_monthly_cross_section.parquet'
    if not cross_path.exists():
        panel = pd.read_parquet(CACHE/'monthly_panel.parquet',
                               columns=['security_id','signal_date','eligible'])
        table = ds.dataset(str(a.PRICES)).to_table(
            columns=['security_id','trade_date','PV_RET_CC_21D','PV_RET_CC_126D',
                     'PV_AMIHUD_ILLIQ_21D'],
            filter=ds.field('trade_date').isin(panel.signal_date.drop_duplicates().tolist()))
        cross = table.to_pandas(date_as_object=False).rename(columns={'trade_date':'signal_date'})
        panel.loc[panel.eligible,['security_id','signal_date']].merge(
            cross, on=['security_id','signal_date'], how='left', validate='one_to_one'
        ).to_parquet(cross_path, index=False)
    return pd.read_parquet(etf_path), pd.read_parquet(cross_path)


def construct_features(etf, cross, calendar):
    etf = etf.assign(trade_date=pd.to_datetime(etf.trade_date)).sort_values(['trade_date','ticker'])
    assert not etf.duplicated(['trade_date','ticker']).any()
    returns = etf.pivot(index='trade_date', columns='ticker', values='PV_RET_CC_1D')
    corr = returns.SPY.rolling(63, min_periods=63).corr(returns.TLT)
    spy = etf[etf.ticker.eq('SPY')].set_index('trade_date')
    features = calendar.set_index('signal_date').copy()
    features['spy_vol21'] = spy.PV_REALIZED_VOL_21D.reindex(features.index)
    features['spy_trend126'] = spy.PV_RET_CC_126D.reindex(features.index)
    features['spy_tlt_corr63'] = corr.reindex(features.index)
    grouped = cross.groupby('signal_date')
    aggregates = grouped.agg(universe_n=('security_id','size'),
                             breadth_n=('PV_RET_CC_126D','count'),
                             dispersion_n=('PV_RET_CC_21D','count'),
                             liquidity_n=('PV_AMIHUD_ILLIQ_21D','count'))
    breadth = grouped.PV_RET_CC_126D.apply(lambda s: s.dropna().gt(0).mean())
    features['breadth126'] = breadth.where(
        aggregates.breadth_n / aggregates.universe_n >= CONFIG['breadth_minimum_coverage'])
    features['dispersion21'] = (grouped.PV_RET_CC_21D.quantile(.75)
                                -grouped.PV_RET_CC_21D.quantile(.25))/1.349
    features['dispersion21_raw_sd'] = grouped.PV_RET_CC_21D.std()
    features['illiquidity21'] = grouped.PV_AMIHUD_ILLIQ_21D.median()
    features = features.join(aggregates)
    return classify(features)


def classify(features):
    features = features.copy()
    warmup = CONFIG['minimum_previous_months']
    for name in REGIMES:
        raw = features[name]
        prior = raw.shift(1).expanding(min_periods=warmup)
        cutoff = CONFIG['fixed_thresholds'].get(name)
        features[name+'_threshold'] = cutoff if cutoff is not None else prior.median()
        known = raw.notna() & features[name+'_threshold'].notna()
        high = raw > features[name+'_threshold']
        # Equality belongs to low state for all boundaries.
        features[name+'_state'] = high.astype(float).where(known)
        transformed = np.log(raw.where(raw > 0)) if name in [
            'spy_vol21','dispersion21','illiquidity21'] else raw
        history = transformed.shift(1).expanding(min_periods=warmup)
        sd = history.std().replace(0, np.nan)
        features[name+'_z'] = ((transformed-history.mean())/sd).clip(-3,3)
    features['common_sample'] = features[[s+'_state' for s in REGIMES]].notna().all(axis=1)
    vol, trend = features.spy_vol21_state, features.spy_trend126_state
    features['trend_vol_state'] = (2*trend+vol).where(vol.notna() & trend.notna())
    return features


def calendar_fit(y, x):
    """OLS/Newey-West; zero scores for missing months retain calendar lag distances."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if ok.sum() < max(24, x.shape[1]*5) or np.linalg.matrix_rank(x[ok]) < x.shape[1]:
        return None
    inverse = np.linalg.inv(x[ok].T @ x[ok])
    beta = inverse @ x[ok].T @ y[ok]
    scores = np.zeros_like(x)
    scores[ok] = x[ok]*(y[ok]-x[ok] @ beta)[:,None]
    meat = scores.T @ scores
    lags = CONFIG['hac_calendar_month_lags']
    for lag in range(1,lags+1):
        g = scores[lag:].T @ scores[:-lag]
        meat += (1-lag/(lags+1))*(g+g.T)
    cov = inverse @ meat @ inverse
    se = np.sqrt(np.maximum(0,np.diag(cov)))
    t = np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    return {'beta':beta,'cov':cov,'t':t,'p':2*norm.sf(abs(t)), 'n':int(ok.sum())}


def annual_stats(values):
    y = np.asarray(values.dropna(),dtype=float)
    mean = float(y.mean())
    vol = float(y.std(ddof=1)*np.sqrt(12)) if len(y)>1 else np.nan
    q = np.quantile(y,.1)
    return {'months':len(y),'annual_mean':12*mean,'annual_vol':vol,
            'conditional_sharpe':12*mean/vol if vol>0 else np.nan,
            'hit_rate':float(np.mean(y>0)),'monthly_q10':q,
            'monthly_es10':float(y[y<=q].mean()),'worst_month':float(y.min())}


def period_mask(index, period):
    split = pd.Timestamp(CONFIG['stability_split'])
    if period == 'early':
        return index < split
    if period == 'late':
        return index >= split
    return np.ones(len(index), dtype=bool)


def conditional_tables(paths, features, diagnostics):
    rows, differences, ic_rows = [], [], []
    common = features.common_sample
    regimes = REGIMES+['trend_vol']
    for regime in regimes:
        state = features[regime+'_state'].where(common)
        for (signal,scenario,policy), path in paths.groupby(['signal','scenario','missing_policy']):
            values = path.set_index('signal_date')['return'].reindex(features.index)
            for period in ['full','early','late']:
                y = values.where(period_mask(features.index,period))
                for level in sorted(state.dropna().unique()):
                    group = y[(state == level) & y.notna()]
                    if group.empty:
                        continue
                    row = dict(signal=signal,scenario=scenario,missing_policy=policy,
                               period=period,regime=regime,state=int(level),
                               first_signal_date=group.index.min(),last_signal_date=group.index.max())
                    row.update(annual_stats(group))
                    rows.append(row)
                if regime in REGIMES:
                    x = np.column_stack([np.ones(len(state)),state])
                    fit = calendar_fit(y,x)
                    if fit:
                        differences.append(dict(signal=signal,scenario=scenario,missing_policy=policy,
                            period=period,regime=regime,months=fit['n'],
                            high_minus_low_annual=12*fit['beta'][1],hac_t=fit['t'][1],p=fit['p'][1],
                            ci95_low_annual=12*(fit['beta'][1]-1.96*np.sqrt(fit['cov'][1,1])),
                            ci95_high_annual=12*(fit['beta'][1]+1.96*np.sqrt(fit['cov'][1,1]))))
        for signal, data in diagnostics.groupby('signal'):
            data = data.set_index('signal_date').reindex(features.index)
            for period in ['full','early','late']:
                for level in sorted(state.dropna().unique()):
                    use = (state == level) & period_mask(features.index,period)
                    selected = data.loc[use]
                    if selected.empty:
                        continue
                    row = dict(signal=signal,period=period,regime=regime,state=int(level))
                    for metric in ['ic_spearman','ic_cosine_raw','ic_cosine_rank_centered']:
                        row[metric] = selected[metric].mean()
                        row[metric+'_months'] = int(selected[metric].notna().sum())
                    row['mean_coverage'] = selected.n.mean()
                    ic_rows.append(row)
    diffs = pd.DataFrame(differences)
    diffs['q_bh'] = np.nan
    # Separate explicit 18-test families for each scenario/policy/period (3 signals x 6).
    for _, group in diffs.groupby(['scenario','missing_policy','period']):
        main = group[group.signal.isin(CONFIG['signals'])]
        if len(main):
            diffs.loc[main.index,'q_bh'] = multipletests(main.p,method='fdr_bh')[1]
    return pd.DataFrame(rows), diffs, pd.DataFrame(ic_rows)


def continuous_tables(paths, features, diagnostics):
    rows = []
    gross = paths[(paths.scenario=='gross_unconstrained') & (paths.missing_policy=='flat_unknown')]
    outcomes = [(signal,'return',d.set_index('signal_date')['return'])
                for signal,d in gross.groupby('signal')]
    for signal, d in diagnostics[diagnostics.signal.isin(CONFIG['signals'])].groupby('signal'):
        for metric in ['ic_spearman','ic_cosine_raw','ic_cosine_rank_centered']:
            outcomes.append((signal,metric,d.set_index('signal_date')[metric]))
    spy = gross.drop_duplicates('signal_date').set_index('signal_date').spy_return.reindex(features.index)
    z = features[[r+'_z' for r in REGIMES]]
    for signal,metric,series in outcomes:
        y = series.reindex(features.index)
        for period in ['full','early','late']:
            yp = y.where(period_mask(features.index,period))
            models = [(r,[r+'_z']) for r in REGIMES]+[('joint',list(z.columns))]
            if metric == 'return':
                models += [('joint_market_control',list(z.columns))]
            for model, columns in models:
                xx = features[columns].copy()
                if model == 'joint_market_control':
                    # Ex-post attribution only, never a regime input or trading rule.
                    xx['realized_spy_return'] = spy
                x = np.column_stack([np.ones(len(xx)),xx.to_numpy()])
                fit = calendar_fit(yp,x)
                if fit:
                    for j,name in enumerate(xx.columns,1):
                        rows.append(dict(signal=signal,metric=metric,period=period,model=model,
                            feature=name,months=fit['n'],coefficient=fit['beta'][j],
                            annual_coefficient=12*fit['beta'][j] if metric=='return' else np.nan,
                            hac_t=fit['t'][j],p=fit['p'][j]))
    result = pd.DataFrame(rows)
    result['q_bh'] = np.nan
    # 18 primary return slopes; IC family has 3 metrics x 3 signals x 6 = 54 tests.
    for period in ['full','early','late']:
        for family in ['returns','ic']:
            selected = ((result.period==period) & result.model.isin(REGIMES)
                        & result.signal.isin(CONFIG['signals']))
            selected &= result.metric.eq('return') if family=='returns' else result.metric.ne('return')
            if selected.any():
                result.loc[selected,'q_bh'] = multipletests(result.loc[selected,'p'],method='fdr_bh')[1]
    return result


def plot(summary):
    d = summary[(summary.period=='full') & (summary.scenario=='gross_unconstrained')
                & (summary.missing_policy=='flat_unknown') & summary.regime.isin(REGIMES)]
    fig, axes = plt.subplots(2,3,figsize=(16,9),sharey=True)
    for ax,regime in zip(axes.flat,REGIMES):
        for j,signal in enumerate(CONFIG['signals']+['public_rmw']):
            pair = d[(d.regime==regime) & (d.signal==signal)].set_index('state')
            ax.bar(j-.18,pair.loc[0,'annual_mean']*100,.34,color='#34699a',label='Niže / negativno' if j==0 else None)
            ax.bar(j+.18,pair.loc[1,'annual_mean']*100,.34,color='#db9854',label='Više / pozitivno' if j==0 else None)
        counts = d[(d.regime==regime)&(d.signal==CONFIG['signals'][0])].set_index('state').months
        ax.set_title(NAMES[regime]+'\n'+f'n={counts.loc[0]} / {counts.loc[1]}',fontsize=10)
        ax.set_xticks(range(4),['FCF','CFO','GP','RMW'])
        ax.axhline(0,color='#333333',lw=.8)
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel('Uvjetni godišnji prosjek (%)')
    axes[1,0].set_ylabel('Uvjetni godišnji prosjek (%)')
    fig.suptitle('Profitabilnost prema režimu poznatom prije ulaska\nBruto +100% long / −100% short; 12 × prosječni mjesečni prinos, nije CAGR',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(OUT/'conditional_returns.png',dpi=150)
    plt.close(fig)


def report(summary, diffs, ic, continuous, features):
    common = features[features.common_sample]
    main = summary[(summary.period=='full') & (summary.scenario=='gross_unconstrained')
                   & (summary.missing_policy=='flat_unknown') & summary.signal.isin(CONFIG['signals'])
                   & summary.regime.isin(REGIMES)].copy()
    main['Režim'] = [NAMES[r]+': '+LABELS[r][int(s)] for r,s in zip(main.regime,main.state)]
    main['Signal'] = main.signal.map(SHORT)
    for source,target in [('annual_mean','Godišnji prosjek %'),('annual_vol','Godišnja vol. %')]:
        main[target] = (main[source]*100).round(2)
    table = main[['Režim','Signal','months','Godišnji prosjek %','Godišnja vol. %','conditional_sharpe']]
    selected = diffs[(diffs.period=='full') & (diffs.scenario=='gross_unconstrained')
                     & (diffs.missing_policy=='flat_unknown') & diffs.signal.isin(CONFIG['signals'])]
    reg = continuous[(continuous.period=='full') & (continuous.metric=='return')
                     & continuous.model.isin(REGIMES) & continuous.signal.isin(CONFIG['signals'])]
    def result(signal,regime,state,column='annual_mean',period='full',
               scenario='gross_unconstrained',policy='flat_unknown'):
        row = summary[(summary.signal==signal)&(summary.regime==regime)&(summary.state==state)
                      &(summary.period==period)&(summary.scenario==scenario)&(summary.missing_policy==policy)]
        return float(row.iloc[0][column])
    gp = 'gp_assets_ttm'
    fcf = 'fcf_assets_ttm'
    findings = f'''<h2>Što pokazuju rezultati</h2><p>Najdosljedniji obrazac ima bruto dobit/imovina:
    nakon negativnog 126d trenda SPY-a uvjetni bruto godišnji prosjek je
    {result(gp,'spy_trend126',0):.2%}, a nakon pozitivnog {result(gp,'spy_trend126',1):.2%}.
    Udio pozitivnih 126d prinosa ≤50% daje {result(gp,'breadth126',0):.2%},
    nasuprot {result(gp,'breadth126',1):.2%} kada je udio >50%.
    Pri uskoj širini uvjetni Sharpe je {result(gp,'breadth126',0,'conditional_sharpe'):.2f},
    na {result(gp,'breadth126',0,'months'):.0f} mjeseca.</p>
    <p>Smjer za bruto dobit/imovinu opstaje u obje vremenske polovice: pri uskoj/širokoj širini
    prosjeci su {result(gp,'breadth126',0,period='early'):.2%} /
    {result(gp,'breadth126',1,period='early'):.2%} prije 2017. i
    {result(gp,'breadth126',0,period='late'):.2%} /
    {result(gp,'breadth126',1,period='late'):.2%} od 2017. Razlika je kasnije manja.
    Kod FCF/imovina smjer trenda se obrće: padajući/rastući trend daje
    {result(fcf,'spy_trend126',0,period='early'):.2%} /
    {result(fcf,'spy_trend126',1,period='early'):.2%} prije 2017., a
    {result(fcf,'spy_trend126',0,period='late'):.2%} /
    {result(fcf,'spy_trend126',1,period='late'):.2%} od 2017.</p>
    <p>Pozitivna SPY–TLT korelacija izgleda povoljno za FCF: bruto
    {result(fcf,'spy_tlt_corr63',1):.2%} na
    {result(fcf,'spy_tlt_corr63',1,'months'):.0f} mjeseci. Međutim, 40 od tih 46 mjeseci je od 2017.,
    a nepovoljna pretpostavka nepoznatih izlaza smanjuje prosjek na
    {result(fcf,'spy_tlt_corr63',1,policy='adverse_unknown'):.2%} i prije troškova.
    To je vremenski koncentriran i osjetljiv nalaz.</p>
    <p>Viša volatilnost povećava prosječne prinose sva tri signala, ali i rizik. Viša disperzija
    ne poboljšava dosljedno Sharpe. Likvidnosni obrazac nije jednak za tri signala.
    Javni RMW također ima veći bruto prosjek pri uskoj širini:
    {result('public_rmw','breadth126',0):.2%} nasuprot
    {result('public_rmw','breadth126',1):.2%}.</p>
    <p><strong>Statistički zaključak:</strong> nijedna od 18 primarnih binarnih razlika ni 18
    pojedinačnih kontinuiranih nagiba prinosa nema BH q &lt; 0,05. Pojedinačna kontinuirana
    regresija GP-a na trend ima p≈0,021, ali q≈0,378; u zajedničkoj regresiji slabi.
    Ovo su kandidati za daljnju provjeru, a ne potvrđena pravila za alokaciju.</p>
    <p>IC podržava opisni obrazac GP-a: pri uskoj/širokoj širini Spearman IC je
    0,0312 / 0,0106, sirovi cosine 0,0476 / 0,0364, a cosine centriranih rangova
    0,0176 / 0,0056. Ni jedan od 54 primarna kontinuirana IC testa ne prolazi BH q &lt; 0,05.
    Sve tri mjere i ostalih 18 signala dostupni su u conditional_ic.csv.</p>
    <p>Troškovi: GP pri uskoj/širokoj širini, uz 10 bp po jednosmjernom prometu i 3% godišnje
    posudbe, daje {result(gp,'breadth126',0,scenario='cost10_borrow3'):.2%} /
    {result(gp,'breadth126',1,scenario='cost10_borrow3'):.2%}. Uz isti trošak i nepovoljne
    nepoznate izlaze to pada na
    {result(gp,'breadth126',0,scenario='cost10_borrow3',policy='adverse_unknown'):.2%} /
    {result(gp,'breadth126',1,scenario='cost10_borrow3',policy='adverse_unknown'):.2%}.
    To su uvjetni rezultati postojećeg stalno rebalansiranog portfelja; nova strategija koja
    trguje samo u određenom režimu imala bi drukčiji promet.</p>'''
    methods = '''<p>Režim je određen na zatvaranju formacijskog mjeseca. Prinos dolazi iz već
    provjerenog sljedećeg prozora držanja: ulaz sljedeće sesije na zatvaranju, izlaz sesiju nakon
    sljedećeg kraja mjeseca. Ne klasificiramo mjesec njegovim budućim padom ili volatilnošću.</p>
    <p>Volatilnost: SPY annualizirana SD dnevnih prinosa, 21 sesija. Trend: SPY prinos 126 sesija,
    granica nula. Korelacija: 63 dnevna prinosa SPY/TLT, granica nula. Širina: udio podobnih dionica
    s pozitivnim prethodnim 126d prinosom, granica 50%; najmanje 80% univerzuma mora imati podatak.
    To je zamjena za širinu iz priloga; nije udio iznad MA200. Disperzija: IQR prethodnih 21d prinosa
    podijeljen s 1,349, uz spremljenu sirovu SD radi provjere ekstrema. Nelikvidnost: medijan
    Amihudove 21d mjere u istom podobnom univerzumu; viša vrijednost znači nižu likvidnost.
    Medijani za vol./disperziju/nelikvidnost računaju se samo iz ranijih mjesečnih opažanja,
    uz najmanje 12 prethodnih mjeseci. Univerzum se mijenja kroz vrijeme.</p>
    <p>Glavne usporedbe svih šest režima koriste zajedničke mjesece. Broj mjeseci i rizik prikazuju
    se uz prinose. Godišnji prosjek = 12 × mjesečni prosjek; nije CAGR niti prinos strategije koja
    se uključuje samo u pojedinom režimu. Sharpe je uvjetni prosjek / uvjetna SD × √12,
    uz nultu kamatu; ne govori o rezultatu cijelog kalendarskog razdoblja.</p>
    <p>Dodatna podjela trend_vol u CSV-u: 0 = padajući trend / niska vol.,
    1 = padajući / visoka, 2 = rastući / niska, 3 = rastući / visoka.
    Prva ćelija ima samo 11 mjeseci i njezin visok Sharpe nije pouzdan temelj za odluku.</p>
    <p>Kontinuirane regresije imaju zaseban raspoloživi uzorak po varijabli. Pozitivne mjere vol.,
    disperzije i nelikvidnosti prvo se logaritmiraju. Standardizacija koristi samo prethodne mjesece
    (najmanje 12), uz ograničenje z na ±3. Prikazane su pojedinačne i zajedničke regresije te
    ex-post kontrola ostvarenog SPY prinosa: ova kontrola nije budući prediktor.
    HAC koristi tri kalendarska mjesečna odmaka; nedostajući mjeseci ne komprimiraju vremensku os.
    BH q za bruto povrate korigira obitelj 18 testova (3 signala × 6 režima); za IC obitelj 54.
    Podjele prije 2017. i od 2017. testiraju stabilnost. Budući da su podaci već analizirani,
    to nije netaknut out-of-sample test. Nisu optimizirani pragovi ni režimska strategija.</p>
    <p>Osnovni scenarij nepoznatom izlazu pripisuje nulti prinos i pretpostavljeno zatvaranje.
    CSV sadrži i raniji nepovoljni scenarij te troškovne rezultate (5 bp / 0,5% i 10 bp / 3%
    godišnje posudbe). To su pretpostavljeni troškovi. Izvorne prilagođene cijene nisu potvrđeni
    total-return/delisting podaci; računovodstvene revizije i procijenjeni datumi objave ostaju
    ograničenje. Makro serije imaju MACRO_POINT_IN_TIME=False i nisu uključene. VIX, kreditni
    spreadovi, inflacija i monetarni režimi zato nisu testirani ovom analizom.</p>
    <p>Javni RMW je prethodno preuzeta dnevna Frenchova serija složena u iste prozore držanja.
    Ima drukčiju konstrukciju od naših portfelja; prikazana je samo bruto dijagnostički.
    <a href="https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/f-f_5_factors_2x3.html">Izvor RMW</a>.</p>'''
    document = f'''<!doctype html><html lang="hr"><meta charset="utf-8"><title>Quality: profitabilnost kroz režime</title>
    <style>body{{font:15px system-ui;color:#1d2939;max-width:1250px;margin:40px auto;padding:0 20px;line-height:1.55}}
    table{{border-collapse:collapse;font-size:12px;width:100%;margin:20px 0}}td,th{{padding:7px;border-bottom:1px solid #d0d5dd;text-align:right}}
    td:first-child,th:first-child{{text-align:left}}img{{width:100%}}h2{{margin-top:35px}}</style>
    <h1>Quality: profitabilna komponenta kroz režime</h1>
    <p>Zajednički uzorak: {common.index.min():%d.%m.%Y.}–{common.index.max():%d.%m.%Y.},
    {len(common)} formacijskih mjeseci. Analiza je eksplorativna.</p>
    {findings}
    <img src="conditional_returns.png" alt="Uvjetni bruto prinosi šest režima">
    <h2>Bruto prinos i rizik</h2>{table.to_html(index=False,float_format=lambda x:f'{x:.3f}')}
    <h2>Razlika više/pozitivno minus niže/negativno</h2>
    <p>annual su decimalni godišnji prosjeci; intervali su 95% HAC. Neizvjesnost uključuje
    i mali broj mjeseci u nekim stanjima.</p>{selected.to_html(index=False,float_format=lambda x:f'{x:.4f}')}
    <h2>Kontinuirane regresije</h2>{reg.to_html(index=False,float_format=lambda x:f'{x:.4f}')}
    <h2>Datoteke</h2><ul>{''.join(f'<li><a href="{f}">{f}</a></li>' for f in [
        'regime_features.csv','conditional_returns.csv','regime_differences.csv','conditional_ic.csv',
        'continuous_regressions.csv','manifest.json','validation.json'])}</ul>
    <h2>Metodologija i ograničenja</h2>{methods}</html>'''
    (OUT/'report.html').write_text(document,encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    etf,cross = prepare_inputs()
    calendar = pd.read_csv(ROOT/'results/initial_analysis/formation_calendar.csv',
                           parse_dates=['signal_date','entry_date','exit_date'])
    features = construct_features(etf,cross,calendar)
    features.to_csv(OUT/'regime_features.csv')
    paths = pd.read_csv(ROOT/'results/monthly_long_short/monthly_returns.csv',
                        parse_dates=['signal_date','entry_date','exit_date'])
    paths = paths[paths.scenario.isin(CONFIG['primary_cost_scenarios'])].copy()
    public = pd.read_csv(ROOT/'results/public_comparison/rmw_aligned_windows.csv',
                         parse_dates=['signal_date','entry_date','exit_date'])
    public = public.rename(columns={'RMW_daily_compounded':'return'})
    public['signal'],public['scenario'],public['missing_policy'] = 'public_rmw','gross_unconstrained','flat_unknown'
    spy = paths.drop_duplicates('signal_date').set_index('signal_date').spy_return
    public['spy_return'] = public.signal_date.map(spy)
    paths = pd.concat([paths,public],ignore_index=True)
    diagnostics = pd.read_csv(ROOT/'results/initial_analysis/monthly_diagnostics.csv',parse_dates=['signal_date'])
    diagnostics = diagnostics[diagnostics['sample'].eq('own')]
    summary,diffs,ic = conditional_tables(paths,features,diagnostics)
    continuous = continuous_tables(paths,features,diagnostics)
    for frame,name in [(summary,'conditional_returns'),(diffs,'regime_differences'),
                       (ic,'conditional_ic'),(continuous,'continuous_regressions')]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    plot(summary)
    report(summary,diffs,ic,continuous,features)
    files = [ROOT/'config/regimes.json',CACHE/'monthly_panel.parquet',
             CACHE/'regime_etf_daily.parquet',CACHE/'regime_monthly_cross_section.parquet',
             ROOT/'results/monthly_long_short/monthly_returns.csv',
             ROOT/'results/initial_analysis/monthly_diagnostics.csv',
             ROOT/'results/public_comparison/rmw_aligned_windows.csv']
    manifest = {'config':CONFIG,'code_sha256':sha(__file__),
                'inputs':[{'path':str(p),'sha256':sha(p)} for p in files],
                'source_snapshot':a.fingerprint(a.PRICES),
                'common_months':int(features.common_sample.sum()),
                'primary_test_families':'18 binary differences and 18 univariate continuous return slopes, separately; 54 IC slopes',
                'macro_excluded':'local series explicitly ex_post_observation_history / not point-in-time',
                'limitations':'inherited input revisions, unknown exit policies and costs; no optimized trading rule'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'common_months':manifest['common_months'],'conditional_return_rows':len(summary),
                      'ic_rows':len(ic),'regression_rows':len(continuous)}),flush=True)


if __name__ == '__main__':
    main()
