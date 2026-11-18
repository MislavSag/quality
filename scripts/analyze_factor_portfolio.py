"""User-selected equal-capital portfolio of three local factors and public RMW."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_quality as a
import analyze_regimes as regimes

ROOT=a.ROOT
OUT=ROOT/'results/factor_portfolio'
CONFIG_PATH=ROOT/'config/factor_portfolio.json'
CONFIG=json.loads(CONFIG_PATH.read_text())
WEIGHTS=pd.Series(CONFIG['weights'])
NAMES={'fcf_assets_ttm':'FCF/imovina','cfo_assets_ttm':'CFO/imovina',
       'gp_assets_ttm':'Bruto dobit/imovina','public_rmw':'Javni RMW',
       'portfolio4':'Portfelj četiri faktora','portfolio3':'Portfelj prva tri faktora'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values,entry,exit):
    y=values.to_numpy(dtype=float)
    equity=np.r_[1,np.cumprod(1+y)]
    years=(exit.iloc[-1]-entry.iloc[0]).days/365.25
    return dict(months=len(y),cagr=equity[-1]**(1/years)-1,
        annual_mean=y.mean()*12,annual_vol=y.std(ddof=1)*np.sqrt(12),
        sharpe_cash0=y.mean()/y.std(ddof=1)*np.sqrt(12),
        monthly_max_drawdown=(equity/np.maximum.accumulate(equity)-1).min())


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    assert np.isclose(WEIGHTS.sum(),1) and WEIGHTS.eq(.25).all()
    local_path=ROOT/'results/monthly_long_short/monthly_returns.csv'
    public_path=ROOT/'results/public_comparison/rmw_aligned_windows.csv'
    regime_path=ROOT/'results/regimes/regime_features.csv'
    local=pd.read_csv(local_path,parse_dates=['signal_date','entry_date','exit_date'])
    local=local[(local.scenario=='gross_unconstrained') & (local.missing_policy=='flat_unknown')]
    public=pd.read_csv(public_path,parse_dates=['signal_date','entry_date','exit_date'])
    calendar=local.drop_duplicates('signal_date').set_index('signal_date')[['entry_date','exit_date']].sort_index()
    pd.testing.assert_frame_equal(calendar,public.set_index('signal_date')[calendar.columns],check_freq=False)
    values=local.pivot(index='signal_date',columns='signal',values='return')
    values['public_rmw']=public.set_index('signal_date').RMW_daily_compounded
    values=values.reindex(columns=WEIGHTS.index)
    assert len(values)==235 and values.notna().all().all()
    values['portfolio4']=values[WEIGHTS.index]@WEIGHTS
    values['portfolio3']=values[WEIGHTS.index[:3]].mean(axis=1)
    monthly=calendar.join(values)
    monthly['nav_index']=(1+monthly.portfolio4).cumprod()
    curve=np.r_[1,monthly.nav_index]
    monthly['drawdown']=(curve/np.maximum.accumulate(curve)-1)[1:]
    monthly.to_csv(OUT/'portfolio_monthly.csv')
    rows=[]
    for period in ['full','early','late']:
        part=monthly[regimes.period_mask(monthly.index,period)]
        for signal in values:
            rows.append(dict(signal=signal,period=period,**stats(part[signal],part.entry_date,part.exit_date)))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv',index=False)
    corr=values[WEIGHTS.index].corr()
    corr.to_csv(OUT/'component_correlations.csv')
    covariance=values[WEIGHTS.index].cov()*12
    variance=float(WEIGHTS@covariance@WEIGHTS)
    contributions=WEIGHTS*(covariance@WEIGHTS)/variance
    risk=pd.DataFrame(dict(capital_weight=WEIGHTS,variance_contribution=contributions))
    risk.to_csv(OUT/'risk_contributions.csv')
    # Actual monthly end weights before rebalancing; capital weights, not stock holdings.
    end_weights=values[WEIGHTS.index].add(1).mul(WEIGHTS).div(1+values.portfolio4,axis=0)
    end_weights.to_csv(OUT/'end_of_month_sleeve_weights.csv')
    features=pd.read_csv(regime_path,parse_dates=['signal_date']).set_index('signal_date')
    conditional=[]
    for regime in regimes.REGIMES:
        state=features[regime+'_state'].where(features.common_sample)
        for level in [0,1]:
            y=monthly.portfolio4.reindex(features.index)[state.eq(level)]
            conditional.append(dict(regime=regime,state=level,**regimes.annual_stats(y)))
    conditional=pd.DataFrame(conditional)
    conditional.to_csv(OUT/'conditional_returns.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for name in ['portfolio4','portfolio3','public_rmw']:
        equity=(1+monthly[name]).cumprod()
        axes[0].plot(monthly.exit_date,equity,label=NAMES[name],lw=2 if name=='portfolio4' else 1.2)
    axes[0].set_title('Bruto portfelj faktora: jednake mjesečne kapitalne težine')
    axes[0].set_ylabel('NAV / početni NAV')
    axes[0].legend()
    axes[1].plot(monthly.exit_date,monthly.drawdown*100,color='#1c5d99')
    axes[1].set_title('Portfelj četiri faktora: drawdown na mjesečnim krajevima')
    axes[1].set_ylabel('Drawdown (%)')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUT/'portfolio.png',dpi=150)
    plt.close(fig)
    table=summary[summary.period=='full'].copy()
    table['signal']=table.signal.map(NAMES)
    for name in ['cagr','annual_mean','annual_vol','monthly_max_drawdown']: table[name]*=100
    counts=conditional[['regime','state','months','annual_mean','annual_vol','conditional_sharpe']].copy()
    counts['regime']=[regimes.NAMES[x]+': '+regimes.LABELS[x][int(y)] for x,y in zip(counts.regime,counts.state)]
    counts[['annual_mean','annual_vol']]*=100
    document=f'''<!doctype html><html lang="hr"><meta charset="utf-8"><title>Quality: portfelj četiriju faktora profitabilnosti</title>
    <style>body{{font:15px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.55;color:#203044}}
    table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}
    td:first-child,th:first-child{{text-align:left}}img{{width:100%}}</style>
    <h1>Quality: portfelj četiriju faktora profitabilnosti</h1>
    <p>Odabrana specifikacija: po 25% kapitala u FCF/imovina, CFO/imovina, bruto dobit/imovina i
    javni RMW. Težine se vraćaju na 25% na početku svakog mjesečnog prozora, a unutar mjeseca driftaju.
    Nema optimizacije težina, izjednačavanja volatilnosti ni režimskog uključivanja.</p>
    <p>R_portfelj,t = 0,25 × (R_FCF,t + R_CFO,t + R_GP,t + R_RMW,t).
    Sva četiri prinosa imaju identične ulazne i izlazne datume. Prvo držanje počinje
    {monthly.entry_date.min():%d.%m.%Y.}, zadnje završava {monthly.exit_date.max():%d.%m.%Y.}: ukupno {len(monthly)} prozora.</p>
    <p>Tri lokalna faktora koriste prethodne portfelje +100% long / −100% short. Javni RMW je
    spremljena dnevna Frenchova serija složena u iste prozore, bez zamjene našim OP/kapital signalom.
    <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html">Izvor RMW</a>.</p>
    <h2>Bruto rezultati</h2><p>CAGR, prosjek, volatilnost i drawdown su postoci; Sharpe koristi nultu cash stopu.</p>
    {table.to_html(index=False,float_format=lambda x:f'{x:.3f}')}
    <img src="portfolio.png" alt="Bruto portfelj četiri faktora i mjesečni drawdown">
    <h2>Diversifikacija</h2>{corr.rename(index=NAMES,columns=NAMES).to_html(float_format=lambda x:f'{x:.3f}')}
    <p>FCF i CFO imaju korelaciju {corr.loc['fcf_assets_ttm','cfo_assets_ttm']:.2f}, pa kombinacija
    sadrži znatno preklapanje. Jednake kapitalne težine ne znače jednake doprinose riziku.
    Doprinosi varijanci u sljedećoj tablici opisne su procjene cijelog uzorka; nisu korišteni za težine.</p>
    {risk.rename(index=NAMES).to_html(float_format=lambda x:f'{x:.3f}')}
    <h2>Režimi</h2><p>Režimi su poznati prije ulaska. Ovdje je annual_mean = 12 × uvjetni mjesečni
    prosjek, ne CAGR strategije koja se uključuje samo u tom režimu. Opisna analiza, bez optimiziranog gatea.</p>
    {counts.to_html(index=False,float_format=lambda x:f'{x:.3f}')}
    <h2>Datoteke i ograničenja</h2><ul>{''.join(f'<li><a href="{f}">{f}</a></li>' for f in [
        'portfolio_monthly.csv','summary.csv','conditional_returns.csv','component_correlations.csv',
        'risk_contributions.csv','end_of_month_sleeve_weights.csv','manifest.json','validation.json'])}</ul>
    <p>Ovo je portfelj istraživačkih serija prinosa, a ne provjeren izvršivi portfelj. Rezultati su
    bruto; troškovi javnog RMW-a i ukupne implementacije nisu procijenjeni. Lokalni nepoznati izlazi
    koriste raniju pretpostavku nultog prinosa i zatvaranja. Cash kamata, kolateral i financiranje
    su nula. Nema konsolidacije preklopljenih pozicija niti procjene stvarnih short količina.
    Revidirani izvještaji, procijenjeni datumi objave i izvorni cjenovni proxy ostaju ograničenja.
    Vremenska podjela prije/od 2017. je provjera stabilnosti, ne netaknut out-of-sample test.</p></html>'''
    (OUT/'report_static.html').write_text(document,encoding='utf-8')
    # Reconcile against the earlier literal four-return basket, computed independently there.
    earlier=pd.read_csv(ROOT/'results/profitability_bundle/four_factor_basket.csv')
    np.testing.assert_allclose(monthly.portfolio4,earlier.basket4_public_rmw,atol=1e-12)
    np.testing.assert_allclose(monthly.portfolio4,values[WEIGHTS.index].sum(axis=1)/4,atol=1e-12)
    np.testing.assert_allclose(end_weights.sum(axis=1),1,atol=1e-12)
    np.testing.assert_allclose(contributions.sum(),1,atol=1e-12)
    np.testing.assert_allclose(monthly.portfolio4.std(ddof=1)*np.sqrt(12),np.sqrt(variance),atol=1e-12)
    assert (monthly.index<monthly.entry_date).all() and (monthly.entry_date<monthly.exit_date).all()
    assert not monthly.index.duplicated().any()
    assert len(monthly)==235
    files=[CONFIG_PATH,local_path,public_path,regime_path]
    (OUT/'manifest.json').write_text(json.dumps(dict(config=CONFIG,code_sha256=sha(__file__),
        inputs=[dict(path=str(p),sha256=sha(p)) for p in files],
        scope='User-selected four-factor portfolio; stock-ranking bundle is separate'),indent=2),encoding='utf-8')
    validation=dict(status='passed',monthly_windows=len(monthly),
        checks=['exact four-way weighted arithmetic','same holding windows','earlier basket reconciliation',
                'drifted sleeve weights sum to one','covariance portfolio variance','variance contributions sum to one'],
        limitations='Calculation checks do not establish executable net returns')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    from build_interactive_report import build
    build()
    print(summary[(summary.signal=='portfolio4')].to_string(index=False),flush=True)


if __name__=='__main__': main()
