"""Build one offline HTML report; portfolio and regime calculations stay unchanged."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from compare_public_rmw import read_factor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/factor_portfolio'
ASSETS=Path(__file__).resolve().parent/'report_assets'

SERIES=[
    dict(id='portfolio4',label='Portfelj četiri faktora',short='Portfelj 4',color='#365eea'),
    dict(id='portfolio3',label='Portfelj prva tri faktora',short='Portfelj 3',color='#93a5c8'),
    dict(id='fcf_assets_ttm',label='FCF / imovina',short='FCF',color='#4c9f89'),
    dict(id='cfo_assets_ttm',label='CFO / imovina',short='CFO',color='#b89b68'),
    dict(id='gp_assets_ttm',label='Bruto dobit / imovina',short='Bruto dobit',color='#b988a4'),
    dict(id='public_rmw',label='Javni RMW',short='RMW',color='#b17b52'),
    dict(id='risk_free',label='Risk-free / cash',short='RF',color='#74869a')]
COMPONENT_IDS=['fcf_assets_ttm','cfo_assets_ttm','gp_assets_ttm','public_rmw']
for s in SERIES: s['component']=s['id'] in COMPONENT_IDS
RF_FILE=ROOT/'data/reference/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
REGIMES=[
    dict(id='spy_vol21',label='Volatilnost',low='Niska volatilnost',high='Visoka volatilnost',
         description='Annualizirana volatilnost dnevnih SPY prinosa u 21 sesiji. Visoka znači iznad medijana prethodnih mjesečnih opažanja (najmanje 12); tekući mjesec ne ulazi u prag.'),
    dict(id='spy_trend126',label='Trend SPY-a',low='Padajući trend',high='Rastući trend',
         description='Prethodni SPY prinos u 126 sesija. Padajući trend označava prinos ≤ 0, rastući > 0.'),
    dict(id='spy_tlt_corr63',label='Korelacija SPY–TLT',low='Korelacija ≤ 0',high='Korelacija > 0',
         description='Korelacija 63 dnevna prinosa SPY-a i TLT-a. Podjela na nepozitivnu i pozitivnu korelaciju.'),
    dict(id='breadth126',label='Širina tržišta',low='Uska širina',high='Široka širina',
         description='Udio podobnih dionica s pozitivnim prethodnim 126d prinosom. Uska širina: udio ≤ 50%. To je zamjena za udio iznad MA200; najmanje 80% univerzuma mora imati podatak.'),
    dict(id='dispersion21',label='Disperzija prinosa',low='Niska disperzija',high='Visoka disperzija',
         description='IQR prethodnih 21d prinosa dionica / 1,349. Visoka znači iznad medijana prethodnih mjesečnih opažanja, uz najmanje 12.'),
    dict(id='illiquidity21',label='Likvidnost',low='Viša likvidnost',high='Niža likvidnost',
         description='Medijan 21d Amihudove nelikvidnosti u podobnom univerzumu. Veća vrijednost znači nižu likvidnost. Prag je medijan prethodnih mjesečnih opažanja, uz najmanje 12.')]
IC_NAMES={
 'roa_ttm':'ROA · TTM','roe_ttm':'ROE · TTM','gp_assets_ttm':'Bruto dobit / imovina · TTM',
 'op_assets_ttm':'Operativna dobit / imovina · TTM','ebit_assets_ttm':'EBIT / imovina · TTM',
 'cfo_assets_ttm':'CFO / imovina · TTM','fcf_assets_ttm':'FCF / imovina · TTM',
 'gross_margin_ttm':'Bruto marža · TTM','operating_margin_ttm':'Operativna marža · TTM',
 'ebit_margin_ttm':'EBIT marža · TTM','ebitda_margin_ttm':'EBITDA marža · TTM','net_margin_ttm':'Neto marža · TTM',
 'ff2015_fy':'FF2015 OP / kapital · FY','gp_assets_fy':'Bruto dobit / imovina · FY',
 'ball2015_fy':'Ball 2015 · FY','ball2016_fy':'Ball 2016 · FY','cbop_bs_fy':'CBOP bilančni proxy · FY',
 'cbop_cf_fy':'CBOP novčani proxy · FY','roic_fy':'ROIC · FY','hxz_roe_q':'HXZ ROE proxy · Q',
 'qmj_like_fy':'QMJ-like profitabilnost · FY'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def risk_free_windows(monthly):
    prior=json.loads((ROOT/'results/public_comparison/manifest.json').read_text())
    expected=next(s['sha256'] for s in prior['sources'] if Path(s['local_file']).name==RF_FILE.name)
    assert sha(RF_FILE)==expected,'RF archive differs from the saved RMW source vintage'
    daily=read_factor(RF_FILE.name,True)
    calendar=pd.read_csv(ROOT/'results/public_comparison/rmw_aligned_windows.csv',parse_dates=['signal_date','entry_date','exit_date']).set_index('signal_date')
    rows=[]
    for row in monthly.itertuples():
        values=daily.loc[(daily.date>row.entry_date)&(daily.date<=row.exit_date)]
        assert 17<=len(values)<=25 and np.isfinite(values.RF).all()
        original=calendar.loc[row.signal_date]
        assert original.entry_date==row.entry_date and original.exit_date==row.exit_date
        assert len(values)==original.n_daily_observations
        rows.append(dict(signal_date=row.signal_date,entry_date=row.entry_date,exit_date=row.exit_date,
                         risk_free=float(np.prod(1+values.RF)-1),daily_observations=len(values)))
    return pd.DataFrame(rows)


def cash_stats(values,rf,entry,exit):
    equity=np.r_[1,(1+values).cumprod()]
    excess=values-rf
    excess_vol=excess.std(ddof=1)*np.sqrt(12)
    years=(exit.iloc[-1]-entry.iloc[0]).days/365.25
    return dict(cagr=equity[-1]**(1/years)-1,annual_mean=values.mean()*12,
                annual_vol=values.std(ddof=1)*np.sqrt(12),
                sharpe_rf=excess.mean()*12/excess_vol if excess_vol>1e-12 else None,
                monthly_max_drawdown=(equity/np.maximum.accumulate(equity)-1).min())


def build():
    source=OUT/'portfolio_monthly.csv'
    regime_source=ROOT/'results/regimes/regime_features.csv'
    ic_source=ROOT/'results/initial_analysis/monthly_diagnostics.csv'
    monthly=pd.read_csv(source,parse_dates=['signal_date','entry_date','exit_date'])
    rf=risk_free_windows(monthly)
    rf.to_csv(OUT/'risk_free_windows.csv',index=False)
    monthly=monthly.merge(rf,on=['signal_date','entry_date','exit_date'],validate='one_to_one')
    features=pd.read_csv(regime_source,parse_dates=['signal_date','entry_date','exit_date'])
    merged=monthly.merge(features,on=['signal_date','entry_date','exit_date'],validate='one_to_one').sort_values('signal_date')
    assert len(merged)==len(monthly)==235
    assert (merged.signal_date<merged.entry_date).all()
    assert (merged.entry_date<merged.exit_date).all()
    np.testing.assert_allclose(merged.portfolio4,merged[['fcf_assets_ttm','cfo_assets_ttm','gp_assets_ttm','public_rmw']].mean(axis=1),atol=1e-12)
    records=[]
    for i,row in enumerate(merged.itertuples()):
        records.append(dict(i=i,date=row.signal_date.strftime('%Y-%m-%d'),entry=row.entry_date.strftime('%Y-%m-%d'),
            exit=row.exit_date.strftime('%Y-%m-%d'),month=row.entry_date.strftime('%Y-%m'),common=bool(row.common_sample),rf=float(row.risk_free),
            r={s['id']:float(getattr(row,s['id'])) for s in SERIES},
            states={r['id']:None if pd.isna(getattr(row,r['id']+'_state')) else int(getattr(row,r['id']+'_state')) for r in REGIMES}))
    diag=pd.read_csv(ic_source,parse_dates=['signal_date'])
    diag=diag[diag['sample'].eq('own')].set_index(['signal_date','signal'])
    ic_values=[]
    for date in merged.signal_date:
        row=[]
        for name in IC_NAMES:
            if (date,name) not in diag.index:
                row.append(None)
                continue
            d=diag.loc[(date,name)]
            row.append([None if pd.isna(d[x]) else float(d[x]) for x in ['n','n_observed','ic_spearman','ic_cosine_raw','ic_cosine_rank_centered']])
        ic_values.append(row)
    payload=dict(series=SERIES,regimes=REGIMES,months=records,
        commonMonths=int(merged.common_sample.sum()),cashPolicy='RF outside the selected regime, including common-sample warmup; active factor returns unchanged',
        ic=dict(signals=[dict(id=k,label=v) for k,v in IC_NAMES.items()],values=ic_values))
    template=(ASSETS/'portfolio.html').read_text(encoding='utf-8')
    html=template.replace('__REPORT_CSS__',(ASSETS/'portfolio.css').read_text(encoding='utf-8'))
    html=html.replace('__REPORT_JS__',(ASSETS/'portfolio.js').read_text(encoding='utf-8'))
    html=html.replace('__REPORT_METHODOLOGY__',(ASSETS/'portfolio_methodology.html').read_text(encoding='utf-8'))
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<',chr(92)+'u003c')
    html=html.replace('__REPORT_DATA__',encoded)
    assert '__REPORT_' not in html
    static=OUT/'report_static.html'
    current=OUT/'report.html'
    if not static.exists() and current.exists():
        static.write_bytes(current.read_bytes())
    current.write_text(html,encoding='utf-8')
    reference=[]
    for regime in [None,*REGIMES]:
        for level in ([None] if regime is None else [0,1]):
            active=pd.Series(True,index=merged.index) if regime is None else merged.common_sample & merged[regime['id']+'_state'].eq(level)
            for signal in SERIES:
                values=merged[signal['id']].where(active,merged.risk_free)
                reference.append(dict(regime='all' if regime is None else regime['id'],state=level,
                    signal=signal['id'],months=len(merged),active_months=int(active.sum()),cash_months=int((~active).sum()),
                    **cash_stats(values,merged.risk_free,merged.entry_date,merged.exit_date)))
    pd.DataFrame(reference).to_csv(OUT/'regime_cash_summary.csv',index=False)
    sources=[source,regime_source,ic_source,RF_FILE,*[ASSETS/x for x in ['portfolio.html','portfolio.css','portfolio.js','portfolio_methodology.html']]]
    receipt=dict(format='self-contained HTML with inline CSS, JavaScript and historical data',
        code_sha256=sha(__file__),inputs=[dict(path=str(p),sha256=sha(p)) for p in sources],
        monthly_rows=len(records),common_regime_months=payload['commonMonths'],ic_signals=len(IC_NAMES),
        filter_logic='all regime filters intersect; factor active only in existing common sample; every other date-window month earns RF',
        regime_chart='full-calendar factor/RF strategies for both states, retaining other active filters',
        annualization='CAGR over the full date window; Sharpe on monthly strategy return minus the same holding-window RF',
        risk_free='saved French daily RF, decimal returns compounded for entry_date < date <= exit_date; never forward-filled or replaced with zero',
        cash_policy='factor gross return in active months; RF in inactive and unavailable-regime months; no extra RF added in active months',
        conditional_curve='continuous calendar curve including RF cash returns; regime switching costs not estimated',
        chart_comparisons='multiple factor checkboxes and saved snapshots of a series with its date range and regime filters; preserved in URL hash')
    (OUT/'interactive_manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(report=str(current),bytes=current.stat().st_size,months=len(records)),ensure_ascii=False),flush=True)


if __name__=='__main__': build()
