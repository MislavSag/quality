"""Check causal timing, source lineage, calendar HAC and conditional aggregation."""
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.api as sm

import analyze_regimes as r


def main():
    features = pd.read_csv(r.OUT/'regime_features.csv',parse_dates=['signal_date','entry_date','exit_date']).set_index('signal_date')
    calendar = features.reset_index()[['signal_date','entry_date','exit_date']]
    etf = pd.read_parquet(r.CACHE/'regime_etf_daily.parquet')
    etf['trade_date'] = pd.to_datetime(etf.trade_date)
    cross = pd.read_parquet(r.CACHE/'regime_monthly_cross_section.parquet')
    panel = pd.read_parquet(r.CACHE/'monthly_panel.parquet',columns=['security_id','signal_date','eligible'])
    keys = panel.loc[panel.eligible,['security_id','signal_date']]
    assert len(cross)==len(keys)==431233
    pd.testing.assert_frame_equal(cross[keys.columns].sort_values(list(keys.columns)).reset_index(drop=True),
                                  keys.sort_values(list(keys.columns)).reset_index(drop=True))
    assert not cross.duplicated(['security_id','signal_date']).any()
    assert (features.index < features.entry_date).all()
    assert (features.entry_date < features.exit_date).all()
    full = r.construct_features(etf,cross,calendar)
    pd.testing.assert_frame_equal(features,full,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-12)

    # Prefix recomputation and arbitrary future data changes cannot alter the past.
    for cut in [pd.Timestamp('2016-12-30'),pd.Timestamp('2021-12-31')]:
        early = calendar.signal_date<=cut
        truncated = r.construct_features(etf[etf.trade_date<=cut],cross[cross.signal_date<=cut],calendar[early])
        pd.testing.assert_frame_equal(full.loc[full.index<=cut],truncated,check_freq=False)
        changed_etf,changed_cross = etf.copy(),cross.copy()
        ec = changed_etf.trade_date>cut
        for column in ['PV_RET_CC_1D','PV_REALIZED_VOL_21D','PV_RET_CC_126D']:
            changed_etf.loc[ec,column] = changed_etf.loc[ec,column]*-17+31
        cc = changed_cross.signal_date>cut
        changed_cross.loc[cc,['PV_RET_CC_21D','PV_RET_CC_126D','PV_AMIHUD_ILLIQ_21D']]=999
        changed = r.construct_features(changed_etf,changed_cross,calendar)
        pd.testing.assert_frame_equal(full.loc[full.index<=cut],changed.loc[changed.index<=cut])

    # Independent source reads for 9 ETF dates, including QC contract status.
    dates = full.index[[24,36,60,84,108,132,168,204,234]]
    source = ds.dataset(str(r.a.PRICES)).to_table(
        columns=['ticker','trade_date','PV_RET_CC_1D','PV_REALIZED_VOL_21D',
                 'PV_RET_CC_126D','PV_QC_FACTOR_CONTRACT_OK'],
        filter=ds.field('ticker').isin(['SPY','TLT']) & ds.field('trade_date').isin(dates.tolist())
    ).to_pandas(date_as_object=False)
    assert len(source)==18 and source.PV_QC_FACTOR_CONTRACT_OK.eq(True).all()
    cached = etf[etf.trade_date.isin(dates)]
    merged = cached.merge(source,on=['ticker','trade_date'],suffixes=('_cached','_source'),validate='one_to_one')
    for column in ['PV_RET_CC_1D','PV_REALIZED_VOL_21D','PV_RET_CC_126D']:
        np.testing.assert_allclose(merged[column+'_cached'],merged[column+'_source'],equal_nan=True)
    returns = etf.pivot(index='trade_date',columns='ticker',values='PV_RET_CC_1D').sort_index()
    spy = etf[etf.ticker.eq('SPY')].set_index('trade_date').sort_index()
    for date in dates:
        sample = returns.loc[:date].tail(63)
        np.testing.assert_allclose(np.corrcoef(sample.SPY,sample.TLT)[0,1],full.loc[date,'spy_tlt_corr63'])
        rv = spy.loc[:date].PV_RET_CC_1D.tail(21).std(ddof=1)*np.sqrt(252)
        np.testing.assert_allclose(rv,full.loc[date,'spy_vol21'])
        prices = spy.loc[:date].PV_CLOSE
        np.testing.assert_allclose(prices.iloc[-1]/prices.iloc[-127]-1,full.loc[date,'spy_trend126'])
        group = cross[cross.signal_date.eq(date)]
        np.testing.assert_allclose(group.PV_RET_CC_126D.dropna().gt(0).mean(),full.loc[date,'breadth126'])
        v = group.PV_RET_CC_21D.dropna().to_numpy()
        np.testing.assert_allclose((np.quantile(v,.75)-np.quantile(v,.25))/1.349,full.loc[date,'dispersion21'])
        np.testing.assert_allclose(np.nanmedian(group.PV_AMIHUD_ILLIQ_21D),full.loc[date,'illiquidity21'])
        for name in ['spy_vol21','dispersion21','illiquidity21']:
            np.testing.assert_allclose(np.nanmedian(full.loc[full.index<date,name]),full.loc[date,name+'_threshold'])

    # Compare HAC against a separate library on a complete calendar.
    rng = np.random.default_rng(73)
    x = np.column_stack([np.ones(180),rng.normal(size=180)])
    y = .04+.2*x[:,1]+rng.normal(size=180)*.1
    fit = r.calendar_fit(y,x)
    independent = sm.OLS(y,x).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    np.testing.assert_allclose(fit['beta'],independent.params,atol=1e-12)
    np.testing.assert_allclose(fit['cov'],independent.cov_params(),atol=1e-12)
    # Gap-aware HAC covariance equals explicit calendar pair sums.
    missing = np.arange(0,len(y),7)
    ym = y.copy()
    ym[missing]=np.nan
    gapfit = r.calendar_fit(ym,x)
    ok = np.isfinite(ym)
    score = np.zeros_like(x)
    score[ok]=x[ok]*(ym[ok]-x[ok]@gapfit['beta'])[:,None]
    meat = np.zeros((2,2))
    for t in range(len(y)):
        for u in range(len(y)):
            distance=abs(t-u)
            if distance<=3:
                meat+=(1-distance/4)*np.outer(score[t],score[u])
    inverse = np.linalg.inv(x[ok].T@x[ok])
    np.testing.assert_allclose(gapfit['cov'],inverse@meat@inverse,atol=1e-12)

    summary = pd.read_csv(r.OUT/'conditional_returns.csv')
    paths = pd.read_csv(r.ROOT/'results/monthly_long_short/monthly_returns.csv',parse_dates=['signal_date'])
    common_dates = features.index[features.common_sample]
    checks = 0
    own = summary[summary.signal.isin(r.CONFIG['signals']) & (summary.period=='full')]
    for (signal,scenario,policy),group in own.groupby(['signal','scenario','missing_policy']):
        path = paths[(paths.signal==signal)&(paths.scenario==scenario)&(paths.missing_policy==policy)]
        expected = path.set_index('signal_date').loc[common_dates,'return']
        for regime,partition in group.groupby('regime'):
            assert partition.months.sum()==len(common_dates)
            np.testing.assert_allclose(np.average(partition.annual_mean,weights=partition.months),expected.mean()*12,atol=1e-12)
            for _,row in partition.iterrows():
                mask = features.loc[common_dates,regime+'_state']==row.state
                np.testing.assert_allclose(expected[mask].mean()*12,row.annual_mean,atol=1e-12)
                checks+=1
    # IC reconciliation uses the exact previously published monthly series.
    ic = pd.read_csv(r.OUT/'conditional_ic.csv')
    diagnostic = pd.read_csv(r.ROOT/'results/initial_analysis/monthly_diagnostics.csv',parse_dates=['signal_date'])
    diagnostic = diagnostic[diagnostic['sample'].eq('own')]
    for signal in r.CONFIG['signals']:
        original = diagnostic[diagnostic.signal.eq(signal)].set_index('signal_date')
        for regime in r.REGIMES:
            for state in [0,1]:
                selected = features.index[features.common_sample & features[regime+'_state'].eq(state)]
                row = ic[(ic.signal==signal)&(ic.period=='full')&(ic.regime==regime)&(ic.state==state)].iloc[0]
                for metric in ['ic_spearman','ic_cosine_raw','ic_cosine_rank_centered']:
                    np.testing.assert_allclose(original.loc[selected,metric].mean(),row[metric],atol=1e-12)
    result = dict(status='passed',common_months=len(common_dates),eligible_cross_section_rows=len(cross),
        causal_prefix_and_future_perturbation_cutoffs=2,independent_source_etf_rows=18,
        independent_feature_dates=9,hac_library_comparison='passed',calendar_gap_covariance='passed',
        conditional_return_cells=checks,ic_cells=108,
        limitations='Verifies computation and sampled lineage; not first-release fundamentals or unknown delisting payoffs')
    (r.OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
