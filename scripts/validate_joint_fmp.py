"""Independent regression, FMP, risk-timing and portfolio checks."""
import json

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata
import statsmodels.api as sm

from analyze_joint_fmp import OUT, ROOT, CACHE, CFG, FACTORS, PROFIT, CONTROLS


def main():
    data=pd.read_parquet(CACHE)
    valid=np.isfinite(data[[*PROFIT,*CONTROLS,'residual_variance_proxy']]).all(axis=1)
    data=data[valid].sort_values(['signal_date','security_id']).set_index(['signal_date','security_id'])
    design=pd.read_parquet(OUT/'design.parquet')
    holdings=pd.read_parquet(OUT/'weights.parquet')
    coefficients=pd.read_csv(OUT/'factor_returns.csv',parse_dates=['signal_date'])
    rf=pd.read_csv(ROOT/'results/factor_portfolio/risk_free_windows.csv',parse_dates=['signal_date']).set_index('signal_date').risk_free
    monthly=pd.read_csv(OUT/'monthly_returns.csv',parse_dates=['signal_date','entry_date','exit_date'])
    weight_groups={(m,d):g.sort_values('security_id') for (m,d),g in holdings.groupby(['model','signal_date'])}
    coef_groups={(m,d):g.set_index('factor') for (m,d),g in coefficients.groupby(['model','signal_date'])}
    monthly_lookup=monthly.set_index(['model','signal','signal_date','scenario','missing_policy'])
    selected_dates=sorted(design.signal_date.unique())[::39]
    max_identity=max_pnl=max_statsmodels=max_transform=0.0
    count=0
    for (method,date),g in design.groupby(['transformation','signal_date']):
        g=g.sort_values('security_id')
        source=data.loc[(date,g.security_id.tolist()),:]
        assert np.array_equal(source.index.get_level_values('security_id'),g.security_id)
        cap=source.FUND_MARKET_CAP_USD_COMPAT.to_numpy()
        x=g[FACTORS].to_numpy()
        np.testing.assert_allclose(np.average(x[:,1:],axis=0,weights=cap),0,atol=1e-12)
        np.testing.assert_allclose(x[:,1:].std(axis=0),1,atol=1e-12)
        if date in selected_dates:
            for j,name in enumerate(FACTORS[1:],start=1):
                raw=source[name].to_numpy()
                v=ndtri((rankdata(raw,method='average')-.5)/len(raw)) if method=='rank_normal' else np.clip(raw,*np.percentile(raw,[1,99]))
                v=(v-np.sum(v*cap)/np.sum(cap))/np.sqrt(np.mean((v-v.mean())**2))
                max_transform=max(max_transform,float(np.max(np.abs(v-x[:,j]))))
                np.testing.assert_allclose(v,x[:,j],atol=1e-12)
        observed=source.return_observed.to_numpy() & np.isfinite(source.forward_return.to_numpy())
        r=np.where(observed,source.forward_return.to_numpy(),0)
        for weighting,wcol in [('sqrt_cap','sqrt_cap_weight'),('inverse_specific_variance','inverse_specific_weight')]:
            model=method+'__'+weighting
            h=weight_groups[model,date]
            assert np.array_equal(h.security_id,g.security_id)
            raw_h=h[['raw_'+s for s in PROFIT]].to_numpy()
            norm_h=h[['gross2_'+s for s in PROFIT]].to_numpy()
            identity=x.T@raw_h
            target=np.eye(len(FACTORS))[:,1:4]
            max_identity=max(max_identity,float(np.max(np.abs(identity-target))))
            np.testing.assert_allclose(identity,target,atol=1e-9)
            np.testing.assert_allclose(np.abs(norm_h).sum(axis=0),2,atol=1e-12)
            np.testing.assert_allclose(norm_h.sum(axis=0),0,atol=1e-10)
            np.testing.assert_allclose(norm_h,raw_h*2/np.abs(raw_h).sum(axis=0),atol=1e-12)
            np.testing.assert_allclose(h.gross2_basket3,norm_h.mean(axis=1),atol=1e-12)
            f=coef_groups[model,date]
            error=float(np.max(np.abs(raw_h.T@r-f.loc[PROFIT,'factor_return'].to_numpy())))
            max_pnl=max(max_pnl,error)
            assert error<1e-10
            if date in selected_dates:
                independent=sm.WLS(r-rf.loc[date],x,weights=g[wcol].to_numpy()).fit().params
                error=float(np.max(np.abs(independent-f.loc[FACTORS,'factor_return'].to_numpy())))
                max_statsmodels=max(max_statsmodels,error)
                assert error<1e-10
            for j,signal in enumerate([*PROFIT,'basket3','basket3_gross2']):
                weights=norm_h[:,j] if j<3 else norm_h.mean(axis=1)
                if signal=='basket3_gross2': weights=weights*2/np.abs(weights).sum()
                entry=source.entry_contract_ok.eq(True).to_numpy() & (source.entry_price.to_numpy()>0)
                weights=np.where(entry,weights,0)
                long,short=weights>0,weights<0
                gross_long=weights[long].sum();gross_short=-weights[short].sum()
                accepted=min(gross_long,gross_short)
                weights[long]*=accepted/gross_long
                weights[short]*=accepted/gross_short
                for adverse in [False,True]:
                    underlying=r.copy()
                    if adverse: underlying[~observed]=np.where(weights[~observed]>0,-1,1)
                    expected=weights@underlying
                    policy='adverse_unknown' if adverse else 'flat_unknown'
                    row=monthly_lookup.loc[(model,signal,date,'gross',policy)]
                    np.testing.assert_allclose(row['return'],expected,atol=1e-11)
                    count+=1
    # Rebuild the risk forecast independently; prior month's holding return is
    # deliberately unavailable at formation, since its exit is the next session.
    ready_checks=0;max_risk=0.0
    decay=.5**(1/CFG['specific_variance']['ewma_half_life_months'])
    for method,frame in design.groupby('transformation'):
        history={};counts={};queue=[]
        for date,g in frame.groupby('signal_date',sort=True):
            g=g.sort_values('security_id')
            source=data.loc[(date,g.security_id.tolist()),:]
            while queue and queue[0][0]<=date:
                exit_date,ids,residual=queue.pop(0)
                for sid,e in zip(ids,residual):
                    history[sid]=decay*history.get(sid,e*e)+(1-decay)*e*e
                    counts[sid]=counts.get(sid,0)+1
            ids=g.security_id.to_numpy()
            ready=np.array([counts.get(s,0)>=12 for s in ids])
            d=np.array([history.get(s,np.nan) for s in ids])
            d=np.where(ready,d,source.residual_variance_proxy)
            d=np.maximum(d,.1*np.median(d[d>0]))
            np.testing.assert_allclose(g.specific_variance,d,rtol=1e-9,atol=1e-12)
            np.testing.assert_array_equal(g.risk_history_ready,ready)
            w=1/d;w=np.clip(w,*np.quantile(w,[.01,.99]));w/=w.mean()
            max_risk=max(max_risk,float(np.max(np.abs(w-g.inverse_specific_weight))))
            np.testing.assert_allclose(g.inverse_specific_weight,w,rtol=1e-9,atol=1e-10)
            last=g.risk_latest_exit.iloc[0]
            assert pd.isna(last) or last<=date
            x=g[['market',*CONTROLS]].to_numpy()
            known=source.return_observed.to_numpy() & np.isfinite(source.forward_return.to_numpy())
            y=source.forward_return.to_numpy()[known]-rf.loc[date]
            cw=g.sqrt_cap_weight.to_numpy()[known]
            coef=sm.WLS(y,x[known],weights=cw).fit().params
            residual=y-x[known]@coef
            queue.append((source.exit_date.iloc[0],ids[known],residual))
            ready_checks+=1
    # Held universe is independent of whether future returns are observed.
    for date,g in holdings[holdings.model==CFG['primary']].groupby('signal_date'):
        assert set(g.security_id)==set(data.loc[date].index)
    receipt=dict(status='passed',fmp_return_cells=count,formations=235,
        independent_risk_forecasts=ready_checks,max_identity_error=max_identity,
        max_raw_fmp_coefficient_error=max_pnl,max_statsmodels_error=max_statsmodels,
        max_transform_error=max_transform,max_risk_weight_error=max_risk,
        checks=['independent transforms','all monthly unit/zero exposures','gross=2 normalization',
                'FMP PnL equals joint coefficient','independent statsmodels WLS',
                'all normalized gross main/adverse portfolio returns',
                'full reconstruction of delayed EWMA risk and fallback',
                'formation universe includes unknown future outcomes'])
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
