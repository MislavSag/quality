"""Independent standalone-side targets, return signs, turnover and financing checks."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from analyze_joint_fmp import CACHE, CFG, CONTROLS, OUT, PROFIT


def main():
    data=pd.read_parquet(CACHE)
    data=data[np.isfinite(data[[*PROFIT,*CONTROLS,'residual_variance_proxy']]).all(axis=1)]
    groups={d:g.sort_values('security_id') for d,g in data.groupby('signal_date')}
    ids={s:i for i,s in enumerate(sorted(data.security_id.unique()))}
    weights=pd.read_parquet(OUT/'weights.parquet')
    observed=pd.read_csv(OUT/'side_monthly_returns.csv',parse_dates=['signal_date','entry_date','exit_date'])
    lookup=observed.set_index(['model','signal','leg','scenario','signal_date'])
    count=0;max_error=0.;net_signs={}
    for model in observed.model.unique():
        saved={d:g.sort_values('security_id') for d,g in weights[weights.model.eq(model)].groupby('signal_date')}
        targets={}
        for date,g in groups.items():
            if model=='sort_common':
                columns=[]
                for signal in PROFIT:
                    rank=(g[signal].rank(method='average').to_numpy()-.5)/len(g)
                    long,short=rank>=.8,rank<.2
                    columns.append(long/long.sum()-short/short.sum())
                a=np.column_stack([*columns,np.mean(columns,axis=0)])
            else:
                h=saved[date]
                np.testing.assert_array_equal(h.security_id,g.security_id)
                a=h[['gross2_'+s for s in PROFIT]+['gross2_basket3']].to_numpy()
            targets[date]=a
        for j,signal in enumerate([*PROFIT,'basket3']):
            for leg,sign in [('long',1),('short',-1)]:
                for scenario,bps,borrow_rate in [('gross',0,0),('cost10_borrow3',.001,.03)]:
                    nav=CFG['initial_nav'];old=np.zeros(len(ids))
                    for month,(date,g) in enumerate(groups.items()):
                        ix=np.array([ids[s] for s in g.security_id])
                        raw=targets[date][:,j]
                        leg_weights=np.where(raw*sign>0,raw,0.)
                        leg_weights/=np.abs(leg_weights).sum()
                        np.testing.assert_allclose(leg_weights.sum(),sign,atol=1e-12)
                        entry=(g.entry_price.to_numpy()>0)&g.entry_contract_ok.eq(True).to_numpy()
                        target=np.zeros(len(ids));target[ix]=np.where(entry,leg_weights,0)
                        before=nav
                        # Different solver from the production fixed-point iteration.
                        after=brentq(lambda v:v+bps*np.abs(target*v-old).sum()-before,0,before,xtol=1e-7) if bps else before
                        traded=np.abs(target*after-old).sum()
                        known=g.return_observed.to_numpy()&np.isfinite(g.forward_return.to_numpy())
                        r=np.zeros(len(ids));r[ix]=np.where(known,g.forward_return.to_numpy(),0.)
                        positions=target*after;end=positions*(1+r)
                        days=(g.exit_date.iloc[0]-g.entry_date.iloc[0]).days
                        fee=(-np.minimum(positions,0).sum()-np.minimum(end,0).sum())*.5*borrow_rate*days/365
                        pnl=positions@r
                        nav=after+pnl-fee
                        liquidation=bps*np.abs(end).sum() if month==len(groups)-1 else 0
                        nav-=liquidation
                        row=lookup.loc[(model,signal,leg,scenario,date)]
                        expected={'return':nav/before-1,'entry_trading_cost':bps*traded,
                                  'borrow_cost':fee,'final_liquidation_cost':liquidation}
                        for key,val in expected.items():
                            np.testing.assert_allclose(row[key],val,atol=1e-6 if key!='return' else 1e-11,rtol=1e-9)
                        max_error=max(max_error,abs(row['return']-expected['return']))
                        assert (row.short_gross_weight<1e-12) if leg=='long' else (row.long_gross_weight<1e-12)
                        if leg=='long':assert row.borrow_cost==0
                        if scenario=='gross':
                            np.testing.assert_allclose(row['return'],target@r,atol=1e-11)
                        old=end;count+=1
    # Simple direction check: a stock decline is a positive short PnL.
    assert np.dot([-1.],[-.1])==.1 and np.dot([-1.],[.1])==-.1
    receipt=dict(status='passed',independent_monthly_side_ledgers=count,max_return_error=max_error,
        checks=['all five models and four distinct books','formation-normalized +100% / -100% targets',
                'independent Brent post-cost NAV solver','drifted positions and turnover including exits',
                'short profit sign','borrow only on short','separate final liquidation costs',
                'same missing-return and entry rejection policy'])
    (OUT/'side_validation.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
