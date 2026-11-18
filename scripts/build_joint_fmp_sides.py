"""Independent 100% long and 100% short books from saved formation FMP weights.

These are separately rebalanced portfolios, not an arbitrary allocation of the
long-short strategy's total costs. A positive short return means a short profit.
"""
import json

import numpy as np
import pandas as pd

import backtest_monthly_long_short as accounting
from analyze_joint_fmp import CACHE, CFG, CONTROLS, OUT, PROFIT, ROOT, make_book, sha
from build_interactive_report import cash_stats


def main():
    data=pd.read_parquet(CACHE)
    valid=np.isfinite(data[[*PROFIT,*CONTROLS,'residual_variance_proxy']]).all(axis=1)
    data=data[valid].sort_values(['signal_date','security_id'])
    groups={d:g for d,g in data.groupby('signal_date',sort=True)}
    weights=pd.read_parquet(OUT/'weights.parquet')
    scenarios=[dict(name='gross',trading_bps=0,borrow_rate=0,unavailable_short_fraction=0,short_adv_fraction_cap=None),
               dict(name='cost10_borrow3',trading_bps=10,borrow_rate=.03,unavailable_short_fraction=0,short_adv_fraction_cap=None)]
    all_rows=[]
    models=[a+'__'+b for a in CFG['transformations'] for b in CFG['weighting']]+['sort_common']
    for model in models:
        saved={d:g.sort_values('security_id') for d,g in weights[weights.model.eq(model)].groupby('signal_date')}
        books={signal:{leg:[] for leg in ['long','short']} for signal in [*PROFIT,'basket3']}
        for date,group in groups.items():
            if model=='sort_common':
                parts=[]
                for signal in PROFIT:
                    rank=(group[signal].rank(method='average').to_numpy()-.5)/len(group)
                    lo,hi=rank<.2,rank>=.8
                    parts.append(hi/hi.sum()-lo/lo.sum())
                targets=np.column_stack([*parts,np.mean(parts,axis=0)])
            else:
                w=saved[date]
                assert np.array_equal(w.security_id,group.security_id)
                targets=w[['gross2_'+s for s in PROFIT]+['gross2_basket3']].to_numpy()
            for j,signal in enumerate([*PROFIT,'basket3']):
                for leg,sign in [('long',1),('short',-1)]:
                    selected=targets[:,j]*sign>0
                    target=targets[selected,j]
                    # Each side has 100% formation notional, even for the
                    # unrescaled basket. Entry rejections are not refilled.
                    target=target/np.abs(target).sum()
                    books[signal][leg].append(make_book(group.loc[selected],target))
        for signal,legs in books.items():
            for leg,sequence in legs.items():
                for scenario in scenarios:
                    result,_=accounting.simulate(sequence,scenario,dict(initial_nav=CFG['initial_nav'],balance_sides=False),
                                                  record_holdings=False)
                    result['model'],result['signal'],result['leg'],result['scenario']=model,signal,leg,scenario['name']
                    all_rows.append(result)
        print('Completed separate sides:',model,flush=True)
    result=pd.concat(all_rows,ignore_index=True)
    result.to_csv(OUT/'side_monthly_returns.csv',index=False)
    rates=pd.read_csv(ROOT/'results/factor_portfolio/risk_free_windows.csv',parse_dates=['signal_date']).set_index('signal_date').risk_free
    summary=[]
    for key,g in result.groupby(['model','signal','leg','scenario'],sort=False):
        summary.append(dict(zip(['model','signal','leg','scenario'],key),
            **cash_stats(g['return'],g.signal_date.map(rates),g.entry_date,g.exit_date)))
    pd.DataFrame(summary).to_csv(OUT/'side_summary.csv',index=False)
    receipt=dict(code_sha256=sha(__file__),accounting_code_sha256=sha(accounting.__file__),
        inputs={str(p):sha(p) for p in [CACHE,OUT/'weights.parquet']},
        definition='separate monthly portfolios: long +100%, short -100%; weights normalized at formation before entry rejection',
        short_sign='positive return when the short book makes money; not the positive-weight return of shorted stocks',
        costs='separate drifted books, own turnover and final liquidation; borrow only on negative positions; no cash/collateral interest or financing in active months',
        basket='net component holdings first, then select sign and normalize to 100%; gross2 and unrescaled basket have identical single-side targets',
        unavailable='public RMW and composites containing RMW have no observed leg holdings',
        unknown_returns='zero underlying return and liquidation assumption, consistent with primary FMP report')
    (OUT/'side_manifest.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
