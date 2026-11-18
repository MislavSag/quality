"""Independent score, timing, common-sample and portfolio reconciliation checks."""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr

import build_quality_profitability_bundle as b


def main():
    data = pd.read_parquet(b.a.CACHE/'monthly_panel.parquet')
    saved = pd.read_parquet(b.OUT/'monthly_bundle_scores.parquet')
    rebuilt = b.build_scores(data)
    pd.testing.assert_frame_equal(saved,rebuilt)
    pd.testing.assert_frame_equal(saved[['security_id','ticker','signal_date','eligible']],
                                  data[['security_id','ticker','signal_date','eligible']])
    assert not saved.duplicated(['security_id','signal_date']).any()
    valid = data.eligible & np.isfinite(data[b.COMPONENTS]).all(axis=1)
    assert saved.bundle4.notna().eq(valid).all()
    assert saved.loc[~data.eligible,'bundle4'].isna().all()
    assert saved.bundle4_score.dropna().between(0,100).all()
    np.testing.assert_allclose(saved.bundle4_score,50*(saved.bundle4+1),equal_nan=True)
    monthly = pd.read_csv(b.OUT/'monthly_returns.csv',parse_dates=['signal_date','entry_date','exit_date'])
    diagnostic = pd.read_csv(b.OUT/'monthly_ic.csv',parse_dates=['signal_date'])
    dates = data.signal_date.drop_duplicates().sort_values().iloc[[0,24,60,84,108,132,168,204,234]]
    for date in dates:
        index = data.index[data.signal_date.eq(date) & valid]
        ranks = np.column_stack([(rankdata(data.loc[index,c],method='average')-.5)/len(index)
                                 for c in b.COMPONENTS])
        np.testing.assert_allclose(saved.loc[index,'bundle4_score'],ranks.mean(axis=1)*100)
        np.testing.assert_allclose(saved.loc[index,'bundle3_common4'],2*ranks[:,:3].mean(axis=1)-1)
        assert abs(saved.loc[index,'bundle4'].mean())<1e-12
        g = data.loc[index].copy()
        g['bundle4']=saved.loc[index,'bundle4']
        observed = g.return_observed & np.isfinite(g.forward_return)
        row = diagnostic[(diagnostic.signal=='bundle4') & (diagnostic.signal_date==date)].iloc[0]
        np.testing.assert_allclose(spearmanr(g.loc[observed,'bundle4'],g.loc[observed,'forward_return']).statistic,row.ic_spearman)
        x = g.loc[observed,'bundle4'].to_numpy()
        y = g.loc[observed,'forward_return'].to_numpy()
        np.testing.assert_allclose(x@y/(np.linalg.norm(x)*np.linalg.norm(y)),row.ic_cosine_raw,atol=1e-12)
        centered_ranks = pd.Series(rankdata(g.bundle4,method='average'),index=g.index)
        centered_ranks-=centered_ranks.mean()
        x = centered_ranks.loc[observed].to_numpy()
        np.testing.assert_allclose(x@y/(np.linalg.norm(x)*np.linalg.norm(y)),row.ic_cosine_rank_centered,atol=1e-12)
        # Independent opening weights and gross first-month P&L. Costs are zero.
        percentile = (rankdata(g.bundle4,method='average')-.5)/len(g)
        sides = np.where(percentile>=.8,1,np.where(percentile<.2,-1,0))
        executable = np.isfinite(g.entry_price)&g.entry_price.gt(0)&g.entry_contract_ok.eq(True)
        w = np.zeros(len(g))
        for side in [1,-1]:
            mask=sides==side
            w[mask]=side/mask.sum()
        w[~executable]=0
        gross = min(w[w>0].sum(),-w[w<0].sum())
        for side in [1,-1]:
            mask=w*side>0
            w[mask]*=gross/abs(w[mask]).sum()
        used=g.forward_return.where(observed,0).to_numpy()
        expected=w@used
        result = monthly[(monthly.signal=='bundle4')&(monthly.scenario=='gross_unconstrained')
                         &(monthly.missing_policy=='flat_unknown')&(monthly.signal_date==date)].iloc[0]
        np.testing.assert_allclose(expected,result['return'],atol=1e-12)

    # Future outcomes and entry tradability cannot influence formation scores.
    changed = data.copy()
    changed['forward_return'] = 99
    changed['return_observed'] = False
    changed['entry_price'] = np.nan
    changed['entry_contract_ok'] = False
    pd.testing.assert_frame_equal(saved,b.build_scores(changed))
    del changed
    cut = pd.Timestamp('2016-12-30')
    pd.testing.assert_frame_equal(saved[data.signal_date<=cut],b.build_scores(data[data.signal_date<=cut]))
    # Strictly increasing transformations preserve ranks; all future predictors can change.
    changed = data.copy()
    for c in b.COMPONENTS:
        changed[c] = changed[c]*3+7
    pd.testing.assert_frame_equal(saved,b.build_scores(changed))
    del changed
    # Exact ties use average ranks; missing fourth component never falls back to three.
    small = data.loc[data.signal_date.eq(dates.iloc[0]) & valid].head(110).copy()
    small.loc[:,b.COMPONENTS]=1.
    small.loc[small.index[0],'ff2015_fy']=np.nan
    test=b.build_scores(small)
    assert np.isnan(test.loc[small.index[0],'bundle4'])
    np.testing.assert_allclose(test.bundle4.dropna(),0,atol=1e-12)
    np.testing.assert_allclose(test.bundle3_score,50,atol=1e-12)

    common = diagnostic[diagnostic.signal.ne('bundle3')]
    assert common.groupby('signal_date').n.nunique().eq(1).all()
    assert common.groupby('signal_date').n_observed.nunique().eq(1).all()
    assert (monthly.signal_date<monthly.entry_date).all()
    assert (monthly.entry_date<monthly.exit_date).all()
    assert not monthly.duplicated(['signal','scenario','missing_policy','signal_date']).any()
    expected=monthly.nav_start-monthly.entry_trading_cost+monthly.gross_pnl-monthly.borrow_cost-monthly.final_liquidation_cost
    np.testing.assert_allclose(monthly.nav_end,expected,atol=1e-6,rtol=1e-12)
    np.testing.assert_allclose(monthly.long_gross_weight,monthly.short_gross_weight,atol=1e-12)
    np.testing.assert_allclose(monthly.long_pnl+monthly.short_pnl,monthly.gross_pnl,atol=1e-6,rtol=1e-12)
    paths=monthly.groupby(['signal','scenario','missing_policy'])
    assert paths.ngroups==70 and len(monthly)==16450
    for _,g in paths:
        assert len(g)==235
        np.testing.assert_allclose(g.nav_start.iloc[1:],g.nav_end.iloc[:-1],atol=1e-6,rtol=1e-12)
        np.testing.assert_allclose((1+g['return']).cumprod(),g.nav_end/g.nav_start.iloc[0],atol=1e-12)
    # Actual four-return basket is kept separate from all stock score computations.
    basket=pd.read_csv(b.OUT/'four_factor_basket.csv')
    np.testing.assert_allclose(basket.basket4_public_rmw,
        basket[['fcf_assets_ttm','cfo_assets_ttm','gp_assets_ttm','public_rmw']].mean(axis=1),atol=1e-12)
    latest = pd.read_csv(b.OUT/'latest_bundle_scores.csv')
    assert len(latest)==saved[saved.signal_date.eq(saved.signal_date.max())].bundle4.notna().sum()
    assert latest.bundle4_score.is_monotonic_decreasing
    result=dict(status='passed',monthly_score_rows=len(saved),bundle4_rows=int(valid.sum()),
        independent_score_ic_and_gross_return_dates=9,
        no_future_outcome_or_execution_dependency=True,prefix_reconstruction=True,
        monotonic_transform_invariance=True,ties_and_strict_missingness=True,
        common_sample_equality=True,portfolio_paths=paths.ngroups,monthly_return_rows=len(monthly),
        return_basket_arithmetic=True,
        limitations='Verifies implementation; not first-release accounting data, executable shorting or true delisting payoffs')
    (b.OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
