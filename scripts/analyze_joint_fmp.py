"""Joint characteristic regressions and ex-ante factor-mimicking portfolios.

No source files are changed. Raw unit-exposure FMPs and gross-2 portfolios are
distinct outputs. Future return availability never selects formation holdings.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import ndtri

import analyze_quality as base
import backtest_monthly_long_short as accounting
from build_interactive_report import cash_stats

ROOT = base.ROOT
OUT = ROOT / 'results/joint_fmp'
CONFIG_PATH = ROOT / 'config/joint_fmp.json'
CFG = json.loads(CONFIG_PATH.read_text())
PROFIT = CFG['profitability']
CONTROLS = CFG['controls']
FACTORS = ['market', *PROFIT, *CONTROLS]
PV_COLS = ['PV_BETA_SPY_63D', 'PV_REALIZED_VOL_63D', 'PV_SYSTEMATIC_VOL_SPY_63D']
CACHE = base.CACHE / 'joint_fmp_inputs.parquet'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    prior = json.loads((base.CACHE / 'contract.json').read_text())
    current = [base.fingerprint(p) for p in [base.PRICES, base.FUND, base.EXTFILE]]
    assert prior['inputs'] == current, 'Existing monthly panel is stale'
    contract = dict(inputs=current, monthly_sha256=sha(base.CACHE/'monthly_panel.parquet'),
                    pv_columns=PV_COLS, version=1)
    stamp = CACHE.with_suffix('.json')
    if CACHE.exists() and stamp.exists() and json.loads(stamp.read_text()) == contract:
        base.log('Reusing joint-regression formation input cache')
        return pd.read_parquet(CACHE)
    columns = ['security_id', 'ticker', 'signal_date', 'entry_date', 'exit_date', 'eligible',
               'entry_price', 'entry_contract_ok', 'return_observed', 'forward_return',
               'FUND_MARKET_CAP_USD_COMPAT', 'PV_AVG_DOLLAR_VOLUME_21D',
               'log_size', 'value', 'momentum', *PROFIT]
    panel = pd.read_parquet(base.CACHE/'monthly_panel.parquet', columns=columns)
    panel = panel[panel.eligible].copy()
    dates = sorted(panel.signal_date.dt.date.unique())
    base.log('Reading only month-end beta and volatility columns from price source')
    controls = base.selected_scan(base.PRICES, ['security_id', 'trade_date', *PV_COLS], dates)
    controls = controls.to_pandas(date_as_object=False).rename(columns={'trade_date':'signal_date'})
    assert not controls.duplicated(['security_id','signal_date']).any()
    panel = panel.merge(controls, on=['security_id','signal_date'], how='left', validate='one_to_one')
    liquid = pd.read_parquet(base.CACHE/'regime_monthly_cross_section.parquet',
                            columns=['security_id','signal_date','PV_AMIHUD_ILLIQ_21D'])
    panel = panel.merge(liquid, on=['security_id','signal_date'], how='left', validate='one_to_one')
    panel['beta'] = panel.PV_BETA_SPY_63D
    residual_var = panel.PV_REALIZED_VOL_63D**2 - panel.PV_SYSTEMATIC_VOL_SPY_63D**2
    # Same-window SPY decomposition, not a claim of full multifactor specific risk.
    panel['residual_variance_proxy'] = (residual_var / 12).where(residual_var > 0)
    panel['log_residual_vol'] = np.log(np.sqrt(residual_var.where(residual_var > 0)))
    panel['log_illiquidity'] = np.log(panel.PV_AMIHUD_ILLIQ_21D.clip(lower=1e-12))
    panel.to_parquet(CACHE, index=False)
    stamp.write_text(json.dumps(contract, indent=2))
    return panel


def transform(values, cap, method):
    values = np.asarray(values, dtype=float)
    if method == 'rank_normal':
        values = ndtri((pd.Series(values).rank(method='average').to_numpy()-.5)/len(values))
    else:
        lo, hi = np.quantile(values, CFG['winsor_quantiles'])
        values = np.clip(values, lo, hi)
    scale = np.std(values, ddof=0)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Degenerate formation exposure')
    return (values-np.average(values, weights=cap))/scale


def design(group, method):
    cap = group.FUND_MARKET_CAP_USD_COMPAT.to_numpy()
    return np.column_stack([np.ones(len(group)), *[
        transform(group[name].to_numpy(), cap, method) for name in FACTORS[1:]]])


def fit_fmp(x, w, y):
    """Use weighted QR, avoiding normal-equation conditioning in estimation."""
    root_w = np.sqrt(w / np.mean(w))
    a = root_w[:, None]*x
    q, r = np.linalg.qr(a, mode='reduced')
    condition = np.linalg.cond(a)
    if np.linalg.matrix_rank(r) != x.shape[1] or condition > 1e8:
        raise ValueError(f'Unidentified/unstable design: condition={condition}')
    # A = (X'WX)^-1 X'W, computed without explicitly inverting X'WX.
    operator = np.linalg.solve(r, q.T)*root_w[None, :]
    coef = operator @ y
    return coef, operator.T, condition


class SpecificRisk:
    """Monthly baseline residual second moments, updated only after exit close."""
    def __init__(self):
        self.variance, self.count, self.last_exit = {}, {}, {}
        self.pending = []
        self.decay = .5**(1/CFG['specific_variance']['ewma_half_life_months'])

    def forecast(self, group):
        date = group.signal_date.iloc[0]
        while self.pending and self.pending[0][0] <= date:
            exit_date, ids, residual = self.pending.pop(0)
            for sid, value in zip(ids, residual):
                previous = self.variance.get(sid, value**2)
                self.variance[sid] = self.decay*previous+(1-self.decay)*value**2
                self.count[sid] = self.count.get(sid, 0)+1
                self.last_exit[sid] = exit_date
        min_n = CFG['specific_variance']['minimum_observed_residuals']
        ready = np.array([self.count.get(sid,0)>=min_n for sid in group.security_id])
        variance = np.array([self.variance.get(sid,np.nan) for sid in group.security_id])
        variance = np.where(ready, variance, group.residual_variance_proxy.to_numpy())
        floor = np.median(variance[variance>0])*CFG['specific_variance']['floor_fraction_of_cross_section_median']
        variance = np.maximum(variance, floor)
        w = 1/variance
        w = np.clip(w, *np.quantile(w, CFG['specific_variance']['weight_clip_quantiles']))
        exits = [self.last_exit[sid] for sid in group.security_id if sid in self.last_exit]
        last = max(exits) if exits else pd.NaT
        assert pd.isna(last) or last <= date
        return w/np.mean(w), ready, variance, last

    def enqueue(self, group, x, y, cap_weights):
        # No fabricated zero-return observations enter the risk history.
        known = group.return_observed.to_numpy() & np.isfinite(group.forward_return.to_numpy())
        controls = x[:, [0, *range(4, x.shape[1])]]
        a = controls[known]*np.sqrt(cap_weights[known, None])
        coef = np.linalg.lstsq(a, y[known]*np.sqrt(cap_weights[known]), rcond=None)[0]
        residual = y[known]-controls[known]@coef
        self.pending.append((group.exit_date.iloc[0], group.security_id.to_numpy()[known], residual))


def make_book(group, weights):
    columns = ['security_id','ticker','signal_date','entry_date','exit_date','entry_price',
               'entry_contract_ok','forward_return','return_observed','PV_AVG_DOLLAR_VOLUME_21D']
    b = group[columns].copy().set_index('security_id', drop=False)
    b['requested_weight'] = weights
    b['side'] = np.where(weights>=0, 1, -1)
    b['entry_executable'] = np.isfinite(b.entry_price) & (b.entry_price>0) & b.entry_contract_ok.eq(True)
    b['availability_u'] = 1.0
    return b


def stats(rows, rf):
    return cash_stats(rows['return'], rf, rows.entry_date, rows.exit_date)


def summarize(monthly, rf):
    results = []
    for key, part in monthly.groupby(['model','signal','scenario','missing_policy'], sort=False):
        for period, subset in [('full',part),('early',part[part.signal_date<'2017-01-01']),
                               ('late',part[part.signal_date>='2017-01-01'])]:
            rates = subset.signal_date.map(rf)
            mean, t = base.hac_mean(subset['return'])
            results.append(dict(zip(['model','signal','scenario','missing_policy'],key),period=period,
                months=len(subset),mean_hac_t=t,**stats(subset,rates),
                turnover=subset.turnover_full_notional.mean(),
                max_abs_weight=subset.max_abs_weight.mean() if 'max_abs_weight' in subset else np.nan))
    return pd.DataFrame(results)


def add_gross2_baskets(monthly, data):
    """Scale the netted basket itself to gross 2, using formation weights only."""
    monthly=monthly[monthly.signal.ne('basket3_gross2')].copy()
    targets=pd.read_parquet(OUT/'weights.parquet',columns=['security_id','signal_date','model','gross2_basket3'])
    source={d:g.sort_values('security_id') for d,g in data.groupby('signal_date')}
    model_books={}
    for model,frame in targets.groupby('model'):
        sequence=[]
        for date,g in frame.groupby('signal_date'):
            group=source[date]
            g=g.sort_values('security_id')
            assert np.array_equal(g.security_id,group.security_id)
            w=g.gross2_basket3.to_numpy()
            sequence.append(make_book(group,w*2/np.abs(w).sum()))
        model_books[model]=sequence
    sequence=[]
    for date,group in source.items():
        parts=[]
        for name in PROFIT:
            r=(group[name].rank(method='average').to_numpy()-.5)/len(group)
            long,short=r>=.8,r<.2
            parts.append(long/long.sum()-short/short.sum())
        w=np.mean(parts,axis=0)
        sequence.append(make_book(group,w*2/np.abs(w).sum()))
    model_books['sort_common']=sequence
    extra=[]
    for model,books in model_books.items():
        for scenario in [dict(name='gross',trading_bps=0,borrow_rate=0,unavailable_short_fraction=0,short_adv_fraction_cap=None),
                         dict(name='cost10_borrow3',trading_bps=10,borrow_rate=.03,unavailable_short_fraction=0,short_adv_fraction_cap=None)]:
            for adverse in [False,True]:
                result,_=accounting.simulate(books,scenario,dict(initial_nav=CFG['initial_nav']),adverse,record_holdings=False)
                result['model'],result['signal']=model,'basket3_gross2'
                result['scenario']=scenario['name']
                result['missing_policy']='adverse_unknown' if adverse else 'flat_unknown'
                extra.append(result)
        base.log('Completed gross-2 net basket: '+model)
    return pd.concat([monthly,*extra],ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_data = load_inputs()
    required = [*PROFIT,*CONTROLS,'residual_variance_proxy']
    valid = np.isfinite(all_data[required]).all(axis=1)
    data = all_data[valid].sort_values(['signal_date','security_id']).copy()
    rf_table = pd.read_csv(ROOT/'results/factor_portfolio/risk_free_windows.csv',
                           parse_dates=['signal_date','entry_date','exit_date']).set_index('signal_date')
    rf = rf_table.risk_free
    calendar = rf_table.reset_index()
    coverage = all_data.groupby('signal_date').size().rename('eligible_n').to_frame()
    coverage['joint_complete_n'] = data.groupby('signal_date').size()
    for name in required:
        coverage['finite_'+name] = all_data.groupby('signal_date')[name].apply(lambda a:np.isfinite(a).sum())
    coverage.to_csv(OUT/'coverage.csv')
    rows, coefficients, diagnostics = [], [], []
    weight_writer = design_writer = None
    account_cfg = dict(initial_nav=CFG['initial_nav'])
    scenarios = [dict(name='gross',trading_bps=0,borrow_rate=0,unavailable_short_fraction=0,short_adv_fraction_cap=None),
                 dict(name='cost10_borrow3',trading_bps=10,borrow_rate=.03,unavailable_short_fraction=0,short_adv_fraction_cap=None)]
    try:
        for method in CFG['transformations']:
            risk = SpecificRisk()
            books = {weighting:{name:[] for name in [*PROFIT,'basket3']} for weighting in CFG['weighting']}
            sorts = {name:[] for name in [*PROFIT,'basket3']}
            for date, group in data.groupby('signal_date', sort=True):
                assert len(group)>=CFG['minimum_stocks']
                original = rf_table.loc[date]
                assert group.entry_date.eq(original.entry_date).all() and group.exit_date.eq(original.exit_date).all()
                x = design(group, method)
                known = group.return_observed.to_numpy() & np.isfinite(group.forward_return.to_numpy())
                underlying = np.where(known, group.forward_return.to_numpy(), 0.0)
                y = underlying-rf.loc[date]
                cap_w = np.sqrt(group.FUND_MARKET_CAP_USD_COMPAT.to_numpy())
                cap_w /= cap_w.mean()
                inverse_w, ready, variance, last_exit = risk.forecast(group)
                d = pd.DataFrame(x, columns=FACTORS)
                d.insert(0,'security_id',group.security_id.to_numpy())
                d.insert(0,'signal_date',date)
                d['transformation']=method
                d['sqrt_cap_weight']=cap_w
                d['inverse_specific_weight']=inverse_w
                d['specific_variance']=variance
                d['risk_history_ready']=ready
                d['risk_latest_exit']=last_exit
                table=pa.Table.from_pandas(d,preserve_index=False)
                if design_writer is None: design_writer=pq.ParquetWriter(OUT/'design.parquet',table.schema)
                design_writer.write_table(table)
                for weighting, w in [('sqrt_cap',cap_w),('inverse_specific_variance',inverse_w)]:
                    model=method+'__'+weighting
                    coef, h_all, condition = fit_fmp(x,w,y)
                    h=h_all[:,1:4]
                    error = np.max(np.abs(x.T@h-np.eye(len(FACTORS))[:,1:4]))
                    assert error<1e-8
                    gross = np.abs(h).sum(axis=0)
                    normalized = h*CFG['portfolio_gross']/gross
                    weight_record = group[['security_id','signal_date']].reset_index(drop=True).copy()
                    weight_record['model']=model
                    for j,name in enumerate(PROFIT):
                        weight_record['raw_'+name]=h[:,j]
                        weight_record['gross2_'+name]=normalized[:,j]
                        books[weighting][name].append(make_book(group,normalized[:,j]))
                        partial = x[:,j+1]-x[:,np.arange(x.shape[1])!=j+1]@np.linalg.lstsq(
                            x[:,np.arange(x.shape[1])!=j+1]*np.sqrt(w[:,None]),
                            x[:,j+1]*np.sqrt(w),rcond=None)[0]
                        partial_sd=np.sqrt(np.average(partial**2,weights=w))
                        diagnostics.append(dict(model=model,signal_date=date,signal=name,n=len(group),
                            observed_n=int(known.sum()),design_condition=condition,max_identity_error=error,
                            raw_gross=gross[j],gross2_factor_exposure=2/gross[j],
                            max_abs_weight=np.abs(normalized[:,j]).max(),
                            effective_positions=4/np.sum(normalized[:,j]**2),
                            partial_exposure_sd=partial_sd,risk_history_ready_fraction=ready.mean(),
                            risk_latest_exit=last_exit,
                            unknown_abs_weight=np.abs(normalized[~known,j]).sum(),
                            ideal_gross2_return=normalized[:,j]@underlying))
                    weight_record['gross2_basket3']=normalized.mean(axis=1)
                    books[weighting]['basket3'].append(make_book(group,normalized.mean(axis=1)))
                    table=pa.Table.from_pandas(weight_record,preserve_index=False)
                    if weight_writer is None: weight_writer=pq.ParquetWriter(OUT/'weights.parquet',table.schema)
                    weight_writer.write_table(table)
                    for j,name in enumerate(FACTORS):
                        coefficients.append(dict(model=model,signal_date=date,factor=name,factor_return=coef[j],
                            raw_portfolio_return=h_all[:,j]@underlying,
                            unit_exposure_error=error,n=len(group),unknown_n=int((~known).sum())))
                risk.enqueue(group,x,y,cap_w)
                if method==CFG['transformations'][0]:
                    sort_weights=[]
                    for name in PROFIT:
                        ranks=(group[name].rank(method='average').to_numpy()-.5)/len(group)
                        long,short=ranks>=.8,ranks<.2
                        weights=long/long.sum()-short/short.sum()
                        sorts[name].append(make_book(group,weights))
                        sort_weights.append(weights)
                    sorts['basket3'].append(make_book(group,np.mean(sort_weights,axis=0)))
            if method==CFG['transformations'][0]: books['sort_common']=sorts
            for weighting, component_books in books.items():
                model='sort_common' if weighting=='sort_common' else method+'__'+weighting
                for signal, sequence in component_books.items():
                    for scenario in scenarios:
                        for adverse in [False,True]:
                            returns, _ = accounting.simulate(sequence,scenario,account_cfg,adverse,record_holdings=False)
                            returns['model'],returns['signal']=model,signal
                            returns['scenario']=scenario['name']
                            returns['missing_policy']='adverse_unknown' if adverse else 'flat_unknown'
                            rows.append(returns)
                base.log(f'Completed {model}: {len(sequence)} formations')
    finally:
        if weight_writer is not None: weight_writer.close()
        if design_writer is not None: design_writer.close()
    monthly=add_gross2_baskets(pd.concat(rows,ignore_index=True),data)
    monthly.to_csv(OUT/'monthly_returns.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'factor_returns.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    summary=summarize(monthly,rf)
    summary.to_csv(OUT/'summary.csv',index=False)
    inputs=[CACHE,base.CACHE/'monthly_panel.parquet',ROOT/'results/factor_portfolio/risk_free_windows.csv',
            ROOT/'results/regimes/regime_features.csv',CONFIG_PATH]
    manifest=dict(config=CFG,code_sha256=sha(__file__),accounting_code_sha256=sha(accounting.__file__),inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],
        sources=[base.fingerprint(base.PRICES)],
        response='monthly underlying adjusted-price return minus exact-window RF; unknown underlying return set to zero',
        weight_timing='X and both W are formed using information available by signal close; risk residuals enter only once exit_date <= signal_date',
        identity='raw FMP excess-return PnL equals WLS coefficient; gross2 scaling changes the target exposure',
        normalization='each profitability FMP gross=2; basket3 averages their targets and nets overlapping positions; basket3_gross2 additionally rescales that net basket to gross 2',
        accounting='existing monthly long-short engine; entry rejections can change exact neutrality; active cash, collateral and funding zero',
        limitations=['no sector neutrality','no complete commercial Barra risk model','EWMA risk uses proxy for short histories',
                    'historical accounting revisions and estimated publication dates','short availability unobserved',
                    'unknown outcomes: flat mark/liquidation and directional sensitivity','chronological split is not untouched OOS'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(summary[(summary.period=='full')&(summary.scenario=='gross')&(summary.missing_policy=='flat_unknown')]
          [['model','signal','cagr','annual_vol','sharpe_rf','monthly_max_drawdown']].to_string(index=False),flush=True)


if __name__=='__main__':
    if '--rescale-only' in sys.argv:
        data=load_inputs()
        valid=np.isfinite(data[[*PROFIT,*CONTROLS,'residual_variance_proxy']]).all(axis=1)
        monthly=pd.read_csv(OUT/'monthly_returns.csv',parse_dates=['signal_date','entry_date','exit_date'])
        monthly=add_gross2_baskets(monthly,data[valid])
        monthly.to_csv(OUT/'monthly_returns.csv',index=False)
        rf=pd.read_csv(ROOT/'results/factor_portfolio/risk_free_windows.csv',parse_dates=['signal_date']).set_index('signal_date').risk_free
        summarize(monthly,rf).to_csv(OUT/'summary.csv',index=False)
        manifest=json.loads((OUT/'manifest.json').read_text())
        manifest.update(code_sha256=sha(__file__),accounting_code_sha256=sha(accounting.__file__),
            normalization='each profitability FMP gross=2; basket3 averages their targets and nets overlapping positions; basket3_gross2 additionally rescales that net basket to gross 2')
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    else:
        main()
