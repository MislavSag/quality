"""IC and monthly portfolio statistics for fixed quality dimension composites."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import analyze_quality as base
import backtest_monthly_long_short as accounting
import build_quality_dimensions as dimensions
from build_interactive_report import REGIMES, cash_stats, risk_free_windows

ROOT = base.ROOT
OUT = dimensions.OUT
CONFIG_PATH = ROOT / 'config/quality_dimension_statistics.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
SIGNALS = list(dimensions.SPECS)
SCORES = OUT / 'monthly_dimension_scores.parquet'
DOLLARS = ['nav_start', 'nav_after_entry_cost', 'nav_end', 'gross_pnl', 'long_pnl',
           'short_pnl', 'entry_trading_cost', 'final_liquidation_cost', 'borrow_cost', 'traded_notional']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs():
    stamp = json.loads(dimensions.CACHE.with_suffix('.json').read_text())
    assert stamp == dimensions.input_contract(), 'Quality indicator input cache is stale'
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['output_sha256'][SCORES.name] == sha(SCORES), 'Quality scores differ from their manifest'
    fields = ['security_id', 'ticker', 'signal_date', 'entry_date', 'exit_date', 'eligible',
              'entry_price', 'entry_contract_ok', 'forward_return', 'return_observed', 'PV_AVG_DOLLAR_VOLUME_21D']
    data = pd.read_parquet(dimensions.MONTHLY, columns=fields)
    scores = pd.read_parquet(SCORES, columns=dimensions.KEYS + ['eligible'] + [s + '_signal' for s in SIGNALS])
    assert np.array_equal(data[dimensions.KEYS], scores[dimensions.KEYS])
    np.testing.assert_array_equal(data.eligible, scores.eligible)
    for signal in SIGNALS:
        data[signal] = scores[signal + '_signal'].to_numpy()
    calendar = data[['signal_date', 'entry_date', 'exit_date']].drop_duplicates().sort_values('signal_date').reset_index(drop=True)
    rf = risk_free_windows(calendar)
    rf.to_csv(OUT / 'statistics_risk_free_windows.csv', index=False)
    calendar = calendar.merge(rf, on=['signal_date', 'entry_date', 'exit_date'], validate='one_to_one')
    previous = pd.read_csv(ROOT / 'results/monthly_long_short/monthly_returns.csv', parse_dates=['signal_date'])
    benchmark = previous.groupby('signal_date').spy_return.agg(['first', 'min', 'max'])
    np.testing.assert_allclose(benchmark['min'], benchmark['max'])
    calendar['spy_return'] = calendar.signal_date.map(benchmark['first'])
    assert np.isfinite(calendar[['spy_return', 'risk_free']]).all().all()
    return data[data.eligible].copy(), calendar


def period_mask(dates, period):
    if period == 'full':
        return np.ones(len(dates), dtype=bool)
    if period == 'early':
        return dates < pd.Timestamp('2017-01-01')
    if period == 'late':
        return dates >= pd.Timestamp('2017-01-01')
    if period == 'common_history':
        return dates >= pd.Timestamp('2012-01-01')
    raise ValueError(period)


def ic_statistics(data):
    records = []
    for signal in SIGNALS:
        for date, group in data[np.isfinite(data[signal])].groupby('signal_date', sort=True):
            known = group.return_observed & np.isfinite(group.forward_return)
            observed = group[known]
            spearman = observed[signal].corr(observed.forward_return, method='spearman') if len(observed) >= 100 else np.nan
            records.append({'signal_date': date, 'signal': signal, 'n': len(group),
                'n_observed': int(known.sum()), 'missing_fraction': 1 - known.mean(),
                'ic_spearman': spearman, **base.cosine_metrics(group, signal)})
    monthly = pd.DataFrame(records)
    monthly.to_csv(OUT / 'monthly_ic.csv', index=False)
    summary = []
    for signal, group in monthly.groupby('signal', sort=False):
        for period in ['full', 'common_history', 'early', 'late']:
            selected = group[period_mask(group.signal_date, period)]
            row = {'signal': signal, 'period': period, 'months': len(selected),
                   'mean_n': selected.n.mean(), 'mean_n_observed': selected.n_observed.mean()}
            for metric in ['ic_spearman', 'ic_cosine_raw', 'ic_cosine_rank_centered']:
                row['mean_' + metric], row[metric + '_hac_t'] = base.hac_mean(selected[metric], CONFIG['hac_lags'])
            summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / 'ic_summary.csv', index=False)
    return monthly


def align_calendar(rows, calendar):
    """Pre-indicator RF cash and homogeneous scaling of the active book ledger."""
    complete = calendar.merge(rows, on=['signal_date', 'entry_date', 'exit_date'], how='left', validate='one_to_one')
    complete['indicator_available'] = complete['return'].notna()
    active = complete.indicator_available.to_numpy()
    first = int(np.flatnonzero(active)[0])
    assert active[first:].all(), 'Internal indicator gaps require explicit liquidation/re-entry accounting'
    prefix_scale = float((1 + complete.risk_free.iloc[:first]).prod())
    for column in DOLLARS:
        complete.loc[active, column] *= prefix_scale
    cash_nav = CONFIG['initial_nav']
    for i in range(first):
        for column in rows.columns:
            if column not in ['signal_date', 'entry_date', 'exit_date']:
                complete.loc[i, column] = 0
        complete.loc[i, 'return'] = complete.loc[i, 'risk_free']
        complete.loc[i, ['nav_start', 'nav_after_entry_cost']] = cash_nav
        cash_nav *= 1 + complete.loc[i, 'risk_free']
        complete.loc[i, 'nav_end'] = cash_nav
    complete['cash_interest'] = np.where(active, 0, complete.nav_end - complete.nav_start)
    complete['cash_reason'] = np.where(active, '', 'indicator_unavailable')
    np.testing.assert_allclose(complete.nav_end / complete.nav_start - 1, complete['return'], atol=1e-12)
    np.testing.assert_allclose(complete.nav_start.iloc[1:], complete.nav_end.iloc[:-1], rtol=1e-10)
    return complete


def portfolio_statistics(data, calendar):
    all_rows, formation = [], []
    for signal in SIGNALS:
        candidate = data[np.isfinite(data[signal])].copy()
        dates = candidate.groupby('signal_date').size()
        assert (dates >= 100).all()
        books = accounting.formation_books(candidate, signal, CONFIG)
        for book in books:
            saved = book[['security_id', 'ticker', 'signal_date', 'entry_date', 'exit_date',
                'requested_weight', 'side', 'entry_executable', 'forward_return', 'return_observed']].copy()
            saved['signal'] = signal
            formation.append(saved.reset_index(drop=True))
        for leg in CONFIG['legs']:
            sequence = books if leg == 'long_short' else [book[book.side.eq(1 if leg == 'long' else -1)].copy() for book in books]
            config = {**CONFIG, 'balance_sides': leg == 'long_short'}
            for scenario in CONFIG['scenarios']:
                for policy in CONFIG['missing_policies']:
                    rows, _ = accounting.simulate(sequence, scenario, config,
                        adverse_missing=policy == 'adverse_unknown', record_holdings=False)
                    aligned = align_calendar(rows, calendar)
                    aligned['signal'], aligned['leg'] = signal, leg
                    aligned['scenario'], aligned['missing_policy'] = scenario['name'], policy
                    all_rows.append(aligned)
        print('Completed quality portfolio statistics:', signal, flush=True)
    holdings = pd.concat(formation, ignore_index=True)
    holdings.to_parquet(OUT / 'statistics_formation_holdings.parquet', index=False)
    monthly = pd.concat(all_rows, ignore_index=True)
    monthly.to_csv(OUT / 'statistics_monthly_returns.csv', index=False)
    return monthly


def summarize(monthly):
    records = []
    keys = ['signal', 'leg', 'scenario', 'missing_policy']
    for key, group in monthly.groupby(keys, sort=False):
        for period in ['full', 'available_history', 'common_history', 'early', 'late']:
            mask = group.indicator_available if period == 'available_history' else period_mask(group.signal_date, period)
            part = group.loc[mask]
            row = dict(zip(keys, key), period=period)
            row.update(cash_stats(part['return'], part.risk_free, part.entry_date, part.exit_date))
            active = part[part.indicator_available]
            fit = sm.OLS(part['return'] - part.risk_free, sm.add_constant(part.spy_return - part.risk_free)).fit(
                cov_type='HAC', cov_kwds={'maxlags': CONFIG['hac_lags']})
            _, excess_t = base.hac_mean(part['return'] - part.risk_free, CONFIG['hac_lags'])
            row.update(months=len(part), active_months=len(active), cash_months=len(part) - len(active),
                first_entry=str(part.entry_date.iloc[0].date()), last_exit=str(part.exit_date.iloc[-1].date()),
                first_active_signal_date=str(active.signal_date.iloc[0].date()),
                positive_month_fraction=float((part['return'] > 0).mean()), excess_mean_hac_t=excess_t,
                spy_beta=float(fit.params.iloc[1]), spy_alpha_annual=float(fit.params.iloc[0] * 12),
                spy_alpha_hac_t=float(fit.tvalues.iloc[0]),
                mean_turnover_active=active.turnover_full_notional.mean(),
                mean_long_n_active=active.long_n.mean(), mean_short_n_active=active.short_n.mean(),
                mean_unknown_long_weight_active=active.unknown_long_weight.mean(),
                mean_unknown_short_weight_active=active.unknown_short_weight.mean(),
                entry_price_rejections=int(active.entry_price_rejected_n.sum()))
            records.append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / 'statistics_summary.csv', index=False)
    return summary


def regime_statistics(monthly):
    features = pd.read_csv(ROOT / 'results/regimes/regime_features.csv', parse_dates=['signal_date'])
    records = []
    keys = ['signal', 'leg', 'scenario', 'missing_policy']
    for key, group in monthly.groupby(keys, sort=False):
        group = group.merge(features[['signal_date', 'common_sample'] + [r['id'] + '_state' for r in REGIMES]],
                            on='signal_date', validate='one_to_one')
        for regime in [None, *REGIMES]:
            for level in ([None] if regime is None else [0, 1]):
                gate = np.ones(len(group), dtype=bool) if regime is None else (group.common_sample & group[regime['id'] + '_state'].eq(level))
                active = gate & group.indicator_available
                values = group['return'].where(active, group.risk_free)
                records.append(dict(zip(keys, key), regime='all' if regime is None else regime['id'], state=level,
                    active_months=int(active.sum()), cash_months=int((~active).sum()),
                    **cash_stats(values, group.risk_free, group.entry_date, group.exit_date)))
    pd.DataFrame(records).to_csv(OUT / 'statistics_regime_cash_summary.csv', index=False)


def main():
    data, calendar = read_inputs()
    ic_statistics(data)
    monthly = portfolio_statistics(data, calendar)
    summary = summarize(monthly)
    regime_statistics(monthly)
    sources = [SCORES, dimensions.MONTHLY, CONFIG_PATH, ROOT / 'results/regimes/regime_features.csv',
               ROOT / 'data/reference/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip']
    manifest = {'date': '2026-11-12', 'config': CONFIG, 'sources': {str(p): sha(p) for p in sources},
        'code_sha256': sha(__file__), 'accounting_sha256': sha(accounting.__file__),
        'rows': len(monthly), 'formation_months': len(calendar), 'signals': SIGNALS,
        'scope': 'monthly quintile stock portfolios; not joint quality FMP regression',
        'comparison': 'each signal uses its own complete-case universe, so return differences also reflect universe differences',
        'funding': 'RF only when indicator or chosen regime is unavailable; active portfolios earn no cash/collateral interest',
        'history': 'full calendar plus available-history and common-start summaries; growth first 60 months explicitly RF',
        'regime_costs': CONFIG['regime_cost_policy'],
        'outputs': {p.name: sha(p) for p in OUT.glob('statistics_*') if p.is_file() and p.name != 'statistics_manifest.json'}}
    (OUT / 'statistics_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary[(summary.period == 'available_history') & (summary.leg == 'long_short') &
        (summary.scenario == 'gross_unconstrained') & (summary.missing_policy == 'flat_unknown')]
        [['signal', 'months', 'cagr', 'annual_vol', 'sharpe_rf']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
