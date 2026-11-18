"""Independent formation, IC, trading ledger, RF and summary verification."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import rankdata

import analyze_quality_dimensions as analysis
import build_quality_dimensions as dimensions

ROOT, OUT = analysis.ROOT, analysis.OUT
KEYS = ['signal', 'leg', 'scenario', 'missing_policy']


def check_close(actual, expected, label, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=1e-8, equal_nan=True, err_msg=str(label))


def independent_metrics(values, rates, entry, exit):
    values, rates = np.asarray(values), np.asarray(rates)
    equity = np.concatenate(([1.0], np.cumprod(1 + values)))
    years = (exit.iloc[-1] - entry.iloc[0]).days / 365.25
    excess = values - rates
    volatility = np.std(excess, ddof=1) * np.sqrt(12)
    return {'cagr': equity[-1] ** (1 / years) - 1, 'annual_vol': np.std(values, ddof=1) * np.sqrt(12),
        'sharpe_rf': np.mean(excess) * 12 / volatility if volatility > 1e-12 else np.nan,
        'monthly_max_drawdown': np.min(equity / np.maximum.accumulate(equity) - 1)}


def verify_formation_and_ic(holdings):
    scores = pd.read_parquet(analysis.SCORES)
    future = pd.read_parquet(dimensions.MONTHLY, columns=dimensions.KEYS + ['return_observed', 'forward_return'])
    data = scores.merge(future, on=dimensions.KEYS, validate='one_to_one')
    ic = pd.read_csv(OUT / 'monthly_ic.csv', parse_dates=['signal_date']).set_index(['signal', 'signal_date'])
    targets = {(signal, date): group for (signal, date), group in holdings.groupby(['signal', 'signal_date'], sort=False)}
    count = 0
    for signal in analysis.SIGNALS:
        field = signal + '_signal'
        for date, group in data[data.eligible & np.isfinite(data[field])].groupby('signal_date', sort=True):
            score = group[field].to_numpy()
            percentile = (rankdata(score, method='average') - 0.5) / len(group)
            expected_ids = set(group.loc[(percentile < 0.2) | (percentile >= 0.8), 'security_id'])
            selected = targets[signal, date]
            assert set(selected.security_id) == expected_ids, (signal, date)
            for sign in [1, -1]:
                part = selected[selected.side.eq(sign)]
                check_close(part.requested_weight, sign / len(part), (signal, date, 'target weights'))
            known = group.return_observed.to_numpy() & np.isfinite(group.forward_return.to_numpy())
            response = group.forward_return.to_numpy()[known]
            x = score[known]
            spearman = np.corrcoef(rankdata(x), rankdata(response))[0, 1] if len(x) >= 100 else np.nan
            def cosine(a, b):
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                return np.dot(a, b) / norm if norm > 0 else np.nan
            centered = rankdata(score) - rankdata(score).mean()
            row = ic.loc[(signal, date)]
            check_close(row.ic_spearman, spearman, 'Spearman')
            check_close(row.ic_cosine_raw, cosine(x, response), 'internal cosine')
            check_close(row.ic_cosine_rank_centered, cosine(centered[known], response), 'rank cosine')
            assert row.n == len(group) and row.n_observed == known.sum()
            count += 1
    return count


def verify_ledgers(holdings, monthly):
    ids = {sid: i for i, sid in enumerate(sorted(holdings.security_id.unique()))}
    groups = {signal: list(group.groupby('signal_date', sort=True)) for signal, group in holdings.groupby('signal')}
    observed = monthly.set_index(KEYS + ['signal_date'])
    calendar = monthly[['signal_date', 'entry_date', 'exit_date', 'risk_free']].drop_duplicates().sort_values('signal_date').reset_index(drop=True)
    rates = calendar.set_index('signal_date').risk_free
    encoded = {}
    for signal, sequence in groups.items():
        encoded[signal] = []
        for date, group in sequence:
            encoded[signal].append((date, group, np.array([ids[sid] for sid in group.security_id])))
    count, cash_count, max_error = 0, 0, 0.0
    for signal, sequence in encoded.items():
        prefix = calendar[calendar.signal_date < sequence[0][0]]
        for leg in analysis.CONFIG['legs']:
            for scenario in analysis.CONFIG['scenarios']:
                cost, borrow = scenario['trading_bps'] / 10000, scenario['borrow_rate']
                for policy in analysis.CONFIG['missing_policies']:
                    nav = float(analysis.CONFIG['initial_nav'])
                    old = np.zeros(len(ids))
                    for cash in prefix.itertuples():
                        row = observed.loc[(signal, leg, scenario['name'], policy, cash.signal_date)]
                        assert not row.indicator_available
                        check_close(row['return'], cash.risk_free, 'RF before indicator')
                        check_close(row.nav_start, nav, 'cash NAV', 1e-6)
                        nav *= 1 + cash.risk_free
                        check_close(row.nav_end, nav, 'cash ending NAV', 1e-6)
                        assert row.entry_trading_cost == row.borrow_cost == row.long_n == row.short_n == 0
                        cash_count += 1
                    for i, (date, group, ix) in enumerate(sequence):
                        select = np.ones(len(group), dtype=bool) if leg == 'long_short' else group.side.to_numpy() == (1 if leg == 'long' else -1)
                        raw = np.where(select & group.entry_executable.to_numpy(), group.requested_weight, 0.0)
                        if leg == 'long_short':
                            accepted = min(np.sum(raw[raw > 0]), -np.sum(raw[raw < 0]))
                            for sign in [1, -1]:
                                side = raw * sign > 0
                                total = np.abs(raw[side]).sum()
                                if total:
                                    raw[side] *= accepted / total
                        target = np.zeros(len(ids))
                        target[ix] = raw
                        before = nav
                        after = brentq(lambda capital: capital + cost * np.abs(target * capital - old).sum() - before,
                                       0, before, xtol=1e-7) if cost else before
                        traded = np.abs(target * after - old).sum()
                        known = group.return_observed.to_numpy() & np.isfinite(group.forward_return.to_numpy())
                        response = np.where(known, group.forward_return, 0.0)
                        if policy == 'adverse_unknown':
                            response[~known & group.side.eq(1)] = -1
                            response[~known & group.side.eq(-1)] = 1
                        returns = np.zeros(len(ids))
                        returns[ix] = response
                        positions = target * after
                        end = positions * (1 + returns)
                        days = (group.exit_date.iloc[0] - group.entry_date.iloc[0]).days
                        fee = -(np.minimum(positions, 0).sum() + np.minimum(end, 0).sum()) * 0.5 * borrow * days / 365
                        pnl = np.dot(positions, returns)
                        liquidation = cost * np.abs(end).sum() if i == len(sequence) - 1 else 0
                        nav = after + pnl - fee - liquidation
                        row = observed.loc[(signal, leg, scenario['name'], policy, date)]
                        assert row.indicator_available
                        expected = {'return': nav / before - 1, 'nav_start': before, 'nav_end': nav,
                            'entry_trading_cost': cost * traded, 'borrow_cost': fee,
                            'final_liquidation_cost': liquidation, 'gross_pnl': pnl}
                        for key, value in expected.items():
                            check_close(row[key], value, (signal, leg, scenario['name'], policy, date, key), 1e-6 if key != 'return' else 1e-10)
                        max_error = max(max_error, abs(row['return'] - expected['return']))
                        if leg == 'long':
                            assert row.short_gross_weight == row.borrow_cost == 0
                        if leg == 'short':
                            assert row.long_gross_weight == 0
                        old = end
                        count += 1
    return count, cash_count, max_error


def verify_summaries(monthly):
    summary = pd.read_csv(OUT / 'statistics_summary.csv').set_index(KEYS + ['period'])
    count = 0
    for key, group in monthly.groupby(KEYS, sort=False):
        for period in ['full', 'available_history', 'common_history', 'early', 'late']:
            if period == 'full':
                part = group
            elif period == 'available_history':
                part = group[group.indicator_available]
            elif period == 'common_history':
                part = group[group.signal_date >= '2012-01-01']
            elif period == 'early':
                part = group[group.signal_date < '2017-01-01']
            else:
                part = group[group.signal_date >= '2017-01-01']
            expected = independent_metrics(part['return'], part.risk_free, part.entry_date, part.exit_date)
            row = summary.loc[(*key, period)]
            assert row.months == len(part) and row.active_months == part.indicator_available.sum()
            for metric, value in expected.items():
                check_close(row[metric], value, (key, period, metric))
            count += 1
    return count


def verify_regimes(monthly):
    features = pd.read_csv(ROOT / 'results/regimes/regime_features.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'])
    summary = pd.read_csv(OUT / 'statistics_regime_cash_summary.csv')
    count = 0
    for key, group in monthly.groupby(KEYS, sort=False):
        group = group.merge(features, on=['signal_date', 'entry_date', 'exit_date'], validate='one_to_one')
        subset = summary
        for name, value in zip(KEYS, key):
            subset = subset[subset[name].eq(value)]
        for _, row in subset.iterrows():
            gate = np.ones(len(group), dtype=bool) if row.regime == 'all' else group.common_sample & group[row.regime + '_state'].eq(row.state)
            active = gate & group.indicator_available
            assert row.active_months == active.sum() and row.cash_months == (~active).sum()
            result = np.where(active, group['return'], group.risk_free)
            for metric, value in independent_metrics(result, group.risk_free, group.entry_date, group.exit_date).items():
                check_close(row[metric], value, (key, row.regime, row.state, metric))
            count += 1
    return count


def main():
    holdings = pd.read_parquet(OUT / 'statistics_formation_holdings.parquet')
    monthly = pd.read_csv(OUT / 'statistics_monthly_returns.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    assert not monthly.duplicated(KEYS + ['signal_date']).any()
    assert len(monthly) == 8 * 3 * 3 * 2 * 235
    original = pd.read_csv(ROOT / 'results/profitability_bundle/monthly_returns.csv', parse_dates=['signal_date'])
    comparisons = 0
    for signal, previous in [('profitability', 'bundle4'), ('profitability_broad', 'bundle3')]:
        left = monthly[monthly.signal.eq(signal) & monthly.leg.eq('long_short')]
        right = original[original.signal.eq(previous)]
        merged = left.merge(right, on=['signal_date', 'scenario', 'missing_policy'], suffixes=('_new', '_old'), validate='one_to_one')
        assert len(merged) == 3 * 2 * 235
        check_close(merged.return_new, merged.return_old, 'existing profitability return reproduction')
        comparisons += len(merged)
    ic_count = verify_formation_and_ic(holdings)
    ledgers, cash_months, max_error = verify_ledgers(holdings, monthly)
    summaries = verify_summaries(monthly)
    regimes = verify_regimes(monthly)
    score_manifest = json.loads((OUT / 'manifest.json').read_text())
    assert score_manifest['output_sha256'][analysis.SCORES.name] == analysis.sha(analysis.SCORES)
    before = json.loads((ROOT / 'data/reference/quality_rename_20261112/numeric_outputs_before.json').read_text())
    assert all(analysis.sha(ROOT / name) == digest for name, digest in before.items())
    for page in [OUT / 'report.html', *OUT.glob('statistics*.html')]:
        text = page.read_text(encoding='utf-8')
        assert '\ufffd' not in text
        for href in re.findall(r'href="([^"]+)"', text):
            if not href.startswith(('http', '#', 'data:')):
                assert (page.parent / href.split('#')[0]).exists(), (page, href)
    receipt = {'status': 'passed', 'monthly_rows': len(monthly), 'independent_active_ledgers': ledgers,
        'independent_pre_indicator_rf_months': cash_months, 'max_return_error': max_error,
        'independent_monthly_ic_cross_sections': ic_count, 'independent_portfolio_summaries': summaries,
        'independent_regime_summaries': regimes, 'existing_profitability_return_cells_reproduced': comparisons,
        'original_numeric_outputs_unchanged': len(before), 'formation_membership_verified': True,
        'borrow_only_on_short': True, 'growth_cash_prefix_verified': True,
        'independent_cost_solver': 'Brent root, not production fixed-point iteration'}
    (OUT / 'statistics_validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
