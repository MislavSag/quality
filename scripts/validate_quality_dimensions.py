"""Independent ranks, historical joins and statement-level checks of quality scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import build_quality_dimensions as builder

ROOT = builder.ROOT
OUT = builder.OUT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, label):
    assert np.isfinite(actual) == np.isfinite(expected), (label, actual, expected)
    if np.isfinite(expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11, err_msg=str(label))


def statement_checks(frame):
    """Reconstruct selected inputs only from statements released by each date."""
    statements = pd.read_parquet(builder.base.SOURCE / 'fundamental_statements_base.parquet')
    statements = statements[statements.FUND_STATEMENT_PERIOD.eq('quarterly')].copy()
    statements['FUND_AVAILABLE_DATE'] = pd.to_datetime(statements.FUND_AVAILABLE_DATE)
    statements['FUND_FISCAL_DATE'] = pd.to_datetime(statements.FUND_FISCAL_DATE)
    cap = pd.read_parquet(builder.MONTHLY, columns=builder.KEYS + ['FUND_MARKET_CAP_USD_COMPAT'])
    cap = cap.set_index(builder.KEYS).FUND_MARKET_CAP_USD_COMPAT
    dates = pd.to_datetime(['2007-01-31', '2012-01-31', '2020-06-30', frame.signal_date.max()])
    chosen = frame[frame.ticker.isin(['AAPL', 'GOOGL', 'NVDA', 'WMT', 'JPM']) & frame.signal_date.isin(dates)]
    checks = []

    def first(row, *names):
        return next((row[name] for name in names if np.isfinite(row[name])), np.nan)

    def add(*values):
        return sum(values) if np.isfinite(values).all() else np.nan

    def divide(numerator, denominator):
        return numerator / denominator if np.isfinite([numerator, denominator]).all() and denominator != 0 else np.nan

    for _, observation in chosen.iterrows():
        sid, date = observation.security_id, observation.signal_date
        history = statements[statements.security_id.eq(sid) & statements.FUND_AVAILABLE_DATE.le(date)]
        history = history.sort_values(['FUND_FISCAL_DATE', 'FUND_AVAILABLE_DATE']).drop_duplicates('FUND_FISCAL_DATE', keep='last')
        if len(history) < 5:
            continue
        current, lag = history.iloc[-1], history.iloc[-5]
        gaps = history.FUND_FISCAL_DATE.diff().dt.days.iloc[1:].to_numpy()
        steps = np.where((gaps >= 70) & (gaps <= 112), 1, np.maximum(2, np.round(gaps / 91.3125)))
        index = np.r_[0, np.cumsum(steps)]
        if index[-1] - index[-5] != 4 or index[-1] - index[-4] != 3:
            continue

        def ttm(name):
            values = history[name].iloc[-4:].to_numpy(dtype=float)
            return values.sum() if np.isfinite(values).all() else np.nan

        def cash(row):
            return first(row, 'FUND_BS_CASH_AND_SHORT_TERM_INVESTMENTS', 'FUND_BS_CASH_AND_EQUIVALENTS', 'FUND_BS_CASH')

        def short_debt(row):
            return first(row, 'FUND_BS_SHORT_TERM_DEBT', 'FUND_BS_SHORT_LONG_TERM_DEBT')

        def total_debt(row):
            candidates = [row.FUND_BS_SHORT_LONG_TERM_DEBT_TOTAL,
                add(row.FUND_BS_SHORT_TERM_DEBT, row.FUND_BS_LONG_TERM_DEBT),
                add(row.FUND_BS_SHORT_LONG_TERM_DEBT, row.FUND_BS_LONG_TERM_DEBT_TOTAL),
                row.FUND_BS_LONG_TERM_DEBT_TOTAL]
            return next((value for value in candidates if np.isfinite(value)), np.nan)

        assets = current.FUND_BS_TOTAL_ASSETS
        prior_assets = lag.FUND_BS_TOTAL_ASSETS
        average_assets = (assets + prior_assets) / 2
        ni, cfo = ttm('FUND_IS_NET_INCOME'), ttm('FUND_CF_OPERATING_CASH_FLOW')
        ca, cl = current.FUND_BS_TOTAL_CURRENT_ASSETS, current.FUND_BS_TOTAL_CURRENT_LIABILITIES
        old_ca, old_cl = lag.FUND_BS_TOTAL_CURRENT_ASSETS, lag.FUND_BS_TOTAL_CURRENT_LIABILITIES
        changes = ((ca - cash(current)) - (old_ca - cash(lag))) - ((cl - short_debt(current)) - (old_cl - short_debt(lag)))
        market_cap = cap.loc[(sid, date)]
        expected = {
            'FUND_CASHFLOW_ACCRUALS_TTM': divide(ni - cfo, average_assets),
            'FZ_OPERATING_ACCRUALS_AT': divide(changes - ttm('FUND_CF_DEPRECIATION'), average_assets),
            'FUND_NET_OPERATING_ASSETS_TO_ASSETS': divide(assets - cash(current) - current.FUND_BS_TOTAL_LIABILITIES + total_debt(current), assets),
            'FUND_DEBT_TO_ASSETS': divide(total_debt(current), assets),
            'FUND_INTEREST_COVERAGE_TTM': divide(ttm('FUND_IS_EBIT'), abs(ttm('FUND_IS_INTEREST_EXPENSE'))),
            'FUND_REFINANCING_PRESSURE_PROXY': divide(short_debt(current), cash(current) + max(cfo, 0)),
            'FUND_ASSET_GROWTH_YOY': divide(assets, prior_assets) - 1,
            'FZ_PPE_INVENTORY_GROWTH_AT': divide(current.FUND_BS_PROPERTY_PLANT_EQUIPMENT_NET - lag.FUND_BS_PROPERTY_PLANT_EQUIPMENT_NET + current.FUND_BS_INVENTORY - lag.FUND_BS_INVENTORY, prior_assets),
            'FUND_DIVIDEND_YIELD_TTM': divide(abs(ttm('FUND_CF_DIVIDENDS_PAID')), market_cap),
            'FUND_NET_EQUITY_ISSUANCE_YIELD_TTM': divide(ttm('FUND_CF_SALE_PURCHASE_OF_STOCK'), market_cap)
        }
        if len(history) >= 20 and index[-1] - index[-20] == 19:
            for field, raw in [('FZ_EARNINGS_VOL_20Q_AT', 'FUND_IS_NET_INCOME'),
                               ('FZ_CFO_VOL_20Q_AT', 'FUND_CF_OPERATING_CASH_FLOW'),
                               ('FZ_SALES_VOL_20Q_AT', 'FUND_IS_TOTAL_REVENUE')]:
                values = history[raw].iloc[-20:].dropna()
                expected[field] = values.std(ddof=1) / average_assets if len(values) >= 12 else np.nan
        for field, value in expected.items():
            compare(observation[field], value, (observation.ticker, str(date.date()), field))
            checks.append({'ticker': observation.ticker, 'security_id': sid, 'signal_date': date,
                'component': field, 'actual': observation[field], 'independent_value': value,
                'last_statement_available': history.FUND_AVAILABLE_DATE.max(),
                'last_fiscal_date': current.FUND_FISCAL_DATE})
    assert len(checks) >= 150, len(checks)
    result = pd.DataFrame(checks)
    result.to_csv(OUT / 'statement_spot_checks.csv', index=False)
    return len(result)


def validate_history(frame):
    key = frame.assign(month=frame.signal_date.dt.to_period('M')).set_index(['security_id', 'month'])
    rows = frame.growth_lag_date.notna()
    assert (frame.loc[rows, 'signal_date'].dt.to_period('M').astype('int64').to_numpy() -
            frame.loc[rows, 'growth_lag_date'].dt.to_period('M').astype('int64').to_numpy() == 60).all()
    dates = sorted(frame.signal_date.unique())
    for date in [dates[60], dates[len(dates) // 2], dates[-1]]:
        group = frame[frame.signal_date.eq(date)]
        for _, row in group.sample(min(60, len(group)), random_state=20261112).iterrows():
            history_key = (row.security_id, row.signal_date.to_period('M') - 60)
            for field in builder.GROWTH_BASE:
                prior = key.loc[history_key, field] if history_key in key.index else np.nan
                compare(row['DELTA60_' + field], row[field] - prior, (history_key, field))
    # An adversarial sparse history must not turn 60 observations into 60 months.
    sparse_dates = pd.date_range('2000-01-31', periods=63, freq='ME').delete(1)
    synthetic = pd.DataFrame({'security_id': 'TEST', 'signal_date': sparse_dates,
                             **{field: np.arange(len(sparse_dates), dtype=float) for field in builder.GROWTH_BASE}})
    derived = builder.add_growth_history(synthetic)
    missing_month = derived.signal_date.dt.to_period('M').eq(pd.Period('2005-02', freq='M'))
    assert derived.loc[missing_month, 'DELTA60_roa_ttm'].isna().all()
    before = derived.iloc[:-1].copy()
    synthetic.loc[synthetic.index[-1], builder.GROWTH_BASE] = 1e9
    after = builder.add_growth_history(synthetic).iloc[:-1]
    pd.testing.assert_frame_equal(before, after)


def main():
    frame = pd.read_parquet(builder.CACHE)
    scores = pd.read_parquet(OUT / 'monthly_dimension_scores.parquet')
    ranks = pd.read_parquet(OUT / 'component_ranks.parquet')
    pd.testing.assert_frame_equal(frame[builder.KEYS], scores[builder.KEYS])
    pd.testing.assert_frame_equal(frame[builder.KEYS], ranks[builder.KEYS])
    assert not scores.duplicated(builder.KEYS).any()
    checked_scores = 0
    checked_component_ranks = 0
    max_error = 0.0
    for dimension, spec in builder.SPECS.items():
        fields = list(spec['components'])
        values = frame[fields].to_numpy(dtype=float)
        finite = np.isfinite(values)
        np.testing.assert_array_equal(scores[dimension + '_component_count'], finite.sum(axis=1))
        valid = frame.eligible.to_numpy() & finite.all(axis=1)
        expected = np.full(len(frame), np.nan)
        for _, index in frame.groupby('signal_date', sort=True).indices.items():
            selected = index[valid[index]]
            if len(selected) < builder.CONFIG['minimum_cross_section']:
                continue
            percentiles = []
            for j, (field, direction) in enumerate(spec['components'].items()):
                r = (rankdata(direction * values[selected, j], method='average') - 0.5) / len(selected)
                actual = ranks.loc[selected, 'rank__' + dimension + '__' + field].to_numpy()
                np.testing.assert_allclose(actual, r, atol=1e-13, rtol=0)
                percentiles.append(r)
                checked_component_ranks += len(r)
            expected[selected] = 100 * np.mean(percentiles, axis=0)
        actual = scores[dimension + '_score'].to_numpy()
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        known = np.isfinite(expected)
        np.testing.assert_allclose(actual[known], expected[known], atol=1e-12, rtol=0)
        np.testing.assert_allclose(scores.loc[known, dimension + '_signal'], expected[known] / 50 - 1, atol=1e-13)
        assert ((actual[known] > 0) & (actual[known] < 100)).all()
        max_error = max(max_error, float(np.max(np.abs(actual[known] - expected[known]))))
        checked_scores += int(known.sum())
    original = pd.read_parquet(ROOT / 'results/profitability_bundle/monthly_bundle_scores.parquet')
    comparison = scores.merge(original[builder.KEYS + ['bundle4_score', 'bundle3_score']], on=builder.KEYS, validate='one_to_one')
    for name, old in [('profitability_score', 'bundle4_score'), ('profitability_broad_score', 'bundle3_score')]:
        np.testing.assert_allclose(comparison[name], comparison[old], atol=1e-12, equal_nan=True)
    for name, old in [('profitability_signal', 'bundle4'), ('profitability_broad_signal', 'bundle3')]:
        np.testing.assert_array_equal(scores[name], original[old])
    assert scores.loc[scores.signal_date < pd.Timestamp('2012-01-01'), 'quality_growth_score'].isna().all()
    assert (frame.loc[frame.eligible, 'FUND_STATEMENT_AVAILABLE_DATE'] <= frame.loc[frame.eligible, 'signal_date']).all()
    validate_history(frame)
    statement_count = statement_checks(frame)
    baseline = json.loads((ROOT / 'data/reference/quality_rename_20261112/numeric_outputs_before.json').read_text())
    assert all(digest(ROOT / name) == value for name, value in baseline.items())
    contract = json.loads(builder.CACHE.with_suffix('.json').read_text())
    assert contract == builder.input_contract()
    page = (OUT / 'report.html').read_text(encoding='utf-8')
    for href in re.findall(r'href="([^"]+)"', page):
        if not href.startswith(('http', '#', 'data:')):
            assert (OUT / href.split('#')[0]).exists(), href
    assert '\ufffd' not in page
    receipt = {'status': 'passed', 'rows': len(scores), 'primary_dimensions': len(builder.CONFIG['dimensions']),
        'formation_months': int(scores.signal_date.nunique()), 'independently_checked_scores': checked_scores,
        'independently_checked_component_ranks': checked_component_ranks, 'max_score_error': max_error,
        'independent_statement_checks': statement_count, 'unchanged_old_numeric_outputs': len(baseline),
        'original_profitability_reproduced': True, 'exact_calendar_growth_lag': True,
        'future_history_perturbation_passed': True, 'all_missing_and_universe_masks_checked': True,
        'future_returns_used_in_construction': False}
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
