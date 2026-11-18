"""Build fixed, stock-level quality dimension scores on the formation panel.

No future returns enter this pipeline. All source Parquets remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_quality as base

ROOT = base.ROOT
OUT = ROOT / 'results/quality_dimensions'
CONFIG_PATH = ROOT / 'config/quality_dimensions.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
SPECS = {**CONFIG['dimensions'], **CONFIG['comparators']}
GROWTH_BASE = ['roa_ttm', 'roe_ttm', 'gp_assets_ttm', 'cfo_assets_ttm', 'gross_margin_ttm']
FZ = base.SOURCE / 'factor_zoo_fundamental_extensions.parquet'
MONTHLY = base.CACHE / 'monthly_panel.parquet'
CACHE = base.CACHE / 'quality_dimension_inputs.parquet'
AUDIT_FIELDS = ['FUND_WORKING_CAPITAL_ACCRUALS_YOY', 'FUND_CFO_CONVERSION_TTM',
                'FUND_DEBT_PAYDOWN_YIELD_TTM', 'FUND_STMT_TOTAL_ASSETS']
KEYS = ['security_id', 'signal_date']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions():
    return [(dimension, name, direction) for dimension, spec in SPECS.items()
            for name, direction in spec['components'].items()]


def input_contract():
    return {'version': 1, 'monthly_sha256': sha(MONTHLY),
            'sources': [base.fingerprint(p) for p in [base.FUND, FZ]],
            'code_sha256': sha(__file__), 'config_sha256': sha(CONFIG_PATH)}


def add_growth_history(frame):
    """Exact calendar-month lookup; never a 60-row or forward-filled lag."""
    frame = frame.copy()
    periods = frame.signal_date.dt.year.to_numpy() * 12 + frame.signal_date.dt.month.to_numpy()
    key = pd.MultiIndex.from_arrays([frame.security_id, periods], names=['security_id', 'month'])
    assert key.is_unique
    history = frame[GROWTH_BASE + ['signal_date']].copy()
    history.index = key
    lookup = pd.MultiIndex.from_arrays([frame.security_id, periods - 60], names=key.names)
    past = history.reindex(lookup).reset_index(drop=True)
    frame['growth_lag_date'] = past.signal_date.to_numpy()
    for name in GROWTH_BASE:
        frame['LAG60_' + name] = past[name].to_numpy()
        frame['DELTA60_' + name] = frame[name].to_numpy() - past[name].to_numpy()
    known = frame.growth_lag_date.notna()
    assert (frame.loc[known, 'growth_lag_date'] < frame.loc[known, 'signal_date']).all()
    return frame


def load_inputs():
    contract = input_contract()
    stamp = CACHE.with_suffix('.json')
    if CACHE.exists() and stamp.exists() and json.loads(stamp.read_text()) == contract:
        base.log('Reusing verified formation-characteristic cache')
        return pd.read_parquet(CACHE), contract
    prior = json.loads((base.CACHE / 'contract.json').read_text())
    assert prior['inputs'] == [base.fingerprint(p) for p in [base.PRICES, base.FUND, base.EXTFILE]], 'Monthly source panel is stale'
    monthly_columns = list(dict.fromkeys([*KEYS, 'ticker', 'eligible', 'FUND_STATEMENT_AVAILABLE_DATE',
        'fcf_assets_ttm', 'cfo_assets_ttm', 'gp_assets_ttm', 'ff2015_fy', *GROWTH_BASE]))
    frame = pd.read_parquet(MONTHLY, columns=monthly_columns)
    dates = sorted(frame.signal_date.dt.date.unique())
    fundamental = sorted({name for _, name, _ in definitions() if name.startswith('FUND_')} | set(AUDIT_FIELDS))
    factor_zoo = sorted({name for _, name, _ in definitions() if name.startswith('FZ_')})
    for source, fields in [(base.FUND, fundamental), (FZ, factor_zoo)]:
        base.log('Reading month-end inputs from ' + source.name)
        values = base.selected_scan(source, ['security_id', 'trade_date', *fields], dates)
        values = values.to_pandas(date_as_object=False).rename(columns={'trade_date': 'signal_date'})
        assert not values.duplicated(KEYS).any()
        frame = frame.merge(values, on=KEYS, how='left', validate='one_to_one', sort=False)
    numeric = frame.select_dtypes(include=['number']).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    # Guard economically invalid signs; negative earnings/interest coverage remain valid.
    positive_assets = frame.FUND_STMT_TOTAL_ASSETS > 0
    extra_fields = fundamental + factor_zoo
    asset_scaled = [s for s in extra_fields if s not in AUDIT_FIELDS and
                    (s.endswith('_AT') or s.endswith('_ASSETS'))]
    frame.loc[~positive_assets, asset_scaled] = np.nan
    for name in ['FUND_DEBT_TO_ASSETS', 'FUND_REFINANCING_PRESSURE_PROXY',
                 'FUND_DIVIDEND_YIELD_TTM', 'FZ_EARNINGS_VOL_20Q_AT',
                 'FZ_CFO_VOL_20Q_AT', 'FZ_SALES_VOL_20Q_AT']:
        frame.loc[frame[name] < 0, name] = np.nan
    frame.loc[frame.FUND_SHARES_GROWTH_YOY <= -1, 'FUND_SHARES_GROWTH_YOY'] = np.nan
    frame = add_growth_history(frame)
    assert not frame.duplicated(KEYS).any()
    assert (frame.loc[frame.eligible, 'FUND_STATEMENT_AVAILABLE_DATE'] <= frame.loc[frame.eligible, 'signal_date']).all()
    frame.to_parquet(CACHE, index=False)
    stamp.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return frame, contract


def build_scores(frame):
    """Only formation-time eligibility and component values determine scores."""
    result = frame[[*KEYS, 'ticker', 'eligible']].copy()
    ranks_out = frame[KEYS].copy()
    coverage = []
    component_coverage = []
    for dimension, spec in SPECS.items():
        fields = list(spec['components'])
        directions = np.array(list(spec['components'].values()), dtype=float)
        values = frame[fields].to_numpy(dtype=float)
        finite = np.isfinite(values)
        count = finite.sum(axis=1)
        complete = frame.eligible.to_numpy() & finite.all(axis=1)
        result[dimension + '_component_count'] = count.astype('int8')
        score = np.full(len(frame), np.nan)
        internal = np.full(len(frame), np.nan)
        all_ranks = np.full(values.shape, np.nan)
        for date, index in frame.groupby('signal_date', sort=True).indices.items():
            chosen = index[complete[index]]
            if len(chosen) >= CONFIG['minimum_cross_section']:
                adjusted = pd.DataFrame(values[chosen] * directions, columns=fields)
                percentiles = (adjusted.rank(method='average') - 0.5) / len(chosen)
                all_ranks[chosen] = percentiles.to_numpy()
                # Preserve the exact old bundle arithmetic, not just its ordering.
                equal_weights = np.repeat(1 / len(fields), len(fields))
                if dimension == 'profitability_broad':
                    internal[chosen] = 2 * percentiles.mean(axis=1).to_numpy() - 1
                    score[chosen] = 50 * (internal[chosen] + 1)
                elif dimension == 'profitability':
                    internal[chosen] = 2 * (percentiles.to_numpy() @ equal_weights) - 1
                    score[chosen] = 50 * (internal[chosen] + 1)
                else:
                    score[chosen] = 100 * (percentiles.to_numpy() @ equal_weights)
                    internal[chosen] = score[chosen] / 50 - 1
            eligible = frame.eligible.to_numpy()[index]
            coverage.append({'signal_date': date, 'dimension': dimension,
                'eligible_n': int(eligible.sum()), 'complete_n': len(chosen),
                'scored_n': int(np.isfinite(score[index]).sum()),
                'required_components': len(fields)})
            for j, field in enumerate(fields):
                component_coverage.append({'signal_date': date, 'dimension': dimension,
                    'component': field, 'direction': int(directions[j]),
                    'eligible_n': int(eligible.sum()),
                    'finite_n': int((eligible & finite[index, j]).sum())})
        result[dimension + '_score'] = score
        # Keep the legacy profitability arithmetic before the score round-trip:
        # ~1e-16 differences can otherwise split ties at a quintile boundary.
        result[dimension + '_signal'] = internal
        for j, field in enumerate(fields):
            ranks_out['rank__' + dimension + '__' + field] = all_ranks[:, j]
        base.log(f'{dimension}: {int(np.isfinite(score).sum()):,} scored stock-months')
    cover = pd.DataFrame(coverage)
    cover['coverage'] = cover.scored_n / cover.eligible_n.replace(0, np.nan)
    return result, ranks_out, cover, pd.DataFrame(component_coverage)


def summarize_coverage(coverage):
    rows = []
    for dimension, group in coverage.groupby('dimension', sort=False):
        active = group[group.scored_n > 0]
        latest = group.iloc[-1]
        rows.append({'dimension': dimension, 'label': SPECS[dimension]['label'],
            'first_scored_date': active.signal_date.min(), 'scored_months': len(active),
            'mean_scored_n': group.scored_n.mean(),
            'mean_scored_n_active_months': active.scored_n.mean(),
            'latest_date': latest.signal_date, 'latest_scored_n': int(latest.scored_n),
            'latest_eligible_n': int(latest.eligible_n), 'latest_coverage': latest.coverage,
            'required_components': len(SPECS[dimension]['components'])})
    return pd.DataFrame(rows)


def component_metadata():
    formulas = {
        'FUND_CASHFLOW_ACCRUALS_TTM': '(NI_TTM - CFO_TTM) / average assets',
        'FZ_OPERATING_ACCRUALS_AT': '(delta noncash current assets - delta operating current liabilities - depreciation_TTM) / average assets',
        'FUND_NET_OPERATING_ASSETS_TO_ASSETS': 'net operating assets / total assets',
        'FUND_DEBT_TO_ASSETS': 'total debt / total assets',
        'FUND_INTEREST_COVERAGE_TTM': 'EBIT_TTM / abs(interest expense_TTM)',
        'FUND_REFINANCING_PRESSURE_PROXY': 'short-term debt / (cash + max(CFO_TTM, 0)), positive denominator',
        'FUND_ASSET_GROWTH_YOY': 'assets / assets_lag4Q - 1',
        'FZ_PPE_INVENTORY_GROWTH_AT': '(delta PPE_net + delta inventory) / assets_lag4Q',
        'FUND_DIVIDEND_YIELD_TTM': 'abs(dividends_paid_TTM) / current USD-compatible market cap',
        'FUND_NET_EQUITY_ISSUANCE_YIELD_TTM': 'net stock sale/purchase_TTM / current USD-compatible market cap; negative denotes repurchases',
        'FUND_SHARES_GROWTH_YOY': 'reported shares / shares for the same reporting source one fiscal year earlier - 1',
        'fcf_assets_ttm': 'FCF_TTM / average assets',
        'cfo_assets_ttm': 'CFO_TTM / average assets',
        'gp_assets_ttm': 'gross profit_TTM / average assets',
        'ff2015_fy': 'existing annual FF2015 OP/book-equity proxy, July aligned'
    }
    rows = []
    for dimension, field, direction in definitions():
        if field.startswith('DELTA60_'):
            formula = field[8:] + '(t) - ' + field[8:] + '(t - 60 calendar months)'
            source = 'monthly formation characteristic history'
        elif field.startswith('FZ_'):
            source = str(FZ)
            formula = formulas.get(field, 'sd of quarterly ' + field.split('_')[1] + ' over 20 contiguous fiscal quarters, min 12 finite, divided by current average assets')
        else:
            source = str(base.EXTFILE if field == 'ff2015_fy' else base.FUND)
            formula = formulas[field]
        rows.append({'dimension': dimension, 'component': field, 'direction': direction,
            'weight': 1 / len(SPECS[dimension]['components']), 'formula': formula,
            'source': source, 'availability': 'source as-of availability; next-session execution if later backtested',
            'status': SPECS[dimension]['status']})
    return pd.DataFrame(rows)


def correlations(scores):
    fields = [name + '_score' for name in CONFIG['dimensions']]
    tables = []
    for date, group in scores.groupby('signal_date', sort=True):
        matrix = group[fields].corr(method='spearman', min_periods=CONFIG['minimum_cross_section'])
        matrix.index.name = 'dimension_a'
        long = matrix.reset_index().melt(id_vars='dimension_a', var_name='dimension_b', value_name='spearman')
        long['signal_date'] = date
        tables.append(long)
    all_months = pd.concat(tables, ignore_index=True)
    all_months.to_csv(OUT / 'monthly_score_correlations.csv', index=False)
    return all_months.groupby(['dimension_a', 'dimension_b']).spearman.agg(['mean', 'count']).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.refresh and CACHE.with_suffix('.json').exists():
        CACHE.with_suffix('.json').unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, contract = load_inputs()
    scores, ranks, coverage, component_cover = build_scores(frame)
    scores.to_parquet(OUT / 'monthly_dimension_scores.parquet', index=False)
    ranks.to_parquet(OUT / 'component_ranks.parquet', index=False)
    coverage.to_csv(OUT / 'coverage.csv', index=False)
    component_cover.to_csv(OUT / 'component_coverage.csv', index=False)
    summary = summarize_coverage(coverage)
    summary.to_csv(OUT / 'dimension_summary.csv', index=False)
    metadata = component_metadata()
    metadata.to_csv(OUT / 'component_metadata.csv', index=False)
    latest = scores[scores.eligible & scores.signal_date.eq(scores.signal_date.max())].copy()
    latest.to_csv(OUT / 'latest_scores.csv', index=False)
    correlations(scores).to_csv(OUT / 'score_correlations_summary.csv', index=False)
    yearly = []
    for year, group in frame[frame.eligible].groupby(frame.signal_date.dt.year):
        for field in AUDIT_FIELDS:
            yearly.append({'year': int(year), 'component': field, 'eligible_n': len(group),
                'finite_n': int(np.isfinite(group[field]).sum()), 'coverage': np.isfinite(group[field]).mean()})
    pd.DataFrame(yearly).to_csv(OUT / 'audited_extra_input_coverage.csv', index=False)
    manifest = {'date': '2026-11-12', 'config': CONFIG, 'input_contract': contract,
        'rows': len(scores), 'formation_months': int(scores.signal_date.nunique()),
        'primary_dimensions': list(CONFIG['dimensions']), 'comparators': list(CONFIG['comparators']),
        'first_date': str(scores.signal_date.min()), 'last_date': str(scores.signal_date.max()),
        'scope': 'stock-level monthly indicators; no new backtest or fitted weights',
        'history': '60-calendar-month lookup of the same security; unavailable prior history stays missing',
        'limitations': ['current vendor histories may contain revisions; estimated release dates remain',
            'complete-case universes differ across dimensions; scores are relative to their own universe',
            'no historical sector neutralization; financials and real estate need special treatment for replication',
            'growth is ratio-change proxy, not residual per-share QMJ growth',
            'payout yields include valuation; share-count growth can reflect vendor basis changes and splits',
            'all listed components required; limited interest/dividend coverage can exclude otherwise valid firms'],
        'source_code_review': {name: sha(Path('C:/Users/Mislav/projects_r/characteristics') / name)
            for name in ['build_fundamental_characteristics.R', 'build_factor_zoo_fundamental_extensions.R',
                         'build_fundamental_base.R', 'statement_pit_core.R']},
        'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.suffix in ['.csv', '.parquet']}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    from build_quality_dimensions_report import build_report
    build_report()
    base.log(summary[['dimension', 'first_scored_date', 'latest_scored_n', 'latest_coverage']].to_string(index=False))


if __name__ == '__main__':
    main()
