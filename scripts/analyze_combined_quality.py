"""Fixed ALL7/CORE5 quality scores, matched sorts and joint-regression FMPs.

Formation weights depend only on characteristics. Source and older numeric
results are read-only. Standard accounting/statistics reuse the existing engine.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtri

import analyze_joint_fmp as joint
import analyze_quality_dimensions as statistics
import backtest_monthly_long_short as accounting
import build_quality_dimensions as dimensions
import build_iwv_universe as universe

ROOT = dimensions.ROOT
OUT = ROOT / 'results/combined_quality'
CONFIG_PATH = ROOT / 'config/combined_quality.json'
CFG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
CONFIG = json.loads((ROOT / 'config/quality_dimension_statistics.json').read_text(encoding='utf-8'))
KEYS = dimensions.KEYS
SPECS = {
    'sort_all7': {'label': 'ALL7 · kvintili', 'bundle': 'all7', 'universe': 'all7', 'kind': 'sort'},
    'sort_core5': {'label': 'CORE5 · kvintili', 'bundle': 'core5', 'universe': 'core5', 'kind': 'sort'},
    'sort_core5_matched': {'label': 'CORE5 · iste dionice kao ALL7', 'bundle': 'core5', 'universe': 'all7', 'kind': 'sort'},
    'sort_all7_common': {'label': 'ALL7 · kvintili na FMP uzorku', 'bundle': 'all7', 'universe': 'all7_common', 'kind': 'sort'},
    'sort_core5_common': {'label': 'CORE5 · kvintili na FMP uzorku', 'bundle': 'core5', 'universe': 'core5_common', 'kind': 'sort'},
    'sort_core5_matched_common': {'label': 'CORE5 · zajednički ALL7 FMP uzorak', 'bundle': 'core5', 'universe': 'all7_common', 'kind': 'sort'},
    'fmp_all7': {'label': 'ALL7 · FMP', 'bundle': 'all7', 'universe': 'all7_common', 'kind': 'fmp', 'controls': 'base'},
    'fmp_core5': {'label': 'CORE5 · FMP', 'bundle': 'core5', 'universe': 'core5_common', 'kind': 'fmp', 'controls': 'base'},
    'fmp_core5_matched': {'label': 'CORE5 · FMP na ALL7 uzorku', 'bundle': 'core5', 'universe': 'all7_common', 'kind': 'fmp', 'controls': 'base'},
    'fmp_all7_defensive': {'label': 'ALL7 · FMP uz beta/vol kontrole', 'bundle': 'all7', 'universe': 'all7_common', 'kind': 'fmp', 'controls': 'defensive'},
    'fmp_core5_defensive': {'label': 'CORE5 · FMP uz beta/vol kontrole', 'bundle': 'core5', 'universe': 'core5_common', 'kind': 'fmp', 'controls': 'defensive'},
    'fmp_core5_matched_defensive': {'label': 'CORE5 · FMP ALL7 uzorak, beta/vol', 'bundle': 'core5', 'universe': 'all7_common', 'kind': 'fmp', 'controls': 'defensive'}
}
SIGNALS = list(SPECS)
SORT_SIGNALS = [s for s, spec in SPECS.items() if spec['kind'] == 'sort']


def sha(path):
    return dimensions.sha(path)


def fields_for(bundle):
    return list(dict.fromkeys(field for dim in CFG[bundle] for field in dimensions.SPECS[dim]['components']))


def dimension_exposures(group, bundle):
    """Recompute every constituent rank on precisely this formation universe."""
    exposures, component_ranks = {}, {}
    for name in CFG[bundle]:
        spec = dimensions.SPECS[name]['components']
        adjusted = group[list(spec)].to_numpy(float) * np.array(list(spec.values()))
        ranks = (pd.DataFrame(adjusted).rank(method='average').to_numpy() - .5) / len(group)
        average = ranks.mean(axis=1)
        z = ndtri((pd.Series(average).rank(method='average').to_numpy() - .5) / len(group))
        scale = z.std(ddof=0)
        assert scale > 1e-12
        exposures[name] = (z - z.mean()) / scale
        for j, field in enumerate(spec):
            component_ranks[name + '__' + field] = ranks[:, j]
    return pd.DataFrame(exposures), pd.DataFrame(component_ranks)


def combined(group, bundle):
    x, ranks = dimension_exposures(group, bundle)
    signal = x.mean(axis=1).to_numpy()
    score = 100 * (pd.Series(signal).rank(method='average').to_numpy() - .5) / len(group)
    return signal, score, x, ranks


def read_inputs():
    formation, _ = dimensions.load_inputs()
    formation = formation[formation.eligible].copy()
    extra = joint.load_inputs()
    columns = ['entry_date', 'exit_date', 'entry_price', 'entry_contract_ok', 'forward_return',
        'return_observed', 'PV_AVG_DOLLAR_VOLUME_21D', 'FUND_MARKET_CAP_USD_COMPAT', *CFG['controls_defensive']]
    data = formation.merge(extra[KEYS + columns], on=KEYS, how='left', validate='one_to_one')
    membership = universe.load_membership()
    data['signal_date'] = data.signal_date.astype('datetime64[ns]')
    data = data.merge(membership[KEYS + ['iwv_member', 'iwv_eligible']], on=KEYS, how='left', validate='one_to_one')
    assert data.iwv_member.notna().all()
    data = data[data.iwv_eligible].copy()
    assert data.entry_date.notna().all() and data.FUND_MARKET_CAP_USD_COMPAT.gt(0).all()
    assert not data.duplicated(KEYS).any()
    calendar = pd.read_csv(dimensions.OUT / 'statistics_risk_free_windows.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'])
    old = pd.read_csv(dimensions.OUT / 'statistics_monthly_returns.csv', usecols=['signal_date', 'spy_return'], parse_dates=['signal_date'])
    benchmark = old.groupby('signal_date').spy_return.first()
    calendar['spy_return'] = calendar.signal_date.map(benchmark)
    assert np.isfinite(calendar[['risk_free', 'spy_return']]).all().all()
    return data.sort_values(['signal_date', 'security_id']).reset_index(drop=True), calendar


def eligible_universes(data):
    valid = {bundle: np.isfinite(data[fields_for(bundle)]).all(axis=1) for bundle in ['all7', 'core5']}
    controls = np.isfinite(data[CFG['controls_defensive']]).all(axis=1)
    valid.update({bundle + '_common': valid[bundle] & controls for bundle in ['all7', 'core5']})
    return valid


def scores_and_books(data, calendar):
    valid = eligible_universes(data)
    scores = data[KEYS + ['ticker', 'eligible']].copy()
    for signal in SIGNALS:
        scores[signal + '_signal'] = np.nan
        scores[signal + '_score'] = np.nan
    books = {s: [] for s in SIGNALS}
    exposure_records, rank_records, weight_records, design_records = [], [], [], []
    diagnostics, coefficients, coverage = [], [], []
    rates = calendar.set_index('signal_date').risk_free
    index_by_date = data.groupby('signal_date', sort=True).indices
    for date in calendar.signal_date:
        date_indices = index_by_date.get(date, np.array([], dtype=int))
        for universe, mask in valid.items():
            index = date_indices[mask.to_numpy()[date_indices]]
            coverage.append(dict(signal_date=date, universe=universe, eligible_n=len(date_indices), complete_n=len(index),
                scored_n=len(index) if len(index) >= CFG['minimum_stocks'] else 0))
            if len(index) < CFG['minimum_stocks']:
                continue
            group = data.iloc[index].copy()
            for bundle in ['all7', 'core5']:
                selected = {s: spec for s, spec in SPECS.items() if spec['universe'] == universe and spec['bundle'] == bundle}
                if not selected:
                    continue
                z, percentile, exposures, ranks = combined(group, bundle)
                meta = group[KEYS].reset_index(drop=True)
                e = pd.concat([meta, exposures], axis=1)
                e['universe'], e['bundle'] = universe, bundle
                exposure_records.append(e)
                r = pd.concat([meta, ranks], axis=1)
                r['universe'], r['bundle'] = universe, bundle
                rank_records.append(r)
                for signal, spec in selected.items():
                    scores.loc[index, signal + '_signal'] = z
                    scores.loc[index, signal + '_score'] = percentile
                    if spec['kind'] == 'sort':
                        candidate = group.copy()
                        candidate[signal] = z
                        book = accounting.formation_books(candidate, signal, CONFIG)[0]
                    else:
                        controls = CFG['controls_' + spec['controls']]
                        cap = group.FUND_MARKET_CAP_USD_COMPAT.to_numpy()
                        x = np.column_stack([np.ones(len(group)), exposures.to_numpy(), *[
                            joint.transform(group[c].to_numpy(), cap, 'rank_normal') for c in controls]])
                        w = np.sqrt(cap)
                        # The operator and weights are determined BEFORE reading future returns.
                        _, h, condition = joint.fit_fmp(x, w, np.zeros(len(group)))
                        target = np.r_[0.0, np.repeat(1 / len(exposures.columns), len(exposures.columns)), np.zeros(len(controls))]
                        raw = h @ target
                        identity = x.T @ raw
                        error = float(np.max(np.abs(identity - target)))
                        assert error < 1e-8
                        scale = CFG['portfolio_gross'] / np.abs(raw).sum()
                        weights = raw * scale
                        np.testing.assert_allclose([weights.sum(), np.abs(weights).sum()], [0, 2], atol=1e-9)
                        book = joint.make_book(group, weights)
                        saved = meta.copy()
                        saved['signal'] = signal
                        saved['raw_combined_weight'] = raw
                        saved['requested_weight'] = weights
                        for j, name in enumerate(exposures):
                            saved['raw__' + name] = h[:, j + 1]
                        weight_records.append(saved)
                        design = pd.DataFrame(x, columns=['market', *exposures.columns, *controls])
                        design = pd.concat([meta, design], axis=1)
                        design['signal'], design['sqrt_cap_weight'] = signal, w / w.mean()
                        design_records.append(design)
                        known = group.return_observed.to_numpy() & np.isfinite(group.forward_return.to_numpy())
                        underlying = np.where(known, group.forward_return.to_numpy(), 0.0)
                        coef = h.T @ (underlying - rates.loc[date])
                        for j, name in enumerate(['market', *exposures.columns, *controls]):
                            coefficients.append(dict(signal=signal, signal_date=date, factor=name, coefficient=coef[j],
                                n=len(group), observed_n=int(known.sum())))
                        actual = weights * book.entry_executable.to_numpy()
                        accepted = min(actual[actual > 0].sum(), -actual[actual < 0].sum())
                        for sign in [1, -1]:
                            side = actual * sign > 0
                            if side.any():
                                actual[side] *= accepted / np.abs(actual[side]).sum()
                        diagnostics.append(dict(signal=signal, signal_date=date, n=len(group), design_condition=condition,
                            raw_identity_error=error, aggregate_raw_gross=np.abs(raw).sum(), aggregate_scale=scale,
                            dimension_budget_before_scaling=1 / len(exposures.columns),
                            max_abs_weight=np.abs(weights).max(), effective_positions=4 / (weights @ weights),
                            unknown_abs_weight=np.abs(weights[~known]).sum(), entry_rejections=int((~book.entry_executable).sum()),
                            max_control_exposure_after_entry=float(np.max(np.abs((x.T @ actual)[1 + len(exposures.columns):]))),
                            raw_quality_return=float(raw @ underlying), coefficient_combination=float(target @ coef)))
                    books[signal].append(book)
        if date.month == 12:
            print('Built combined quality formations through', date.date(), flush=True)
    scores.to_parquet(OUT / 'monthly_combined_scores.parquet', index=False)
    pd.concat(exposure_records, ignore_index=True).to_parquet(OUT / 'dimension_exposures.parquet', index=False)
    pd.concat(rank_records, ignore_index=True).to_parquet(OUT / 'component_ranks.parquet', index=False)
    pd.concat(weight_records, ignore_index=True).to_parquet(OUT / 'fmp_weights.parquet', index=False)
    pd.concat(design_records, ignore_index=True).to_parquet(OUT / 'fmp_design.parquet', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / 'fmp_diagnostics.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'factor_returns.csv', index=False)
    latest = scores[scores.signal_date.eq(scores.signal_date.max())].copy()
    latest.to_csv(OUT / 'latest_scores.csv', index=False)
    return scores, books


def configure_statistics():
    statistics.OUT, statistics.CONFIG = OUT, CONFIG
    statistics.SIGNALS = SORT_SIGNALS


def portfolio_statistics(books, calendar):
    all_rows, formation = [], []
    for signal, sequence in books.items():
        assert len(sequence) > 0, signal
        for book in sequence:
            saved = book[['security_id', 'ticker', 'signal_date', 'entry_date', 'exit_date',
                'requested_weight', 'side', 'entry_executable', 'forward_return', 'return_observed']].copy()
            saved['signal'] = signal
            formation.append(saved.reset_index(drop=True))
        for leg in CONFIG['legs']:
            if leg == 'long_short':
                selected = sequence
            else:
                selected = []
                sign = 1 if leg == 'long' else -1
                for book in sequence:
                    part = book[book.side.eq(sign)].copy()
                    # Full single-side notional even for numerical FMP asymmetry.
                    part['requested_weight'] *= 1 / part.requested_weight.abs().sum()
                    selected.append(part)
            cfg = {**CONFIG, 'balance_sides': leg == 'long_short'}
            for scenario in CONFIG['scenarios']:
                for policy in CONFIG['missing_policies']:
                    rows, _ = accounting.simulate(selected, scenario, cfg, policy == 'adverse_unknown', record_holdings=False)
                    aligned = statistics.align_calendar(rows, calendar)
                    aligned['signal'], aligned['leg'] = signal, leg
                    aligned['scenario'], aligned['missing_policy'] = scenario['name'], policy
                    all_rows.append(aligned)
        print('Completed combined quality portfolios:', signal, flush=True)
    pd.concat(formation, ignore_index=True).to_parquet(OUT / 'statistics_formation_holdings.parquet', index=False)
    monthly = pd.concat(all_rows, ignore_index=True)
    monthly.to_csv(OUT / 'statistics_monthly_returns.csv', index=False)
    return monthly


def matched_comparisons(monthly):
    pairs = [('all7_vs_core5_same_stocks', 'sort_all7', 'sort_core5_matched'),
        ('all7_vs_core5_own_stocks', 'sort_all7', 'sort_core5'),
        ('all7_fmp_vs_sort', 'fmp_all7', 'sort_all7_common'),
        ('core5_fmp_vs_sort', 'fmp_core5', 'sort_core5_common'),
        ('all7_vs_core5_fmp_same_stocks', 'fmp_all7', 'fmp_core5_matched'),
        ('all7_extra_beta_vol_controls', 'fmp_all7_defensive', 'fmp_all7'),
        ('core5_extra_beta_vol_controls', 'fmp_core5_defensive', 'fmp_core5')]
    rows = []
    for name, a, b in pairs:
        left = monthly[monthly.signal.eq(a) & monthly.indicator_available]
        right = monthly[monthly.signal.eq(b) & monthly.indicator_available]
        keys = ['signal_date', 'leg', 'scenario', 'missing_policy']
        common = left.merge(right, on=keys, suffixes=('_a', '_b'), validate='one_to_one')
        for key, group in common.groupby(keys[1:], sort=False):
            for period in ['full', 'early', 'late']:
                part = group[statistics.period_mask(group.signal_date, period)]
                if len(part) < 12:
                    continue
                difference = part.return_a - part.return_b
                mean, t = statistics.base.hac_mean(difference, CONFIG['hac_lags'])
                row = dict(zip(keys[1:], key), comparison=name, a=a, b=b, period=period,
                    months=len(part), first_signal_date=part.signal_date.min(), last_signal_date=part.signal_date.max(),
                    mean_monthly_difference=mean, annualized_mean_difference=12 * mean, difference_hac_t=t,
                    return_correlation=part.return_a.corr(part.return_b))
                for suffix in ['a', 'b']:
                    metrics = statistics.cash_stats(part['return_' + suffix], part['risk_free_' + suffix],
                        part['entry_date_' + suffix], part['exit_date_' + suffix])
                    row.update({k + '_' + suffix: v for k, v in metrics.items()})
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'matched_comparisons.csv', index=False)


def compare_prior_universe(monthly):
    prior_path = ROOT / 'results/combined_quality__broad_20261112/statistics_monthly_returns.csv'
    previous = pd.read_csv(prior_path, parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    keys = ['signal', 'leg', 'scenario', 'missing_policy', 'signal_date']
    joined = monthly[monthly.indicator_available].merge(previous[previous.indicator_available], on=keys,
        suffixes=('_iwv', '_broad'), validate='one_to_one')
    np.testing.assert_allclose(joined.risk_free_iwv, joined.risk_free_broad)
    rows = []
    for key, group in joined.groupby(keys[:-1], sort=False):
        for period in ['full', 'common_history', 'early', 'late']:
            part = group[statistics.period_mask(group.signal_date, period)]
            mean, t = statistics.base.hac_mean(part.return_iwv - part.return_broad, CONFIG['hac_lags'])
            row = dict(zip(keys[:-1], key), period=period, months=len(part),
                first_signal_date=part.signal_date.min(), last_signal_date=part.signal_date.max(),
                annualized_mean_difference=12 * mean, difference_hac_t=t,
                return_correlation=part.return_iwv.corr(part.return_broad))
            for suffix in ['iwv', 'broad']:
                metrics = statistics.cash_stats(part['return_' + suffix], part['risk_free_' + suffix],
                    part['entry_date_' + suffix], part['exit_date_' + suffix])
                row.update({k + '_' + suffix: v for k, v in metrics.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'universe_change_comparison.csv', index=False)


def old_numeric_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ['quality_dimensions', 'joint_fmp', 'quality_profitability_bundle', 'monthly_long_short']
        for p in (ROOT / 'results' / folder).glob('*') if p.suffix in ['.csv', '.parquet']}


def factor_statistics():
    coefficients = pd.read_csv(OUT / 'factor_returns.csv', parse_dates=['signal_date'])
    rows = []
    for (signal, factor), group in coefficients.groupby(['signal', 'factor'], sort=False):
        for period in ['full', 'common_history', 'early', 'late']:
            selected = group[statistics.period_mask(group.signal_date, period)]
            mean, t = statistics.base.hac_mean(selected.coefficient, CONFIG['hac_lags'])
            rows.append(dict(signal=signal, factor=factor, period=period, months=len(selected),
                annualized_mean_coefficient=12 * mean, coefficient_hac_t=t,
                mean_observed_fraction=(selected.observed_n / selected.n).mean()))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'factor_summary.csv', index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = old_numeric_hashes()
    data, calendar = read_inputs()
    calendar.to_csv(OUT / 'statistics_risk_free_windows.csv', index=False)
    scores, books = scores_and_books(data, calendar)
    configure_statistics()
    predictors = scores[KEYS + [s + '_signal' for s in SORT_SIGNALS]].rename(columns={s + '_signal': s for s in SORT_SIGNALS})
    returns = data[KEYS + ['return_observed', 'forward_return']].merge(predictors, on=KEYS, validate='one_to_one')
    statistics.ic_statistics(returns)
    monthly = portfolio_statistics(books, calendar)
    statistics.summarize(monthly)
    statistics.regime_statistics(monthly)
    matched_comparisons(monthly)
    compare_prior_universe(monthly)
    factor_statistics()
    assert before == old_numeric_hashes(), 'An older numeric result changed'
    manifest = dict(config=CFG, accounting_config=CONFIG, specs=SPECS,
        universe=json.loads(universe.RECEIPT.read_text()),
        code_sha256=sha(__file__), accounting_sha256=sha(accounting.__file__),
        inputs={str(p): sha(p) for p in [dimensions.CACHE, dimensions.MONTHLY, joint.CACHE, CONFIG_PATH,
            ROOT / 'config/quality_dimensions.json', ROOT / 'config/quality_dimension_statistics.json',
            ROOT / 'results/regimes/regime_features.csv', dimensions.OUT / 'statistics_risk_free_windows.csv',
            universe.SOURCE, Path(CFG['universe']['generation_summary_path']),
            ROOT / 'results/combined_quality__broad_20261112/statistics_monthly_returns.csv',
            Path(CFG['universe']['package_config_path']), universe.FLAGS, universe.SELECTED, universe.CALENDAR]},
        calendar_months=len(calendar), monthly_rows=len(monthly), old_numeric_sha256=before,
        limitations=['exploratory viewed sample; no untouched OOS', 'vendor revisions and estimated publication dates',
            'adjusted-price proxy and incomplete delistings', 'no observed short availability or security-specific borrow fees',
            'no PIT sector neutrality', 'active cash/collateral interest and funding zero',
            'RF before score availability and outside reporting regimes', 'regime switching turnover not recomputed',
            'raw WLS coefficients use flat unknown underlying returns; adverse sensitivity is in portfolio accounting',
            'entry rejections can break theoretical FMP neutrality', 'matched pair tests use identical monthly windows'],
        output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.suffix in ['.csv', '.parquet']})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = pd.read_csv(OUT / 'statistics_summary.csv')
    print(summary[(summary.period == 'available_history') & (summary.leg == 'long_short') &
        (summary.scenario == 'gross_unconstrained') & (summary.missing_policy == 'flat_unknown')]
        [['signal', 'months', 'cagr', 'annual_vol', 'sharpe_rf']].to_string(index=False), flush=True)


def seal_outputs():
    """Seal only outputs actually covered by current numeric/browser receipts."""
    path = OUT / 'manifest.json'
    manifest = json.loads(path.read_text())
    validation = json.loads((OUT / 'validation.json').read_text())
    ui = json.loads((OUT / 'ui_validation.json').read_text())
    assert validation['status'] == ui['status'] == 'passed'
    for name, digest in validation['numeric_output_sha256'].items():
        assert sha(OUT / name) == digest, 'Numeric receipt is stale: ' + name
    for name, digest in ui['html_sha256'].items():
        assert sha(OUT / name) == digest, 'Browser receipt is stale: ' + name
    for name, digest in manifest['inputs'].items():
        assert sha(Path(name)) == digest, 'Input changed: ' + name
    assert old_numeric_hashes() == manifest['old_numeric_sha256']
    baseline = json.loads((ROOT / 'data/reference/quality_rename_20261112/numeric_outputs_before.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in baseline.items())
    manifest.update(status='validated', validation={k: v for k, v in validation.items() if k != 'numeric_output_sha256'},
        ui_validation={k: v for k, v in ui.items() if k != 'html_sha256'}, old_rename_baseline_files_unchanged=len(baseline),
        code_sha256=sha(__file__), code_files_sha256={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'scripts' / name for name in
            ['analyze_combined_quality.py', 'build_iwv_universe.py', 'build_combined_quality_report.py', 'validate_combined_quality.py',
             'validate_combined_quality_ui.py', 'build_quality_dimension_statistics_report.py', 'build_quality_dimensions_report.py']]},
        linked_indicator_report_sha256=sha(dimensions.OUT / 'report.html'),
        output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'})
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Sealed validated outputs; older numeric results unchanged.', flush=True)


if __name__ == '__main__':
    if '--seal' in sys.argv:
        seal_outputs()
    else:
        main()
