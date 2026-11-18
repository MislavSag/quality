"""Independent common-universe ranks, WLS operators, ledgers and comparisons."""
from __future__ import annotations

import json
import sys
from bisect import bisect_right

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

import analyze_combined_quality as analysis
import validate_quality_dimension_statistics as ledger

ROOT, OUT = analysis.ROOT, analysis.OUT


def close(a, b, label, atol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=1e-8, equal_nan=True, err_msg=str(label))


def verify_iwv_membership():
    """Independently select snapshots from the original ETF package output."""
    source = analysis.universe.SOURCE
    source_dates = pd.read_parquet(source, columns=['target_snapshot_date'], filters=[('etf', '==', 'IWV')])
    dates = sorted(source_dates.target_snapshot_date.unique())
    flags = pd.read_parquet(analysis.universe.FLAGS)
    monthly = pd.read_parquet(analysis.dimensions.MONTHLY, columns=analysis.KEYS + ['eligible'])
    assert not flags.duplicated(analysis.KEYS).any()
    assert len(flags) == len(monthly)
    np.testing.assert_array_equal(flags[analysis.KEYS], monthly[analysis.KEYS])
    np.testing.assert_array_equal(flags.eligible, monthly.eligible)
    chosen = {date: (dates[bisect_right(dates, date) - 1] if bisect_right(dates, date) else None)
        for date in flags.signal_date.unique()}
    used = sorted({d for d in chosen.values() if d is not None})
    raw = pd.read_parquet(source, columns=['target_snapshot_date', 'source_snapshot_date', 'sid', 'identity_usable', 'weight'],
        filters=[('etf', '==', 'IWV'), ('target_snapshot_date', 'in', used)])
    groups = {d: g for d, g in raw.groupby('target_snapshot_date')}
    checked, active, zero_weight_checks = 0, 0, 0
    for date, g in flags.groupby('signal_date', sort=True):
        snapshot_date = chosen[date]
        if snapshot_date is None:
            assert not g.iwv_member.any() and not g.iwv_eligible.any()
            assert g.target_snapshot_date.isna().all()
            continue
        assert g.target_snapshot_date.eq(snapshot_date).all()
        snapshot = groups[snapshot_date]
        assert (snapshot.source_snapshot_date <= snapshot_date).all() and snapshot_date <= date
        ids = set(snapshot.loc[snapshot.identity_usable, 'sid'])
        expected = g.security_id.isin(ids)
        np.testing.assert_array_equal(g.iwv_member, expected)
        np.testing.assert_array_equal(g.iwv_eligible, expected & g.eligible)
        zero_ids = set(snapshot.loc[snapshot.identity_usable & snapshot.weight.eq(0), 'sid'])
        present_zero = g.security_id.isin(zero_ids)
        assert g.loc[present_zero, 'iwv_member'].all()
        zero_weight_checks += int(present_zero.sum())
        checked += len(g)
        active += 1
    # Every holding (including FMP) must be a validated exact-SID member at formation.
    holdings = pd.read_parquet(OUT / 'statistics_formation_holdings.parquet', columns=analysis.KEYS).drop_duplicates()
    admitted = flags[flags.iwv_eligible][analysis.KEYS]
    matched = holdings.merge(admitted.assign(admitted=True), on=analysis.KEYS, how='left', validate='one_to_one')
    assert matched.admitted.eq(True).all()
    return dict(formation_months=active, checked_panel_rows=checked, retained_zero_weight_members=zero_weight_checks,
        verified_distinct_holdings=len(holdings), pre_snapshot_months=len(chosen) - active,
        first_source_snapshot=str(pd.Timestamp(dates[0]).date()), first_formation=str(min(d for d, s in chosen.items() if s is not None).date()))


def verify_scores():
    data, _ = analysis.read_inputs()
    data = data.set_index(analysis.KEYS)
    scores = pd.read_parquet(OUT / 'monthly_combined_scores.parquet').set_index(analysis.KEYS)
    exposures = pd.read_parquet(OUT / 'dimension_exposures.parquet')
    ranks_saved = pd.read_parquet(OUT / 'component_ranks.parquet')
    rank_groups = {(u, b, d): g.set_index('security_id') for (u, b, d), g in ranks_saved.groupby(['universe', 'bundle', 'signal_date'])}
    count, rank_count, universe_checks = 0, 0, 0
    for (universe, bundle, date), g in exposures.groupby(['universe', 'bundle', 'signal_date'], sort=False):
        g = g.sort_values('security_id')
        raw = data.xs(date, level='signal_date').loc[g.security_id]
        expected_columns = analysis.fields_for('all7' if universe.startswith('all7') else 'core5')
        eligible = data.xs(date, level='signal_date')
        valid = np.isfinite(eligible[expected_columns]).all(axis=1)
        if universe.endswith('_common'):
            valid &= np.isfinite(eligible[analysis.CFG['controls_defensive']]).all(axis=1)
        assert set(g.security_id) == set(eligible.index[valid]), (universe, date)
        universe_checks += 1
        components = rank_groups[universe, bundle, date].loc[g.security_id]
        zs = []
        for dimension in analysis.CFG[bundle]:
            spec = analysis.dimensions.SPECS[dimension]['components']
            ranked = []
            for field, direction in spec.items():
                x = raw[field].to_numpy() * direction
                rank = (rankdata(x, method='average') - .5) / len(x)
                close(components[dimension + '__' + field], rank, 'component ranks')
                ranked.append(rank)
                rank_count += len(x)
            avg = np.vstack(ranked).mean(axis=0)
            z = ndtri((rankdata(avg, method='average') - .5) / len(avg))
            z = (z - z.mean()) / z.std(ddof=0)
            close(g[dimension], z, 'dimension normal score')
            zs.append(z)
        z = np.vstack(zs).mean(axis=0)
        pct = 100 * (rankdata(z, method='average') - .5) / len(z)
        for signal, spec in analysis.SPECS.items():
            if spec['universe'] == universe and spec['bundle'] == bundle:
                saved = scores.xs(date, level='signal_date').loc[g.security_id]
                close(saved[signal + '_signal'], z, signal)
                close(saved[signal + '_score'], pct, signal + ' displayed percentile')
                count += len(z)
    # Matched universes must be identical, with exact calendar-month warm-up.
    for date, g in scores.reset_index().groupby('signal_date'):
        for a, b in [('sort_all7', 'sort_core5_matched'), ('sort_all7_common', 'sort_core5_matched_common'),
                     ('fmp_all7', 'fmp_core5_matched'), ('fmp_all7', 'fmp_all7_defensive')]:
            np.testing.assert_array_equal(g[a + '_signal'].notna(), g[b + '_signal'].notna())
        if date < pd.Timestamp('2012-01-01'):
            assert g.sort_all7_signal.isna().all()
    return count, rank_count, universe_checks


def verify_fmps():
    designs = pd.read_parquet(OUT / 'fmp_design.parquet')
    weights = pd.read_parquet(OUT / 'fmp_weights.parquet')
    groups = {(s, d): g.set_index('security_id') for (s, d), g in weights.groupby(['signal', 'signal_date'])}
    data, calendar = analysis.read_inputs()
    raw_groups = {d: g.set_index('security_id') for d, g in data.groupby('signal_date')}
    rf = calendar.set_index('signal_date').risk_free
    coefs = pd.read_csv(OUT / 'factor_returns.csv', parse_dates=['signal_date']).set_index(['signal', 'signal_date', 'factor'])
    diagnostics = pd.read_csv(OUT / 'fmp_diagnostics.csv', parse_dates=['signal_date']).set_index(['signal', 'signal_date'])
    count, max_identity, max_operator = 0, 0., 0.
    for (signal, date), g in designs.groupby(['signal', 'signal_date'], sort=False):
        g = g.sort_values('security_id')
        spec = analysis.SPECS[signal]
        dimensions = analysis.CFG[spec['bundle']]
        controls = analysis.CFG['controls_' + spec['controls']]
        names = ['market', *dimensions, *controls]
        x = g[names].to_numpy()
        original = raw_groups[date].loc[g.security_id]
        cap = original.FUND_MARKET_CAP_USD_COMPAT.to_numpy()
        close(g.sqrt_cap_weight, np.sqrt(cap) / np.sqrt(cap).mean(), 'WLS sqrt cap objective')
        for c in controls:
            z = ndtri((rankdata(original[c]) - .5) / len(original))
            z = (z - np.average(z, weights=cap)) / z.std(ddof=0)
            close(g[c], z, 'transformed control ' + c)
        # SVD pseudoinverse is independent of production's QR solver.
        root = np.sqrt(g.sqrt_cap_weight.to_numpy())
        operator = np.linalg.pinv(x * root[:, None], rcond=1e-12) * root[None, :]
        saved = groups[signal, date].loc[g.security_id]
        expected_target = np.r_[0., np.repeat(1 / len(dimensions), len(dimensions)), np.zeros(len(controls))]
        raw = operator.T @ expected_target
        close(saved.raw_combined_weight, raw, 'raw aggregate FMP')
        for j, dimension in enumerate(dimensions):
            close(saved['raw__' + dimension], operator[j + 1], 'individual raw FMP')
            max_operator = max(max_operator, float(np.max(np.abs(saved['raw__' + dimension] - operator[j + 1]))))
        target = raw * 2 / np.abs(raw).sum()
        close(saved.requested_weight, target, 'net aggregate then gross 2')
        error = float(np.max(np.abs(x.T @ saved.raw_combined_weight - expected_target)))
        assert error < 1e-8
        max_identity = max(max_identity, error)
        known = original.return_observed & np.isfinite(original.forward_return)
        response = np.where(known, original.forward_return, 0.)
        f = operator @ (response - rf.loc[date])
        for j, name in enumerate(names):
            close(coefs.loc[(signal, date, name), 'coefficient'], f[j], 'regression coefficient')
        diag = diagnostics.loc[signal, date]
        close(diag.raw_quality_return, raw @ response, 'raw FMP PnL')
        close(diag.coefficient_combination, expected_target @ f, 'coefficient combination')
        close(diag.raw_quality_return, diag.coefficient_combination, 'raw identity in returns')
        assert diag.design_condition < 1e8
        count += 1
    return count, max_identity, max_operator


def verify_sort_formation_and_ic(holdings):
    scores = pd.read_parquet(OUT / 'monthly_combined_scores.parquet')
    future = pd.read_parquet(analysis.dimensions.MONTHLY, columns=analysis.KEYS + ['return_observed', 'forward_return'])
    scores = scores.merge(future, on=analysis.KEYS, validate='one_to_one')
    ic = pd.read_csv(OUT / 'monthly_ic.csv', parse_dates=['signal_date']).set_index(['signal', 'signal_date'])
    selected = {(s, d): g for (s, d), g in holdings.groupby(['signal', 'signal_date'])}
    count = 0
    for signal in analysis.SORT_SIGNALS:
        for date, g in scores[scores[signal + '_signal'].notna()].groupby('signal_date'):
            z = g[signal + '_signal'].to_numpy()
            pct = (rankdata(z) - .5) / len(z)
            book = selected[signal, date]
            assert set(book.security_id) == set(g.loc[(pct < .2) | (pct >= .8), 'security_id'])
            for sign in [1, -1]:
                part = book[book.side.eq(sign)]
                close(part.requested_weight, sign / len(part), 'equal tail weights')
            known = g.return_observed.to_numpy() & np.isfinite(g.forward_return.to_numpy())
            x, y = z[known], g.forward_return.to_numpy()[known]
            def cosine(a, b):
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                return a @ b / denom if denom > 0 else np.nan
            expected = np.corrcoef(rankdata(x), rankdata(y))[0, 1] if len(x) >= 100 else np.nan
            row = ic.loc[signal, date]
            close(row.ic_spearman, expected, 'IC')
            close(row.ic_cosine_raw, cosine(x, y), 'cosine signal')
            r = rankdata(z) - rankdata(z).mean()
            close(row.ic_cosine_rank_centered, cosine(r[known], y), 'cosine centered ranks')
            assert row.n == len(z) and row.n_observed == known.sum()
            count += 1
    return count


def verify_matched(monthly):
    comparisons = pd.read_csv(OUT / 'matched_comparisons.csv')
    n = 0
    for row in comparisons.itertuples():
        common = monthly[monthly.leg.eq(row.leg) & monthly.scenario.eq(row.scenario) &
            monthly.missing_policy.eq(row.missing_policy) & monthly.indicator_available]
        a = common[common.signal.eq(row.a)].set_index('signal_date')
        b = common[common.signal.eq(row.b)].set_index('signal_date')
        dates = a.index.intersection(b.index)
        if row.period == 'early':
            dates = dates[dates < pd.Timestamp('2017-01-01')]
        elif row.period == 'late':
            dates = dates[dates >= pd.Timestamp('2017-01-01')]
        a, b = a.loc[dates], b.loc[dates]
        assert len(a) == row.months
        for suffix, g in [('a', a), ('b', b)]:
            metrics = ledger.independent_metrics(g['return'], g.risk_free, g.entry_date, g.exit_date)
            for metric, value in metrics.items():
                close(getattr(row, metric + '_' + suffix), value, 'matched metric')
        close(row.annualized_mean_difference, (a['return'] - b['return']).mean() * 12, 'paired annual mean difference')
        n += 1
    return n


def verify_inference():
    def independent_hac(values):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        avg = values.mean()
        residual = values - avg
        meat = residual @ residual
        for lag in range(1, 4):
            meat += 2 * (1 - lag / 4) * (residual[lag:] @ residual[:-lag])
        return avg, avg * len(values) / np.sqrt(meat) if meat > 0 and len(values) >= 12 else np.nan
    coefficients = pd.read_csv(OUT / 'factor_returns.csv', parse_dates=['signal_date'])
    summary = pd.read_csv(OUT / 'factor_summary.csv')
    count = 0
    for row in summary.itertuples():
        g = coefficients[(coefficients.signal == row.signal) & (coefficients.factor == row.factor)]
        if row.period == 'early':
            g = g[g.signal_date < '2017-01-01']
        elif row.period == 'late':
            g = g[g.signal_date >= '2017-01-01']
        elif row.period == 'common_history':
            g = g[g.signal_date >= '2012-01-01']
        avg, t = independent_hac(g.coefficient)
        close(row.annualized_mean_coefficient, 12 * avg, 'coefficient annual mean')
        close(row.coefficient_hac_t, t, 'coefficient HAC t')
        assert row.months == len(g)
        count += 1
    monthly = pd.read_csv(OUT / 'statistics_monthly_returns.csv', parse_dates=['signal_date'], low_memory=False)
    comparisons = pd.read_csv(OUT / 'matched_comparisons.csv')
    for row in comparisons.itertuples():
        g = monthly[(monthly.leg == row.leg) & (monthly.scenario == row.scenario) &
            (monthly.missing_policy == row.missing_policy) & monthly.indicator_available]
        a = g[g.signal == row.a].set_index('signal_date')['return']
        b = g[g.signal == row.b].set_index('signal_date')['return']
        dates = a.index.intersection(b.index)
        if row.period == 'early':
            dates = dates[dates < '2017-01-01']
        elif row.period == 'late':
            dates = dates[dates >= '2017-01-01']
        avg, t = independent_hac(a.loc[dates] - b.loc[dates])
        close(row.difference_hac_t, t, 'paired difference HAC t')
        count += 1
    return count


def verify_universe_comparison(monthly):
    prior = pd.read_csv(ROOT / 'results/combined_quality__broad_20261112/statistics_monthly_returns.csv',
        parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    comparison = pd.read_csv(OUT / 'universe_change_comparison.csv')
    count = 0
    for row in comparison.itertuples():
        def select(frame):
            return frame[(frame.signal == row.signal) & (frame.leg == row.leg) & (frame.scenario == row.scenario) &
                (frame.missing_policy == row.missing_policy) & frame.indicator_available].set_index('signal_date')
        a, b = select(monthly), select(prior)
        dates = a.index.intersection(b.index)
        if row.period == 'common_history':
            dates = dates[dates >= pd.Timestamp('2012-01-01')]
        elif row.period == 'early':
            dates = dates[dates < pd.Timestamp('2017-01-01')]
        elif row.period == 'late':
            dates = dates[dates >= pd.Timestamp('2017-01-01')]
        assert len(dates) == row.months
        a, b = a.loc[dates], b.loc[dates]
        for suffix, g in [('iwv', a), ('broad', b)]:
            for metric, value in ledger.independent_metrics(g['return'], g.risk_free, g.entry_date, g.exit_date).items():
                close(getattr(row, metric + '_' + suffix), value, 'universe comparison metric')
        difference = (a['return'] - b['return']).to_numpy()
        residual = difference - difference.mean()
        meat = residual @ residual
        for lag in range(1, 4):
            meat += 2 * (1 - lag / 4) * (residual[lag:] @ residual[:-lag])
        t = difference.mean() * len(difference) / np.sqrt(meat)
        close(row.difference_hac_t, t, 'universe difference HAC t')
        count += 1
    return count


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        assert analysis.sha(OUT / name) == digest, name
    assert manifest['old_numeric_sha256'] == analysis.old_numeric_hashes()
    iwv = verify_iwv_membership()
    scored, ranks, universes = verify_scores()
    fmp, identity_error, operator_error = verify_fmps()
    monthly = pd.read_csv(OUT / 'statistics_monthly_returns.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    holdings = pd.read_parquet(OUT / 'statistics_formation_holdings.parquet')
    ic = verify_sort_formation_and_ic(holdings)
    ledger.OUT, ledger.analysis = OUT, analysis
    ledgers, cash, error = ledger.verify_ledgers(holdings, monthly)
    summaries = ledger.verify_summaries(monthly)
    regimes = ledger.verify_regimes(monthly)
    matched = verify_matched(monthly)
    inference = verify_inference()
    universe_comparisons = verify_universe_comparison(monthly)
    receipt = dict(status='passed', iwv_membership=iwv, scored_stock_months=scored, independently_checked_component_ranks=ranks,
        common_universe_checks=universes, fmp_formations=fmp, max_identity_error=identity_error,
        max_svd_qr_operator_error=operator_error, ic_cross_sections=ic, active_ledgers=ledgers,
        rf_cash_months=cash, max_return_error=error, summary_rows=summaries, regime_rows=regimes,
        matched_comparisons=matched, universe_comparison_rows=universe_comparisons, hac_inference_rows=inference,
        numeric_output_sha256={name: digest for name, digest in manifest['output_sha256'].items() if name.endswith(('.csv', '.parquet'))},
        old_numeric_files_unchanged=len(manifest['old_numeric_sha256']))
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    if '--inference-only' in sys.argv:
        count = verify_inference()
        p = OUT / 'validation.json'
        receipt = json.loads(p.read_text())
        receipt['hac_inference_rows'] = count
        p.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print('Independently verified HAC inference rows:', count)
    else:
        main()
