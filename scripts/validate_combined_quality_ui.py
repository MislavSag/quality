"""Browser verification of summary links, series, sides, costs, RF and exports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

import analyze_combined_quality as analysis
from build_quality_dimension_statistics_report import filename
from validate_quality_dimension_statistics import independent_metrics
from validate_quality_dimension_statistics_ui import clean, hac_t

ROOT, OUT = analysis.ROOT, analysis.OUT
ARTIFACTS = ROOT / 'output/playwright'
ORIGIN = 'http://127.0.0.1:8770/combined_quality/'


def main():
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    for path in OUT.glob('*.html'):
        scripts = re.findall(r'<script>(.*?)</script>', path.read_text(encoding='utf-8'), re.S)
        for javascript in scripts:
            result = subprocess.run(['node', '--check'], input=javascript, capture_output=True, text=True, encoding='utf-8')
            assert result.returncode == 0, str(path) + ': ' + result.stderr
    monthly = pd.read_csv(OUT / 'statistics_monthly_returns.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    features = pd.read_csv(ROOT / 'results/regimes/regime_features.csv', parse_dates=['signal_date'])
    monthly = monthly.merge(features[['signal_date', 'common_sample'] + [r['id'] + '_state' for r in analysis.statistics.REGIMES]], on='signal_date', validate='many_to_one')
    summary = pd.read_csv(OUT / 'statistics_summary.csv')
    ic = pd.read_csv(OUT / 'monthly_ic.csv', parse_dates=['signal_date'])
    pairs = pd.read_csv(OUT / 'matched_comparisons.csv')
    factors = pd.read_csv(OUT / 'factor_summary.csv')
    overview_cases, cases = [], []
    for i, signal in enumerate(analysis.SIGNALS):
        for j in range(2):
            period = ['full', 'available_history', 'common_history', 'early', 'late'][(i+j) % 5]
            leg = analysis.CONFIG['legs'][(i+j) % 3]
            scenario = analysis.CONFIG['scenarios'][(i+j) % 3]['name']
            policy = analysis.CONFIG['missing_policies'][(i+j) % 2]
            row = summary[(summary.signal == signal) & (summary.period == period) & (summary.leg == leg) & (summary.scenario == scenario) & (summary.missing_policy == policy)].iloc[0]
            overview_cases.append(dict(signal=signal, period=period, leg=leg, scenario=scenario, policy=policy,
                expected={k: row[k] for k in ['cagr', 'annual_vol', 'sharpe_rf', 'monthly_max_drawdown', 'months', 'active_months', 'cash_months']}))

    def add(signal, leg='long_short', scenario='gross_unconstrained', policy='flat_unknown', filters=None, start='2007-02', end='2026-08'):
        filters = filters or {}
        compare = {}
        for name in analysis.SIGNALS:
            group = monthly[(monthly.signal == name) & (monthly.leg == leg) & (monthly.scenario == scenario) & (monthly.missing_policy == policy)]
            group = group[group.entry_date.dt.strftime('%Y-%m').between(start, end)]
            gate = np.ones(len(group), bool)
            if filters:
                gate &= group.common_sample
                for regime, level in filters.items():
                    gate &= group[regime + '_state'].eq(level)
            active = gate & group.indicator_available
            compare[name] = {**independent_metrics(np.where(active, group['return'], group.risk_free), group.risk_free, group.entry_date, group.exit_date), 'active': int(active.sum()), 'cash': int((~active).sum())}
            if name == signal:
                dates = group.loc[gate, 'signal_date']
        ic_expected = {name: dict(spearman=g.ic_spearman.mean(), hac_t=hac_t(g.ic_spearman), cosine_raw=g.ic_cosine_raw.mean(), cosine_rank=g.ic_cosine_rank_centered.mean()) for name, g in ic[ic.signal_date.isin(dates)].groupby('signal')}
        cases.append(dict(signal=signal, leg=leg, scenario=scenario, policy=policy, filters=filters, start=start, end=end,
            file=filename(scenario, policy), expected=compare[signal], comparison=compare, ic=ic_expected))

    for i, signal in enumerate(analysis.SIGNALS):
        add(signal)
        add(signal, leg=analysis.CONFIG['legs'][i % 3], scenario=analysis.CONFIG['scenarios'][i % 3]['name'], policy=analysis.CONFIG['missing_policies'][i % 2], filters={'spy_vol21': i % 2})
    add('sort_all7', end='2012-01')
    add('fmp_core5', end='2009-06')
    add('fmp_all7', leg='short', scenario='cost10_borrow3', policy='adverse_unknown', filters={'spy_vol21': 1, 'spy_trend126': 0})
    add('fmp_core5', start='2012-02')
    script = r'''async page => {
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
const close=(a,b,label,tol=1e-9)=>{if(b===null){if(Number.isFinite(a))throw Error(label+' expected NaN, got '+a);}else if(!Number.isFinite(a)||Math.abs(a-b)>tol)throw Error(label+' '+a+' != '+b);};
await page.setViewportSize({width:1500,height:1100});await page.goto(__ORIGIN__+'report.html');
for(const [key,value]of Object.entries(__UNIVERSE_COUNTS__))close(Number(await page.locator('[data-universe="'+key+'"]').textContent()),value,'IWV coverage '+key);
let overview=0,comparison=0,icCells=0;
for(const c of __OVERVIEW__){for(const [id,v]of Object.entries({signal:c.signal,period:c.period,scenario:c.scenario,leg:c.leg,policy:c.policy}))await page.locator('#'+id).selectOption(v);
const snap=await page.evaluate(()=>window.combinedQualitySnapshot);for(const [key,wanted]of Object.entries(c.expected))close(snap.statistics[key],wanted,'Overview '+key);overview++;
const value=await page.locator('#metrics .metric strong').first().textContent();close(Number(value.replaceAll('.','').replace(',','.').replace('−','-').replace('%',''))/100,c.expected.cagr,'Visible CAGR',.000051);}
await page.locator('#signal').selectOption('fmp_all7');await page.locator('#period').selectOption('available_history');await page.locator('#scenario').selectOption('cost10_borrow3');await page.locator('#leg').selectOption('short');await page.locator('#policy').selectOption('adverse_unknown');
await page.locator('#pairPeriod').selectOption('late');const snap=await page.evaluate(()=>window.combinedQualitySnapshot);
const wantedPairs=__PAIRS__;if(snap.matched.length!==wantedPairs.length)throw Error('Paired rows');for(const r of snap.matched){const w=wantedPairs.find(p=>p.comparison===r.comparison);for(const k of ['cagr_a','cagr_b','difference_hac_t','annualized_mean_difference','months'])close(r[k],w[k],'Paired '+k);}
const wantedFactors=__FACTORS__;for(const r of wantedFactors)close(Number(await page.locator('#factorBody [data-factor="'+r.factor+'"] [data-coefficient]').getAttribute('data-coefficient')),r.annualized_mean_coefficient,'Coefficient');
await page.locator('#signal').selectOption('sort_all7');await page.locator('#search').fill('AAPL');if(await page.locator('#stocks tr').count()>1)throw Error('Stock search');await page.locator('#search').fill('');
await page.locator('h1').scrollIntoViewIfNeeded();await page.screenshot({path:__OVERVIEW_IMAGE__});
await page.locator('#dashboard').click();const linked=await page.evaluate(()=>window.qualityReportSnapshot);if(linked.series!=='sort_all7'||linked.leg!=='short'||linked.calendar.length!==175)throw Error('Overview/dashboard state');
for(const c of __CASES__){await page.goto(__ORIGIN__+c.file);await page.locator('#reset').click();await page.locator('#focal').selectOption(c.signal);await page.locator('#legMode').selectOption(c.leg);await page.locator('#from').fill(c.start);await page.locator('#to').fill(c.end);for(const [r,v]of Object.entries(c.filters))await page.locator('#filter-'+r).selectOption(String(v));
const result=await page.evaluate(()=>window.qualityReportSnapshot);for(const [key,source]of Object.entries({cagr:'cagr',vol:'annual_vol',sharpe:'sharpe_rf',dd:'monthly_max_drawdown'}))close(result.metrics[key],c.expected[source],'Dashboard '+c.signal+' '+key);
if(result.activeMonths!==c.expected.active||result.cashMonths!==c.expected.cash)throw Error('Active/cash months');
for(const [s,w]of Object.entries(c.comparison)){const row=page.locator('#comparisonBody tr[data-signal="'+s+'"]');for(const [key,source]of Object.entries({return:'cagr',vol:'annual_vol',sharpe:'sharpe_rf',dd:'monthly_max_drawdown'})){close(Number(await row.locator('[data-metric="'+key+'"]').getAttribute('data-value')),w[source],'Comparison');comparison++;}}
for(const s of __SORTS__){const row=page.locator('#icBody tr[data-ic-signal="'+s+'"]');for(const key of ['spearman','hac_t','cosine_raw','cosine_rank']){close(Number(await row.locator('[data-ic="'+key+'"]').getAttribute('data-value')),c.ic[s]?.[key]??null,'IC '+key);icCells++;}}
if(await page.locator('#performanceChart').getAttribute('data-x-scale')!=='calendar')throw Error('Calendar axis');
}
await page.goto(__ORIGIN__+'statistics__cost10_borrow3__adverse_unknown.html');await page.locator('#reset').click();await page.locator('#focal').selectOption('fmp_all7');await page.locator('#legMode').selectOption('short');await page.locator('[data-preset="spy_vol21"]').click();
const downloadP=page.waitForEvent('download');await page.locator('#export').click();const dl=await downloadP;await dl.saveAs(__EXPORT__);if(!dl.suggestedFilename().startsWith('combined_quality_'))throw Error('Export filename');
await page.locator('#reset').click();await page.locator('#focal').selectOption('sort_all7');await page.locator('#legMode').selectOption('long');await page.locator('#addComparison').click();await page.locator('#legMode').selectOption('short');await page.locator('#addComparison').click();await page.locator('#legMode').selectOption('long_short');await page.locator('#focal').selectOption('fmp_all7');await page.locator('[data-preset="spy_vol21"]').click();await page.locator('#addComparison').click();await page.reload();if((await page.evaluate(()=>window.qualityReportSnapshot.comparisons)).length!==3)throw Error('Saved comparison hash');
await page.locator('h1').scrollIntoViewIfNeeded();await page.screenshot({path:__DASHBOARD_IMAGE__});
await page.setViewportSize({width:390,height:844});if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error('Dashboard overflow');await page.screenshot({path:__MOBILE_IMAGE__});
await page.goto(__ORIGIN__+'report.html');if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error('Overview overflow');await page.screenshot({path:__MAIN_MOBILE_IMAGE__});
await page.goto(__ORIGIN__+'../quality_dimensions/report.html');const link=page.locator('a[href="../combined_quality/report.html"]');if(await link.count()!==1)throw Error('Existing report link');
if(errors.length)throw Error(errors.join('\n'));return {status:'passed',overview_cases:overview,dashboard_cases:__CASE_COUNT__,comparison_cells:comparison,ic_cells:icCells,matched_rows:wantedPairs.length,coefficient_rows:wantedFactors.length,saved_comparisons:3,mobile_overflow:false,calendar_axis:true,page_errors:errors};
}'''
    selected_pairs = pairs[(pairs.period == 'late') & (pairs.leg == 'short') & (pairs.scenario == 'cost10_borrow3') & (pairs.missing_policy == 'adverse_unknown')]
    selected_factors = factors[(factors.signal == 'fmp_all7') & (factors.period == 'full') & factors.factor.isin(analysis.CFG['all7'])]
    replacements = {'__ORIGIN__': ORIGIN, '__OVERVIEW__': overview_cases, '__CASES__': cases, '__SORTS__': analysis.SORT_SIGNALS,
        '__PAIRS__': selected_pairs.to_dict('records'), '__FACTORS__': selected_factors.to_dict('records'), '__CASE_COUNT__': len(cases),
        '__OVERVIEW_IMAGE__': str(ARTIFACTS / 'combined-quality-overview.png'),
        '__DASHBOARD_IMAGE__': str(ARTIFACTS / 'combined-quality-dashboard.png'),
        '__MOBILE_IMAGE__': str(ARTIFACTS / 'combined-quality-mobile.png'),
        '__MAIN_MOBILE_IMAGE__': str(ARTIFACTS / 'combined-quality-overview-mobile.png'),
        '__EXPORT__': str(ARTIFACTS / 'combined-quality-export.csv')}
    last = pd.read_csv(OUT / 'iwv_universe_coverage.csv').iloc[-1]
    cover = pd.read_csv(OUT / 'coverage.csv')
    latest = cover[cover.signal_date.eq(cover.signal_date.max())].set_index('universe')
    replacements['__UNIVERSE_COUNTS__'] = dict(holdings=int(last.holdings_n), usable=int(last.holdings_identity_usable_n),
        panel=int(last.panel_matched_members_n), eligible=int(last.iwv_eligible_n),
        all7=int(latest.loc['all7','scored_n']), core5=int(latest.loc['core5','scored_n']))
    for token, value in replacements.items():
        script = script.replace(token, json.dumps(clean(value), ensure_ascii=False, allow_nan=False))
    path = ARTIFACTS / 'verify-combined-quality.js'
    path.write_text(script, encoding='utf-8')
    packages = Path(os.environ['LOCALAPPDATA']) / 'npm-cache/_npx'
    package = max(packages.glob('*/node_modules/@playwright/cli/package.json'), key=lambda p: p.stat().st_mtime)
    cli = package.parent / 'playwright-cli.js'
    opened = subprocess.run(['node', str(cli), '--session', 'combined_quality', 'open', ORIGIN + 'report.html'], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert opened.returncode == 0 and '### Error' not in opened.stdout, opened.stdout
    result = subprocess.run(['node', str(cli), '--session', 'combined_quality', 'run-code', '--filename', str(path)], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8', timeout=240)
    assert result.returncode == 0 and '### Error' not in result.stdout, result.stdout
    receipt = json.loads(result.stdout.split('### Result\n', 1)[1].split('\n### Ran Playwright code', 1)[0])
    exported = pd.read_csv(ARTIFACTS / 'combined-quality-export.csv')
    assert exported.universe.eq('IWV_historical').all()
    expected = monthly[(monthly.leg == 'short') & (monthly.scenario == 'cost10_borrow3') & (monthly.missing_policy == 'adverse_unknown')]
    for signal, group in expected.groupby('signal'):
        active = group.indicator_available & group.common_sample & group.spy_vol21_state.eq(1)
        np.testing.assert_allclose(exported[signal], np.where(active, group['return'], group.risk_free), atol=1e-12)
        np.testing.assert_array_equal(exported[signal + '_active'], active.astype(int))
    receipt.update(export_rows=len(exported), export_return_values=len(exported) * len(analysis.SIGNALS),
        html_sha256={p.name: analysis.sha(p) for p in OUT.glob('*.html')})
    (OUT / 'ui_validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
