"""Verify offline indicator/statistics controls and browser metrics against CSVs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import analyze_quality_dimensions as analysis
from build_quality_dimension_statistics_report import filename
from validate_quality_dimension_statistics import independent_metrics

ROOT, OUT = analysis.ROOT, analysis.OUT
ARTIFACTS = ROOT / 'output/playwright'
ORIGIN = 'http://127.0.0.1:8769/quality_dimensions/'


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def hac_t(values):
    x = np.asarray(pd.Series(values).dropna())
    if len(x) < 12 or np.std(x) == 0:
        return np.nan
    residual = x - x.mean()
    meat = np.dot(residual, residual)
    for lag in range(1, 4):
        meat += 2 * (1 - lag / 4) * np.dot(residual[lag:], residual[:-lag])
    return x.mean() * len(x) / np.sqrt(meat) if meat > 0 else np.nan


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(OUT / 'statistics_monthly_returns.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    features = pd.read_csv(ROOT / 'results/regimes/regime_features.csv', parse_dates=['signal_date'])
    monthly = monthly.merge(features[['signal_date', 'common_sample'] + [r['id'] + '_state' for r in analysis.REGIMES]], on='signal_date', validate='many_to_one')
    ic = pd.read_csv(OUT / 'monthly_ic.csv', parse_dates=['signal_date'])
    summary = pd.read_csv(OUT / 'statistics_summary.csv')
    main_cases = []
    for index, signal in enumerate(analysis.SIGNALS):
        periods = ['full', 'available_history', 'common_history', 'early', 'late']
        for j in range(2):
            period = periods[(index + j) % len(periods)]
            leg = ['long_short', 'long', 'short'][(index + j) % 3]
            scenario = analysis.CONFIG['scenarios'][(index + j) % 3]['name']
            policy = analysis.CONFIG['missing_policies'][(index + j) % 2]
            row = summary[(summary.signal == signal) & (summary.period == period) & (summary.leg == leg) &
                          (summary.scenario == scenario) & (summary.missing_policy == policy)].iloc[0]
            main_cases.append(dict(signal=signal, period=period, leg=leg, scenario=scenario, policy=policy,
                expected={key: row[key] for key in ['cagr', 'annual_vol', 'sharpe_rf', 'monthly_max_drawdown', 'months', 'active_months', 'cash_months']}))
    cases = []
    def add(signal, leg='long_short', scenario='gross_unconstrained', policy='flat_unknown', filters=None, start='2007-02', end='2026-08'):
        filters = filters or {}
        comparison = {}
        active_dates = None
        for name in analysis.SIGNALS:
            group = monthly[(monthly.signal == name) & (monthly.leg == leg) & (monthly.scenario == scenario) & (monthly.missing_policy == policy)]
            group = group[group.entry_date.dt.strftime('%Y-%m').between(start, end)].copy()
            gate = np.ones(len(group), dtype=bool)
            if filters:
                gate &= group.common_sample.to_numpy()
                for regime, level in filters.items():
                    gate &= group[regime + '_state'].eq(level).to_numpy()
            active = gate & group.indicator_available
            values = np.where(active, group['return'], group.risk_free)
            comparison[name] = {**independent_metrics(values, group.risk_free, group.entry_date, group.exit_date), 'active': int(active.sum()), 'cash': int((~active).sum())}
            if name == signal:
                active_dates = group.loc[active, 'signal_date'].dt.strftime('%Y-%m-%d').tolist()
                selected_dates = group.loc[gate, 'signal_date']
        expected_ic = {}
        for name, selected in ic[ic.signal_date.isin(selected_dates)].groupby('signal'):
            expected_ic[name] = dict(spearman=selected.ic_spearman.mean(), hac_t=hac_t(selected.ic_spearman),
                cosine_raw=selected.ic_cosine_raw.mean(), cosine_rank=selected.ic_cosine_rank_centered.mean(),
                months=int(selected.ic_spearman.notna().sum()))
        cases.append(dict(signal=signal, leg=leg, scenario=scenario, policy=policy, filters=filters,
            start=start, end=end, file=filename(scenario, policy), expected=comparison[signal],
            comparison=comparison, active_dates=active_dates, expected_ic=expected_ic))
    for signal in analysis.SIGNALS:
        add(signal)
        add(signal, filters={'spy_vol21': 1})
    for leg in ['long_short', 'long', 'short']:
        for scenario in analysis.CONFIG['scenarios']:
            add('earnings_quality', leg=leg, scenario=scenario['name'])
    add('quality_growth', leg='short', scenario='cost10_borrow3', policy='adverse_unknown', filters={'spy_vol21': 1})
    add('profitability_broad', policy='adverse_unknown', filters={'spy_trend126': 0, 'breadth126': 0})
    add('quality_growth', end='2012-01')
    add('quality_growth', start='2012-02')
    add('earnings_stability', scenario='cost10_borrow3', start='2017-02')
    script = r'''async page => {
      const errors=[];page.on('pageerror',error=>errors.push(String(error)));
      const close=(actual,expected,label,tol=1e-9)=>{
        if(expected===null){if(Number.isFinite(actual))throw Error(label+' expected undefined, got '+actual);}
        else if(!Number.isFinite(actual)||Math.abs(actual-expected)>tol)throw Error(label+' '+actual+' vs '+expected);
      };
      await page.setViewportSize({width:1440,height:1080});
      await page.goto(__ORIGIN__+'report.html');
      for(const c of __MAIN_CASES__){
        await page.locator('#dimension').selectOption(c.signal);
        await page.locator('#statsPeriod').selectOption(c.period);
        await page.locator('#statsScenario').selectOption(c.scenario);
        await page.locator('#statsLeg').selectOption(c.leg);
        await page.locator('#statsMissing').selectOption(c.policy);
        const result=await page.evaluate(()=>window.qualityDimensionSnapshot.statistics);
        for(const [key,value]of Object.entries(c.expected))close(result[key],value,'Main '+key);
        const display=await page.locator('#statsMetrics .metric strong').first().textContent();
        const numeric=Number(display.replaceAll('.','').replace(',','.').replace('−','-').replace('%',''))/100;
        close(numeric,c.expected.cagr,'Visible CAGR',.000051);
      }
      await page.locator('#dimension').selectOption('quality_growth');
      await page.locator('#statsPeriod').selectOption('available_history');
      await page.locator('#statsScenario').selectOption('cost10_borrow3');
      await page.locator('#statsLeg').selectOption('short');
      await page.locator('#statsMissing').selectOption('adverse_unknown');
      await page.locator('h1').scrollIntoViewIfNeeded();
      await page.screenshot({path:__MAIN_IMAGE__});
      await page.locator('#detailedStatistics').click();
      let linked=await page.evaluate(()=>window.qualityReportSnapshot);
      if(linked.series!=='quality_growth'||linked.leg!=='short'||linked.calendar.length!==175)throw Error('Detailed link state');
      let tableCells=0,icCells=0;
      for(const c of __CASES__){
        await page.goto(__ORIGIN__+c.file);
        await page.locator('#reset').click();
        await page.locator('#focal').selectOption(c.signal);
        await page.locator('#legMode').selectOption(c.leg);
        await page.locator('#from').fill(c.start);await page.locator('#to').fill(c.end);
        for(const [key,value]of Object.entries(c.filters))await page.locator('#filter-'+key).selectOption(String(value));
        const result=await page.evaluate(()=>window.qualityReportSnapshot);
        for(const [key,source]of Object.entries({cagr:'cagr',vol:'annual_vol',sharpe:'sharpe_rf',dd:'monthly_max_drawdown'}))close(result.metrics[key],c.expected[source],'KPI '+c.signal+' '+key);
        if(result.activeMonths!==c.expected.active||result.cashMonths!==c.expected.cash)throw Error('Availability counts');
        if(JSON.stringify(result.selected)!==JSON.stringify(c.active_dates))throw Error('Selected formation dates');
        for(const [signal,wanted]of Object.entries(c.comparison)){
          const row=page.locator('#comparisonBody tr[data-signal="'+signal+'"]');
          for(const [key,source]of Object.entries({return:'cagr',vol:'annual_vol',sharpe:'sharpe_rf',dd:'monthly_max_drawdown'})){
            close(Number(await row.locator('[data-metric="'+key+'"]').getAttribute('data-value')),wanted[source],'Comparison '+signal+' '+key);tableCells++;
          }
        }
        for(const signal of __SIGNALS__){
          const wanted=c.expected_ic[signal]||{spearman:null,hac_t:null,cosine_raw:null,cosine_rank:null,months:0};
          const row=page.locator('#icBody tr[data-ic-signal="'+signal+'"]');
          for(const key of ['spearman','hac_t','cosine_raw','cosine_rank']){
            close(Number(await row.locator('[data-ic="'+key+'"]').getAttribute('data-value')),wanted[key],'IC '+signal+' '+key);icCells++;
          }
          if(Number(await row.locator('[data-ic="months"]').textContent())!==wanted.months)throw Error('IC months');
        }
        if(await page.locator('#performanceChart').getAttribute('data-x-scale')!=='calendar')throw Error('Time axis');
      }
      await page.goto(__ORIGIN__+'statistics__cost10_borrow3__adverse_unknown.html');
      await page.locator('#reset').click();
      await page.locator('#focal').selectOption('quality_growth');await page.locator('#legMode').selectOption('short');
      await page.locator('[data-preset="spy_vol21"]').click();
      const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadPromise;
      await download.saveAs(__EXPORT__);
      if(!download.suggestedFilename().includes('cost10_borrow3_adverse_unknown'))throw Error('CSV scenario filename');
      await page.locator('#reset').click();
      await page.locator('#focal').selectOption('profitability_broad');
      await page.locator('#legMode').selectOption('long');await page.locator('#addComparison').click();
      await page.locator('#legMode').selectOption('short');await page.locator('#addComparison').click();
      await page.locator('#legMode').selectOption('long_short');await page.locator('[data-preset="spy_vol21"]').click();await page.locator('#addComparison').click();
      const comparisons=await page.evaluate(()=>window.qualityReportSnapshot.comparisons);
      if(comparisons.length!==3||comparisons[0].leg!=='long'||comparisons[1].leg!=='short')throw Error('Saved side comparisons');
      await page.reload();if((await page.evaluate(()=>window.qualityReportSnapshot.comparisons)).length!==3)throw Error('Hash restore');
      await page.locator('h1').scrollIntoViewIfNeeded();await page.screenshot({path:__DASH_IMAGE__});
      await page.setViewportSize({width:390,height:844});await page.locator('h1').scrollIntoViewIfNeeded();
      if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error('Dashboard mobile overflow');
      await page.screenshot({path:__MOBILE_IMAGE__});
      await page.goto(__ORIGIN__+'report.html');
      if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error('Main mobile overflow');
      if(errors.length)throw Error(errors.join('\n'));
      return {status:'passed',main_cases:__MAIN_COUNT__,dashboard_cases:__COUNT__,comparison_numeric_cells:tableCells,ic_numeric_cells:icCells,rf_export:download.suggestedFilename(),saved_comparisons:3,calendar_axis:true,mobile_overflow:false,page_errors:errors};
    }'''
    replacements = {'__ORIGIN__': ORIGIN, '__MAIN_CASES__': main_cases, '__CASES__': cases,
        '__SIGNALS__': analysis.SIGNALS, '__MAIN_COUNT__': len(main_cases), '__COUNT__': len(cases),
        '__MAIN_IMAGE__': str(ARTIFACTS / 'quality-dimension-statistics-main.png'),
        '__DASH_IMAGE__': str(ARTIFACTS / 'quality-dimension-statistics-dashboard.png'),
        '__MOBILE_IMAGE__': str(ARTIFACTS / 'quality-dimension-statistics-mobile.png'),
        '__EXPORT__': str(ARTIFACTS / 'quality-dimension-statistics-export.csv')}
    for token, value in replacements.items():
        script = script.replace(token, json.dumps(clean(value), ensure_ascii=False, allow_nan=False))
    path = ARTIFACTS / 'verify-quality-dimension-statistics.js'
    path.write_text(script, encoding='utf-8')
    packages = Path(os.environ['LOCALAPPDATA']) / 'npm-cache/_npx'
    package = max(packages.glob('*/node_modules/@playwright/cli/package.json'), key=lambda p: p.stat().st_mtime)
    cli = package.parent / 'playwright-cli.js'
    result = subprocess.run(['node', str(cli), '--session', 'quality_dimension_stats', 'run-code', '--filename', str(path)],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=240)
    assert result.returncode == 0 and '### Error' not in result.stdout, result.stdout
    receipt = json.loads(result.stdout.split('### Result\n', 1)[1].split('\n### Ran Playwright code', 1)[0].strip())
    exported = pd.read_csv(ARTIFACTS / 'quality-dimension-statistics-export.csv')
    expected = monthly[(monthly.scenario == 'cost10_borrow3') & (monthly.missing_policy == 'adverse_unknown') & (monthly.leg == 'short')]
    for signal, group in expected.groupby('signal'):
        active = group.indicator_available & group.common_sample & group.spy_vol21_state.eq(1)
        values = np.where(active, group['return'], group.risk_free)
        np.testing.assert_allclose(exported[signal], values, atol=1e-12)
        np.testing.assert_array_equal(exported[signal + '_active'], active.astype(int))
    assert len(exported) == 235 and exported.portfolio_side.eq('short').all()
    receipt.update(csv_rows_verified=len(exported), csv_dimension_values_verified=len(exported) * len(analysis.SIGNALS))
    (OUT / 'statistics_ui_validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
