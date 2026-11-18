"""Compare real browser-filtered numbers with independent pandas calculations.

Uses an existing Playwright CLI session pointed at the built local report.
"""
import json
import os
import re
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd

from build_interactive_report import ROOT,OUT,REGIMES,SERIES,RF_FILE


def main():
    artifact=ROOT/'output/playwright'
    artifact.mkdir(exist_ok=True,parents=True)
    cases=[dict(name='all',filters={})]
    for r in REGIMES:
        for level in [0,1]: cases.append(dict(name=r['id']+'_'+str(level),filters={r['id']:level}))
    cases += [dict(name='vol_and_downtrend',filters={'spy_vol21':1,'spy_trend126':0}),
              dict(name='vol_breadth_correlation',filters={'spy_vol21':1,'breadth126':0,'spy_tlt_corr63':1}),
              dict(name='late',filters={},start='2017-01'),
              dict(name='late_high_vol',filters={'spy_vol21':1},start='2017-01'),
              dict(name='short_date_window',filters={},start='2008-01',end='2012-12'),
              dict(name='empty_regime_window',filters={'spy_vol21':1},end='2007-12'),
              dict(name='single_month',filters={},start='2007-02',end='2007-02'),
              dict(name='invalid_range',filters={},start='2026-08',end='2007-02'),
              dict(name='rmw_high_vol',filters={'spy_vol21':1},focal='public_rmw'),
              dict(name='fcf_low_vol',filters={'spy_vol21':0},focal='fcf_assets_ttm')]
    js='''async (page) => {
      const cases=__CASES__;
      await page.setViewportSize({width:1440,height:1080});
      await page.reload();
      const results=[];
      for(const c of cases){
        await page.locator('#reset').click();
        await page.locator('#focal').selectOption(c.focal||'portfolio4');
        if(c.start)await page.locator('#from').fill(c.start);
        if(c.end)await page.locator('#to').fill(c.end);
        for(const [id,value] of Object.entries(c.filters))await page.locator('#filter-'+id).selectOption(String(value));
        results.push(await page.evaluate(name=>({name,
          snapshot:window.qualityReportSnapshot,
          returnLabel:document.getElementById('returnLabel').textContent,
          displayedReturn:Number(document.getElementById('returnValue').dataset.value),
          disabledExport:document.getElementById('export').disabled,
          alert:document.getElementById('sampleAlert').hidden?'':document.getElementById('sampleAlert').textContent,
          table:[...document.querySelectorAll('#comparisonBody tr')].map(row=>({
            id:row.dataset.signal,
            values:Object.fromEntries([...row.querySelectorAll('[data-metric]')].map(td=>[td.dataset.metric,td.dataset.value===''?null:Number(td.dataset.value)]))})),
          ic:[...document.querySelectorAll('#icBody tr')].map(row=>[...row.cells].map(td=>td.textContent)),
          options:[...document.querySelectorAll('.regime-fields select')].map(s=>s.options.length)
        }),c.name));
      }
      await page.goto('http://127.0.0.1:8767/report.html');
      await page.locator('[data-preset="spy_vol21"]').click();
      const before=await page.evaluate(()=>window.qualityReportSnapshot);
      const filterURL=page.url();
      await page.locator('.method-link').click();
      await page.locator('.brand').click();
      const navigationPreservesFilters=filterURL===page.url();
      await page.reload();
      const after=await page.evaluate(()=>window.qualityReportSnapshot);
      const downloadPromise=page.waitForEvent('download');
      await page.locator('#export').click();
      const download=await downloadPromise;
      await download.saveAs('output/playwright/export-high-vol.csv');
      await page.locator('[data-chart="drawdown"]').click();
      const ddTitle=await page.locator('#chartTitle').textContent();
      await page.locator('[data-chart="equity"]').click();
      await page.locator('[data-series="fcf_assets_ttm"]').click();
      const legend=await page.locator('[data-series="fcf_assets_ttm"]').isChecked();
      await page.locator('[data-series="fcf_assets_ttm"]').click();
      await page.locator('#performanceChart').focus();
      await page.keyboard.press('ArrowRight');
      const tooltip=await page.locator('#chartTooltip').isVisible();
      await page.locator('#seriesLegend').hover();
      const desktopOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      await page.screenshot({path:'output/playwright/portfolio-high-vol.png'});
      await page.setViewportSize({width:390,height:844});
      await page.reload();
      const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      await page.screenshot({path:'output/playwright/portfolio-mobile.png'});
      await page.locator('.filter-panel>summary').click();
      await page.locator('#filter-spy_vol21').selectOption('0');
      const mobileLowVol=await page.evaluate(()=>window.qualityReportSnapshot.activeMonths);
      await page.locator('.filter-panel>summary').click();
      await page.setViewportSize({width:1440,height:1080});
      await page.goto('http://127.0.0.1:8767/report.html');
      await page.screenshot({path:'output/playwright/portfolio-desktop.png'});
      // Calendar geometry, multiple factors and snapshots with independent filters.
      const inspectChart=async()=>await page.evaluate(()=>({
        scale:document.getElementById('performanceChart').dataset.xScale,
        start:document.getElementById('performanceChart').dataset.domainStart,
        end:document.getElementById('performanceChart').dataset.domainEnd,
        width:document.getElementById('performanceChart').viewBox.baseVal.width,
        ticks:[...document.querySelectorAll('[data-axis="time"]')].map(e=>e.textContent),
        paths:[...document.querySelectorAll('[data-chart-series]')].map(e=>({
          key:e.dataset.chartSeries,id:e.dataset.signal,segments:Number(e.dataset.segments),
          points:Number(e.dataset.points),final:Number(e.dataset.finalValue),path:e.getAttribute('d')})),
        snapshot:window.qualityReportSnapshot
      }));
      await page.locator('#showAllSeries').click();
      const allSeries=await inspectChart();
      await page.locator('[data-preset="spy_vol21"]').click();
      const highCalendar=await inspectChart();
      await page.locator('#addComparison').click();
      await page.locator('#addComparison').click();
      const duplicateCount=await page.locator('.saved-series-chip').count();
      await page.locator('[data-preset="all"]').click();
      const savedHighOnAll=await inspectChart();
      await page.locator('#filter-spy_vol21').selectOption('0');
      await page.locator('#addComparison').click();
      await page.locator('[data-preset="all"]').click();
      const eightCurves=await inspectChart();
      await page.reload();
      const restoredCurves=await inspectChart();
      await page.locator('#showFocalSeries').click();
      await page.locator('#performanceChart').screenshot({path:'output/playwright/portfolio-calendar-chart.png'});
      await page.locator('.chart-panel').screenshot({path:'output/playwright/portfolio-comparisons.png'});
      await page.locator('[data-chart="drawdown"]').click();
      const comparisonDrawdown=await inspectChart();
      await page.locator('[data-chart="equity"]').click();
      await page.locator('#from').fill('2020-01');
      const shiftedDates=await inspectChart();
      await page.locator('[data-remove-comparison="0"]').click();
      const afterRemoval=await inspectChart();
      await page.setViewportSize({width:390,height:844});
      await page.reload();
      const comparisonMobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      await page.screenshot({path:'output/playwright/portfolio-comparisons-mobile.png'});
      await page.setViewportSize({width:1440,height:1080});
      await page.goto('http://127.0.0.1:8767/report.html');
      return {results,before,after,ddTitle,legend,tooltip,desktopOverflow,mobileOverflow,mobileLowVol,navigationPreservesFilters,
        allSeries,highCalendar,duplicateCount,savedHighOnAll,eightCurves,restoredCurves,comparisonDrawdown,shiftedDates,afterRemoval,comparisonMobileOverflow};
    }'''.replace('__CASES__',json.dumps(cases))
    jsfile=artifact/'verify-dashboard.js'
    jsfile.write_text(js,encoding='utf-8')
    cache=Path(os.environ.get('LOCALAPPDATA',''))/'npm-cache/_npx'
    packages=list(cache.glob('*/node_modules/@playwright/cli/package.json'))
    assert packages,'Initialize Playwright CLI with the skill wrapper first'
    package=max(packages,key=lambda p:p.stat().st_mtime)
    cli=package.parent/json.loads(package.read_text())['bin']['playwright-cli']
    command=['node',str(cli),'--session','quality_report','run-code','--filename',str(jsfile)]
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=100)
    if completed.returncode or '### Error' in completed.stdout:
        raise RuntimeError(completed.stdout+'\n'+completed.stderr)
    raw=completed.stdout.split('### Result\n',1)[1].split('\n### Ran Playwright code',1)[0].strip()
    result=json.loads(raw)
    (artifact/'browser-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    source=pd.read_csv(OUT/'portfolio_monthly.csv',parse_dates=['signal_date','entry_date','exit_date'])
    features=pd.read_csv(ROOT/'results/regimes/regime_features.csv',parse_dates=['signal_date'])
    data=source.merge(features.drop(columns=['entry_date','exit_date']),on='signal_date',validate='one_to_one')
    rf=pd.read_csv(OUT/'risk_free_windows.csv',parse_dates=['signal_date'])
    data=data.merge(rf[['signal_date','risk_free']],on='signal_date',validate='one_to_one')
    # Independent ZIP parsing and cumulative log returns check every RF boundary.
    with zipfile.ZipFile(RF_FILE) as archive:
        text=archive.read(archive.namelist()[0]).decode('utf-8-sig')
    lines=[line.split(',') for line in text.splitlines() if re.match(r'^\d{8},',line)]
    daily_rf=pd.Series([float(line[-1])/100 for line in lines],
                       index=pd.to_datetime([line[0] for line in lines],format='%Y%m%d'))
    log_index=np.log1p(daily_rf).cumsum()
    independent=np.expm1(log_index.reindex(data.exit_date).to_numpy()-log_index.reindex(data.entry_date).to_numpy())
    np.testing.assert_allclose(independent,data.risk_free,atol=1e-12)
    for row in rf.itertuples():
        assert len(daily_rf.loc[(daily_rf.index>pd.Timestamp(row.entry_date)) & (daily_rf.index<=pd.Timestamp(row.exit_date))])==row.daily_observations
    data['month']=data.entry_date.dt.strftime('%Y-%m')
    cells=0
    for case,browser in zip(cases,result['results']):
        period=data[data.month.between(case.get('start','2007-02'),case.get('end','2026-08'))]
        selected=period
        if case['filters']: selected=selected[selected.common_sample]
        for key,value in case['filters'].items(): selected=selected[selected[key+'_state'].eq(value)]
        assert browser['snapshot']['selected']==selected.signal_date.dt.strftime('%Y-%m-%d').tolist(),case['name']
        assert browser['snapshot']['calendar']==period.signal_date.dt.strftime('%Y-%m-%d').tolist()
        assert browser['snapshot']['activeMonths']==len(selected)
        assert browser['snapshot']['cashMonths']==len(period)-len(selected)
        assert browser['snapshot']['metrics']['n']==len(period)
        assert browser['snapshot']['conditional']==bool(case['filters'])
        assert browser['options']==[3]*6
        assert browser['disabledExport']==period.empty
        if len(selected)<24: assert browser['alert']
        assert 'CAGR' in browser['returnLabel']
        if case['filters']: assert 'RF' in browser['returnLabel']
        for series,shown in zip(SERIES,browser['table']):
            assert series['id']==shown['id']
            active=period.index.isin(selected.index)
            y=period[series['id']].where(active,period.risk_free)
            if y.empty:
                assert all(v is None for v in shown['values'].values());continue
            equity=np.r_[1,(1+y).cumprod().to_numpy()]
            years=(period.exit_date.iloc[-1]-period.entry_date.iloc[0]).days/365.25
            annual=equity[-1]**(1/years)-1
            vol=y.std(ddof=1)*np.sqrt(12)
            excess=y-period.risk_free
            excess_vol=excess.std(ddof=1)*np.sqrt(12)
            sharpe=excess.mean()*12/excess_vol if excess_vol>1e-12 else np.nan
            dd=(equity/np.maximum.accumulate(equity)-1).min()
            for key,expected in dict(return_=annual,vol=vol,sharpe=sharpe,dd=dd).items():
                key='return' if key=='return_' else key
                actual=shown['values'][key]
                if np.isfinite(expected): np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10,err_msg=case['name']+' '+series['id']+' '+key)
                else: assert actual is None
                cells+=1
    assert result['before']==result['after'],'Filter hash restoration'
    assert result['navigationPreservesFilters']
    assert result['legend'] is True and result['tooltip']
    assert 'drawdown' in result['ddTitle']
    assert not result['desktopOverflow'] and not result['mobileOverflow']
    assert result['mobileLowVol']==135
    exported=pd.read_csv(artifact/'export-high-vol.csv')
    expected=data[data.common_sample & data.spy_vol21_state.eq(1)]
    assert len(exported)==len(data)==235 and len(expected)==88
    assert exported.factor_active.sum()==88
    np.testing.assert_allclose(exported.risk_free,data.risk_free,atol=1e-12)
    np.testing.assert_allclose(exported.loc[exported.factor_active.eq(1),'portfolio4'],expected.portfolio4,atol=1e-12)
    np.testing.assert_allclose(exported.loc[exported.factor_active.eq(0),'portfolio4'],data.loc[~data.index.isin(expected.index),'risk_free'],atol=1e-12)
    np.testing.assert_allclose(exported.raw_portfolio4,data.portfolio4,atol=1e-12)
    assert exported.cash_reason.eq('regime_unavailable').sum()==12
    # Calendar paths now include all cash months and have no omitted-month gaps.
    def verify_calendar(chart,expected_rows,key,drawdown=False):
        assert chart['scale']=='calendar'
        assert all(re.fullmatch(r'\d{4}',t) for t in chart['ticks'])
        path=next(p for p in chart['paths'] if p['key']==key)
        selected=expected_rows.reset_index(drop=True)
        runs=1+sum(selected.exit_date.iloc[i-1]!=selected.entry_date.iloc[i] for i in range(1,len(selected)))
        assert path['segments']==runs and path['points']==len(selected)+runs
        assert path['path'].count('M')==runs
        points=[]
        for i,row in selected.iterrows():
            if i==0 or selected.exit_date.iloc[i-1]!=row.entry_date: points.append(row.entry_date)
            points.append(row.exit_date)
        coordinates=re.findall(r'[ML]([\d.]+),([\d.]+)',path['path'])
        assert len(coordinates)==len(points)
        start,end=pd.Timestamp(chart['start']),pd.Timestamp(chart['end'])
        width=chart['width']-45-14
        for point,(actual_x,_) in zip(points,coordinates):
            expected_x=45+(point-start).total_seconds()/(end-start).total_seconds()*width
            np.testing.assert_allclose(float(actual_x),expected_x,atol=.0051)
        values=np.r_[1,(1+selected[path['id']]).cumprod()]
        final=(values/np.maximum.accumulate(values)-1)[-1] if drawdown else values[-1]
        np.testing.assert_allclose(path['final'],final,atol=1e-12)
    assert len(result['allSeries']['paths'])==len(SERIES)==7
    assert result['duplicateCount']==1
    verify_calendar(result['allSeries'],data,'portfolio4')
    high_path=data.copy()
    for series in SERIES: high_path[series['id']]=data[series['id']].where(data.index.isin(expected.index),data.risk_free)
    verify_calendar(result['highCalendar'],high_path,'portfolio4')
    verify_calendar(result['savedHighOnAll'],high_path,'comparison-0')
    assert result['highCalendar']['paths'][0]['segments']==1
    assert len(result['eightCurves']['paths'])==9
    assert result['eightCurves']['snapshot']==result['restoredCurves']['snapshot']
    low=data[data.common_sample & data.spy_vol21_state.eq(0)]
    low_path=data.copy()
    for series in SERIES: low_path[series['id']]=data[series['id']].where(data.index.isin(low.index),data.risk_free)
    verify_calendar(result['restoredCurves'],low_path,'comparison-1')
    verify_calendar(result['comparisonDrawdown'],high_path,'comparison-0',drawdown=True)
    verify_calendar(result['shiftedDates'],high_path,'comparison-0')
    assert len(result['afterRemoval']['snapshot']['comparisons'])==1
    verify_calendar(result['afterRemoval'],low_path,'comparison-0')
    assert not result['comparisonMobileOverflow']
    # Filtered IC table independently reconciles the three primary signals.
    diagnostic=pd.read_csv(ROOT/'results/initial_analysis/monthly_diagnostics.csv',parse_dates=['signal_date'])
    high=next(x for x in result['results'] if x['name']=='spy_vol21_1')
    names=['FCF / imovina · TTM','CFO / imovina · TTM','Bruto dobit / imovina · TTM']
    for signal,name in zip(['fcf_assets_ttm','cfo_assets_ttm','gp_assets_ttm'],names):
        original=diagnostic[(diagnostic['sample']=='own') & diagnostic.signal.eq(signal) & diagnostic.signal_date.isin(expected.signal_date)]
        row=next(x for x in high['ic'] if x[0]==name)
        for position,metric in [(1,'ic_spearman'),(2,'ic_cosine_raw'),(3,'ic_cosine_rank_centered')]:
            actual=float(row[position].replace('\u2212','-').replace(',','.'))
            np.testing.assert_allclose(actual,original[metric].mean(),atol=.000051)
    validation=dict(status='passed',browser_filter_cases=len(cases),numeric_table_cells=cells,
        reference='Independent pandas calculations from original monthly returns and formation-date regimes',
        checks=['all six regime controls, both states','combined filters','date ranges','empty and single-month selections',
            'series switching','filter restoration after reload and section navigation','CSV download and data reconciliation','IC means',
            'drawdown toggle','series legend','keyboard chart tooltip','desktop and mobile layout without page overflow',
            'seven simultaneous series including RF benchmark','saved regime comparisons and duplicate prevention','nine simultaneous lines',
            'calendar x coordinates and year labels','continuous cash-inclusive paths','comparison compounding and drawdown',
            'independent daily RF window compounding for all 235 windows','cash-only cases and undefined RF-only Sharpe',
            'no RF double counting in active months','RF-relative Sharpe','warmup months held in RF',
            'saved comparisons survive filter changes and reload','comparison removal and mobile layout'],
        screenshots=['output/playwright/portfolio-desktop.png','output/playwright/portfolio-high-vol.png','output/playwright/portfolio-mobile.png'])
    (OUT/'interactive_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation),flush=True)


if __name__=='__main__': main()
