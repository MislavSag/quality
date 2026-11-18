"""Real-browser verification of standalone FMP report calculations and filters."""
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from analyze_joint_fmp import OUT,ROOT,CFG
from build_joint_fmp_report import filename


def main():
    pages={}
    for model in [a+'__'+b for a in CFG['transformations'] for b in CFG['weighting']]:
        for scenario in ['gross','cost10_borrow3']:
            name=filename(model,scenario)
            content=(OUT/name).read_text(encoding='utf-8')
            pages[name]=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',content,re.S).group(1))
    cases=[dict(page='report.html',filters={}),dict(page='report.html',filters={'spy_vol21':1}),
           dict(page='report.html',filters={'spy_vol21':0}),dict(page='report.html',filters={'spy_trend126':0}),
           dict(page='report.html',filters={'spy_vol21':1,'breadth126':0}),
           dict(page='report.html',filters={'spy_vol21':1},end='2007-12'),
           dict(page='report.html',filters={},start='2017-01'),
           dict(page='report.html',filters={'spy_vol21':1},focal='cfo_assets_ttm'),
           *[dict(page=p,filters={}) for p in pages if p!='report.html'],
           *[dict(page=p,filters={},leg=leg) for p in pages for leg in ['long','short']],
           *[dict(page='report.html',filters={'spy_vol21':1},leg=leg) for leg in ['long','short']],
           dict(page='report.html',filters={'spy_trend126':0,'spy_vol21':1},leg='short'),
           dict(page='report.html',filters={'spy_vol21':1},leg='long',end='2007-12')]
    expected=[]
    for c in cases:
        data=pages[c['page']]
        months=[m for m in data['months'] if c.get('start','2007-02')<=m['month']<=c.get('end','2026-08')]
        active=np.array([not c['filters'] or (m['common'] and all(m['states'][k]==v for k,v in c['filters'].items())) for m in months])
        rf=np.array([m['rf'] for m in months])
        years=(pd.Timestamp(months[-1]['exit'])-pd.Timestamp(months[0]['entry'])).days/365.25
        metrics={}
        leg=c.get('leg','long_short')
        for s in data['series']:
            if leg not in s.get('legs',['long_short']):continue
            r=np.array([(m['r'] if leg=='long_short' else m['legs'][leg])[s['id']] for m in months]);r=np.where(active,r,rf)
            wealth=np.r_[1,np.cumprod(1+r)]
            excess=r-rf;sd=np.std(excess,ddof=1)
            metrics[s['id']]=dict(return_=wealth[-1]**(1/years)-1,vol=np.std(r,ddof=1)*np.sqrt(12),
                sharpe=float(excess.mean()/sd*np.sqrt(12)) if sd>1e-12 else None,
                dd=np.min(wealth/np.maximum.accumulate(wealth)-1))
        expected.append(dict(active=int(active.sum()),cash=int((~active).sum()),metrics=metrics))
    artifact=ROOT/'output/playwright';artifact.mkdir(exist_ok=True,parents=True)
    result_path=artifact/'joint-fmp-browser-result.json'
    js=r'''async (page) => {
      const cases=__CASES__,expected=__EXPECTED__;
      const results=[];let cells=0;
      await page.setViewportSize({width:1440,height:1080});
      for(let i=0;i<cases.length;i++){
        const c=cases[i],e=expected[i];
        await page.goto('http://127.0.0.1:8768/'+c.page);
        if(c.leg)await page.locator('#legMode').selectOption(c.leg);
        if(c.start)await page.locator('#from').fill(c.start);
        if(c.end)await page.locator('#to').fill(c.end);
        if(c.focal)await page.locator('#focal').selectOption(c.focal);
        for(const [k,v] of Object.entries(c.filters))await page.locator('#filter-'+k).selectOption(String(v));
        const got=await page.evaluate(()=>({s:window.qualityReportSnapshot,
          rows:[...document.querySelectorAll('#comparisonBody tr')].map(r=>({id:r.dataset.signal,
            metrics:Object.fromEntries([...r.querySelectorAll('[data-metric]')].map(td=>[td.dataset.metric,td.dataset.value===''?null:Number(td.dataset.value)]))})),
          scale:document.querySelector('#performanceChart').dataset.xScale,
          years:[...document.querySelectorAll('[data-axis="time"]')].map(e=>e.textContent)}));
        if(got.s.activeMonths!==e.active || got.s.cashMonths!==e.cash)throw Error('Regime count '+i);
        if(got.s.leg!==(c.leg||'long_short'))throw Error('Side state '+i);
        if(got.rows.length!==Object.keys(e.metrics).length)throw Error('Unavailable side leaked into table '+i);
        if(got.scale!=='calendar'||!got.years.some(y=>/20\d\d/.test(y)))throw Error('Calendar axis '+i);
        for(const row of got.rows)for(const [key,val] of Object.entries(row.metrics)){
          const ref=e.metrics[row.id][key==='return'?'return_':key];
          if(ref===null ? val!==null : val===null||Math.abs(val-ref)>1e-10)throw Error('Numeric mismatch '+i+' '+row.id+' '+key+' '+val+' '+ref);
          cells++;
        }
        results.push({page:c.page,filters:c.filters,active:e.active,cash:e.cash});
      }
      await page.goto('http://127.0.0.1:8768/report.html');
      await page.locator('[data-preset="spy_vol21"]').click();
      await page.locator('#addComparison').click();
      await page.locator('#reset').click();
      await page.locator('#showAllSeries').click();
      const comparison=await page.evaluate(()=>({saved:window.qualityReportSnapshot.comparisons.length,
        paths:document.querySelectorAll('path[data-chart-series]').length}));
      if(comparison.saved!==1 || comparison.paths!==15)throw Error('Saved comparison');
      const exportPromise=page.waitForEvent('download');await page.locator('#export').click();
      const download=await exportPromise;await download.saveAs(__CSV__);
      await page.locator('#legMode').selectOption('long');
      await page.locator('#addComparison').click();
      await page.locator('#legMode').selectOption('short');
      await page.locator('#addComparison').click();
      await page.locator('#addComparison').click();
      await page.reload();
      const sides=await page.evaluate(()=>window.qualityReportSnapshot);
      if(sides.leg!=='short'||sides.comparisons.length!==3||sides.comparisons[1].leg!=='long'||sides.comparisons[2].leg!=='short')throw Error('Side comparison/hash/duplicate handling');
      await page.locator('[data-preset="spy_vol21"]').click();
      const sideDownloadPromise=page.waitForEvent('download');await page.locator('#export').click();
      const sideDownload=await sideDownloadPromise;await sideDownload.saveAs(__SIDE_CSV__);
      await page.setViewportSize({width:1440,height:1080});
      await page.screenshot({path:__DESKTOP__});
      await page.setViewportSize({width:390,height:844});
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
      if(overflow)throw Error('Mobile horizontal overflow');
      await page.screenshot({path:__MOBILE__});
      const result={status:'passed',browser_cases:results.length,numeric_cells:cells,comparison,side_comparisons:sides.comparisons.length,mobile_overflow:overflow,results};
      return result;
    }'''
    values={'__CASES__':cases,'__EXPECTED__':expected,'__CSV__':str(artifact/'joint-fmp-export.csv'),
            '__SIDE_CSV__':str(artifact/'joint-fmp-side-export.csv'),
            '__DESKTOP__':str(artifact/'joint-fmp-desktop.png'),'__MOBILE__':str(artifact/'joint-fmp-mobile.png'),'__RESULT__':str(result_path)}
    for key,value in values.items():js=js.replace(key,json.dumps(value,allow_nan=False))
    code=artifact/'verify-joint-fmp.js';code.write_text(js,encoding='utf-8')
    cache=Path(os.environ['LOCALAPPDATA'])/'npm-cache/_npx'
    package=max(cache.glob('*/node_modules/@playwright/cli/package.json'),key=lambda p:p.stat().st_mtime)
    cli=package.parent/'playwright-cli.js'
    completed=subprocess.run(['node',str(cli),'--session','joint_fmp','run-code','--filename',str(code)],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
    assert completed.returncode==0,completed.stdout
    assert '### Error' not in completed.stdout,completed.stdout
    raw=completed.stdout.split('### Result\n',1)[1].split('\n### Ran Playwright code',1)[0].strip()
    result=json.loads(raw)
    export=pd.read_csv(artifact/'joint-fmp-export.csv')
    assert len(export)==235 and export.factor_active.eq(1).all()
    result['csv_rows']=len(export)
    side_export=pd.read_csv(artifact/'joint-fmp-side-export.csv')
    assert len(side_export)==235 and side_export.portfolio_side.eq('short').all()
    assert side_export.factor_active.sum()==88
    assert 'public_rmw' not in side_export.columns
    active=side_export.factor_active.eq(1)
    np.testing.assert_allclose(side_export.loc[~active,'portfolio3_gross2'],side_export.loc[~active,'risk_free'],atol=1e-12)
    np.testing.assert_allclose(side_export.loc[active,'portfolio3_gross2'],side_export.loc[active,'raw_portfolio3_gross2'],atol=1e-12)
    result['side_csv_rows']=len(side_export)
    (OUT/'browser_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['status','browser_cases','numeric_cells','csv_rows','mobile_overflow']}),flush=True)


if __name__=='__main__':main()
