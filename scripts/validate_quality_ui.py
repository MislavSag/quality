"""Check renamed report navigation and literature filters in a real browser."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    sources=json.loads((ROOT/'data/reference/quality_literature.json').read_text(encoding='utf-8'))['sources']
    cases=[dict(category='',dimension='',query='')]
    cases += [dict(category=c,dimension='',query='') for c in sorted({s['category'] for s in sources})]
    cases += [dict(category='',dimension=d,query='') for d in sorted({d for s in sources for d in s['dimensions']})]
    cases += [dict(category='Akademski rad',dimension='Sigurnost',query=''),dict(category='',dimension='',query='qmj'),dict(category='',dimension='',query='xyz-no-results')]
    for c in cases:
        c['expected']=sorted(s['id'] for s in sources if (not c['category'] or s['category']==c['category']) and
            (not c['dimension'] or c['dimension'] in s['dimensions']) and
            (not c['query'] or c['query'] in ' '.join([s['title'],s['authors'],s['summary'],*s['dimensions']]).lower()))
    artifacts=ROOT/'output/playwright';artifacts.mkdir(parents=True,exist_ok=True)
    js=r'''async page => {
      const errors=[];page.on('pageerror',e=>errors.push(String(e)));
      await page.setViewportSize({width:1440,height:1080});
      await page.goto('http://127.0.0.1:8768/quality_literature/report.html');
      for(const c of __CASES__){
        await page.locator('#category').selectOption(c.category);
        await page.locator('#dimension').selectOption(c.dimension);
        await page.locator('#query').fill(c.query);
        const visible=await page.locator('.source-card:visible').evaluateAll(nodes=>nodes.map(n=>n.id).sort());
        if(JSON.stringify(visible)!==JSON.stringify(c.expected))throw Error('Literature filter '+JSON.stringify(c));
      }
      await page.locator('#reset').click();
      if(await page.locator('.source-card:visible').count()!==16)throw Error('Reset');
      await page.locator('h1').scrollIntoViewIfNeeded();
      await page.screenshot({path:__OVERVIEW__});
      await page.locator('#category').scrollIntoViewIfNeeded();
      await page.screenshot({path:__FILTERS__});
      await page.setViewportSize({width:390,height:844});
      await page.locator('h1').scrollIntoViewIfNeeded();
      const literatureOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      if(literatureOverflow)throw Error('Literature mobile overflow');
      await page.screenshot({path:__MOBILE__});
      await page.setViewportSize({width:1440,height:1080});
      await page.getByRole('link',{name:'FMP izvještaj s long i short prikazom'}).click();
      if(!(await page.title()).startsWith('Quality'))throw Error('FMP title');
      if(!(await page.locator('h1').textContent()).startsWith('Quality:'))throw Error('FMP heading');
      const baseline=await page.evaluate(()=>({canonical:window.qualityReportSnapshot.metrics,legacy:window.profitabilityReportSnapshot.metrics}));
      if(JSON.stringify(baseline.canonical)!==JSON.stringify(baseline.legacy))throw Error('Legacy API');
      if(Math.abs(baseline.canonical.cagr-.039865)>1e-6)throw Error('Changed FMP baseline');
      await page.locator('#legMode').selectOption('long');
      await page.locator('#addComparison').click();
      await page.locator('#legMode').selectOption('short');
      const side=await page.evaluate(()=>window.qualityReportSnapshot);
      if(side.leg!=='short'||side.comparisons[0].leg!=='long')throw Error('Side comparison');
      await page.locator('[data-preset="spy_vol21"]').click();
      const gated=await page.evaluate(()=>window.qualityReportSnapshot);
      if(gated.activeMonths!==88||gated.cashMonths!==147)throw Error('RF calendar');
      const dl=page.waitForEvent('download');await page.locator('#export').click();const download=await dl;
      if(!download.suggestedFilename().startsWith('quality_'))throw Error('Export name');
      await page.locator('h1').scrollIntoViewIfNeeded();
      await page.screenshot({path:__FMP__});
      await page.goto('http://127.0.0.1:8768/factor_portfolio/report.html');
      const original=await page.evaluate(()=>window.qualityReportSnapshot.metrics.cagr);
      if(Math.abs(original-.0275696)>1e-6)throw Error('Changed original portfolio');
      if(errors.length)throw Error(errors.join('\n'));
      return {status:'passed',literature_filter_cases:__N__,literature_sources:16,mobile_overflow:literatureOverflow,
        original_portfolio_cagr:original,fmp_cagr:baseline.canonical.cagr,legacy_api_compatible:true,
        side_comparison_preserved:true,rf_cash_months:gated.cashMonths,export_name:download.suggestedFilename(),page_errors:errors};
    }'''
    replacements={'__CASES__':cases,'__N__':len(cases),'__OVERVIEW__':str(artifacts/'quality-literature-overview.png'),
        '__FILTERS__':str(artifacts/'quality-literature-filters.png'),'__MOBILE__':str(artifacts/'quality-literature-mobile.png'),'__FMP__':str(artifacts/'quality-fmp.png')}
    for key,value in replacements.items():js=js.replace(key,json.dumps(value))
    script=artifacts/'verify-quality-rename.js';script.write_text(js,encoding='utf-8')
    cache=Path(os.environ['LOCALAPPDATA'])/'npm-cache/_npx'
    package=max(cache.glob('*/node_modules/@playwright/cli/package.json'),key=lambda p:p.stat().st_mtime)
    cli=package.parent/'playwright-cli.js'
    result=subprocess.run(['node',str(cli),'--session','quality_review','run-code','--filename',str(script)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=100)
    assert result.returncode==0 and '### Error' not in result.stdout,result.stdout
    parsed=json.loads(result.stdout.split('### Result\n',1)[1].split('\n### Ran Playwright code',1)[0].strip())
    (ROOT/'results/quality_ui_validation.json').write_text(json.dumps(parsed,indent=2))
    print(json.dumps(parsed),flush=True)


if __name__=='__main__':main()
