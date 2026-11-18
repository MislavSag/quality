"""Offline return/regime dashboards for quality composites, reusing proven assets."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import analyze_quality_dimensions as analysis
from build_interactive_report import ASSETS, REGIMES
from build_quality_dimensions import SPECS

ROOT, OUT = analysis.ROOT, analysis.OUT
COLORS = ['#365eea', '#4c9f89', '#b17b52', '#8b6cb5', '#c17896', '#719846', '#bf8e3d', '#8293b5']
SHORT = ['Profit 4', 'Kval. dobiti', 'Fin. sigurnost', 'Stabilnost', 'Rast 5Y', 'Investiranje', 'Payout', 'Profit 3']


def replace_function(javascript, name, body):
    pattern = r'  function ' + name + r'\([^\n]*\) \{.*?(?=\n  function )'
    javascript, count = re.subn(pattern, lambda _: body.rstrip() + '\n', javascript, count=1, flags=re.S)
    assert count == 1, name
    return javascript


def local_javascript():
    js = (ASSETS / 'portfolio.js').read_text(encoding='utf-8')
    js = js.replace("visible:new Set(['portfolio4','public_rmw'])", "visible:new Set(['profitability','profitability_broad','earnings_quality'])")
    js = js.replace("'portfolio4'", "'profitability'")
    js = js.replace('<strong>25%</strong>', '<strong>${s.componentCount} mjere</strong>')
    js = replace_function(js, 'strategyRows', '''  function strategyRows(selection=state) {
    const gated=Object.keys(selection.filters).length>0;
    const selectedId=selection.series || selection.focal || state.focal;
    return months.filter(m=>m.month>=selection.from && m.month<=selection.to).map(m=>{
      const gateActive=!gated || (m.common && regimes.every(r=>selection.filters[r.id]===undefined || m.states[r.id]===selection.filters[r.id]));
      const raw=returnsFor(m,selection),activeBySeries={};
      const r=Object.fromEntries(availableSeries(selection).map(s=>{
        const ready=s.id==='risk_free' || m.available[s.id];
        activeBySeries[s.id]=gateActive && ready;
        return [s.id,gateActive && ready?raw[s.id]:m.rf];
      }));
      const active=Boolean(activeBySeries[selectedId]);
      const cashReason=active?'':!m.available[selectedId]?'indicator_unavailable':!m.common?'regime_unavailable':'outside_regime';
      return {...m,active,activeBySeries,cashReason,r};
    });
  }''')
    js = replace_function(js, 'renderTable', '''  function renderTable() {
    $('tableReturnLabel').textContent='CAGR';$('drawdownLabel').textContent='mjesečni krajevi';
    $('comparisonBody').innerHTML=availableSeries().map(s=>{
      const m=calculate(currentStrategyRows,s.id),active=currentStrategyRows.filter(row=>row.activeBySeries[s.id]).length;
      return `<tr class="${s.id===state.focal?'focal-row':''}" data-signal="${s.id}"><td><span class="table-series"><i class="series-dot" style="background:${s.color}"></i><strong>${seriesLabel(s)}</strong></span></td><td data-metric="return" data-value="${m.cagr}">${pct(m.cagr)}</td><td data-metric="vol" data-value="${m.vol}">${pct(m.vol)}</td><td data-metric="sharpe" data-value="${m.sharpe}">${fmt(m.sharpe)}</td><td data-metric="dd" data-value="${m.dd}">${pct(m.dd)}</td><td>${pct(m.hit,1)}</td><td data-metric="active">${s.id==='risk_free'?'RF':active+' / '+currentStrategyRows.length}</td></tr>`;
    }).join('');
    $('tableNote').textContent='Cijeli odabrani kalendar. RF u mjesecima bez indikatora i izvan odabranog režima. Dimenzije imaju različite potpune univerzume. Troškovi režimskog uključivanja nisu ponovno izračunati.';
  }''')
    js = replace_function(js, 'renderCorrelation', '''  function renderCorrelation() {
    const components=availableSeries().filter(s=>s.component),selected=filterRows();
    let table='<table class="correlation-table"><thead><tr><th></th>'+components.map(s=>`<th>${s.short}</th>`).join('')+'</tr></thead><tbody>';
    for(const a of components){table+=`<tr><td>${a.short}</td>`;for(const b of components){
      const usable=selected.filter(m=>m.available[a.id] && m.available[b.id]);
      const value=correlation(usable.map(m=>m.r[a.id]),usable.map(m=>m.r[b.id]));
      const color=Number.isFinite(value)?`rgba(111,145,228,${Math.abs(value)*.4})`:'#f7f8fb';
      table+=`<td title="${usable.length} zajedničkih aktivnih mjeseci" style="background:${color}">${fmt(value)}</td>`;
    }table+='</tr>';}$('correlationMatrix').innerHTML=table+'</tbody></table>';
  }''')
    js = replace_function(js, 'renderIC', '''  function renderIC() {
    const selected=filterRows();
    function hac(values){const x=values.filter(Number.isFinite),n=x.length;if(n<12||sd(x)<1e-15)return NaN;
      const avg=mean(x),e=x.map(v=>v-avg);let sum=e.reduce((a,v)=>a+v*v,0);
      for(let lag=1;lag<=3;lag++){let cross=0;for(let i=lag;i<n;i++)cross+=e[i]*e[i-lag];sum+=2*(1-lag/4)*cross;}
      return sum>0?avg/(Math.sqrt(sum)/n):NaN;}
    $('icBody').innerHTML=DATA.ic.signals.map((signal,j)=>{
      const records=selected.map(m=>DATA.ic.values[m.i][j]).filter(Boolean),average=k=>mean(records.map(r=>r[k]).filter(Number.isFinite));
      const t=hac(records.map(r=>r[2])),known=records.filter(r=>Number.isFinite(r[2])).length;
      return `<tr data-ic-signal="${signal.id}"><td>${signal.label}</td><td data-ic="spearman" data-value="${average(2)}">${fmt(average(2),4)}</td><td data-ic="hac_t" data-value="${t}">${fmt(t)}</td><td data-ic="cosine_raw" data-value="${average(3)}">${fmt(average(3),4)}</td><td data-ic="cosine_rank" data-value="${average(4)}">${fmt(average(4),4)}</td><td data-ic="months">${known}</td><td>${fmt(average(0),0)}</td></tr>`;
    }).join('');
  }''')
    js = replace_function(js, 'exportCSV', '''  function exportCSV() {
    const shown=availableSeries(),ids=shown.filter(s=>s.id!=='risk_free').map(s=>s.id);
    const headers=['signal_date','entry_date','exit_date','holding_month','factor_active','cash_reason','portfolio_side',...shown.map(s=>s.id),...ids.map(id=>id+'_active'),...regimes.map(r=>r.id+'_state'),...ids.map(id=>'raw_'+id)];
    const rows=currentStrategyRows.map(m=>[m.date,m.entry,m.exit,m.month,m.active?1:0,m.cashReason,state.leg,...shown.map(s=>m.r[s.id]),...ids.map(id=>m.activeBySeries[id]?1:0),...regimes.map(r=>m.states[r.id]??''),...ids.map(id=>returnsFor(months[m.i])[id]??'')]);
    const csv='\\uFEFF'+[headers,...rows].map(row=>row.join(',')).join('\\r\\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'})),link=document.createElement('a');
    link.href=url;link.download=`quality_dimensions_${DATA.scenario}_${DATA.missingPolicy}_${state.from}_${state.to}_${state.leg}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }''')
    before = 'currentRows=filterRows();currentStrategyRows=strategyRows();'
    after = 'currentRows=filterRows().filter(m=>m.available[state.focal]);currentStrategyRows=strategyRows();'
    assert before in js
    js = js.replace(before, after)
    js = js.replace('renderHeatmap();renderIC();saveHash();', 'renderHeatmap();renderIC();renderAvailabilityNotes();saveHash();')
    js = js.replace("point?.row && !point.row.active?' · cash RF':''", "point?.row && !point.row.activeBySeries[line.id]?' · cash RF':''")
    notes = '''
  function renderAvailabilityNotes(){
    const cash=currentStrategyRows.length-currentRows.length;
    if(cash){$('returnLabel').textContent='Faktor / RF · CAGR';$('returnNote').textContent='RF izvan režima ili prije raspoloživosti indikatora';$('monthsNote').textContent=cash+' cash · '+currentStrategyRows.length+' ukupno';}
    $('legNote').textContent=state.leg==='long_short'?'Gornjih 20% long +100%, donjih 20% short −100%, jednaki ponderi unutar strana.':
      (state.leg==='long'?'Zaseban long portfelj gornjeg kvintila, +100% NAV-a.':'Zaseban short portfelj donjeg kvintila, −100% NAV-a. Pozitivan prinos znači dobit od shortanja.')+' Vlastiti mjesečni promet i troškovi; u neaktivnim mjesecima RF.';
    $('chartNote').textContent+=' Prije raspoloživosti pojedinog indikatora uvijek se drži RF. '+DATA.costNote;
  }
'''
    js = js.replace('  // Numeric snapshot', notes + '  // Numeric snapshot')
    return js


def filename(scenario, policy):
    return 'statistics.html' if scenario == 'gross_unconstrained' and policy == 'flat_unknown' else 'statistics__' + scenario + '__' + policy + '.html'


def build(update_indicator_report=True):
    monthly = pd.read_csv(OUT / 'statistics_monthly_returns.csv', parse_dates=['signal_date', 'entry_date', 'exit_date'], low_memory=False)
    ic = pd.read_csv(OUT / 'monthly_ic.csv', parse_dates=['signal_date']).set_index(['signal_date', 'signal'])
    features = pd.read_csv(ROOT / 'results/regimes/regime_features.csv', parse_dates=['signal_date'])
    calendar = monthly[['signal_date', 'entry_date', 'exit_date', 'risk_free']].drop_duplicates().sort_values('signal_date')
    calendar = calendar.merge(features[['signal_date', 'common_sample'] + [r['id'] + '_state' for r in REGIMES]], on='signal_date', validate='one_to_one')
    series = [dict(id=signal, label=SPECS[signal]['label'], short=SHORT[i], color=COLORS[i], component=True,
                   componentCount=len(SPECS[signal]['components']), legs=['long_short', 'long', 'short'])
              for i, signal in enumerate(analysis.SIGNALS)]
    series.append(dict(id='risk_free', label='Risk-free / cash', short='RF', color='#74869a', legs=['long_short', 'long', 'short']))
    template = (ASSETS / 'portfolio.html').read_text(encoding='utf-8')
    template = template.replace('Quality · Profitabilna komponenta', 'Quality · Statistike dimenzija')
    template = template.replace('<h1>Quality: profitabilna komponenta</h1><p>Dosadašnji portfelj profitabilnosti u širem projektu quality faktora.</p>',
        '<h1>Quality: statistike dimenzija</h1><p>Mjesečni kvintilni portfelji, IC i cosine, long/short strane i tržišni režimi.</p>')
    template = template.replace('STRUKTURA PORTFELJA', 'ZASEBNE DIMENZIJE')
    template = template.replace('Jednake kapitalne težine.<br>Mjesečni rebalans, bez optimizacije.',
        'Zasebni indikatori i zasebni portfelji. Ovdje se ne kombiniraju u ukupni quality portfelj.')
    template = template.replace('Profitabilnost je jedna dimenzija kvalitete. Ove rezultate još ne tumačimo kao test potpunog quality faktora.',
        'Svaka dimenzija koristi vlastiti potpuni univerzum. <a href="report.html">Sastavnice, pokrivenost i presjek dionica</a>.')
    template = template.replace('Postojeći 21 signal', 'Kompozitni signali')
    template = template.replace('Mjesečni prosjeci za postojeći 21 signal profitabilnosti u aktivnim mjesecima, uz njihove vlastite formacijske uzorke. IC se odnosi na rangiranje dionica; portfelj faktorskih prinosa nema zaseban cross-sectional IC. Za usporedbu provjeri dostupne mjesece i pokrivenost.',
        'Mjesečni prosjeci kompozita u odabranim režimskim mjesecima, uz vlastite formacijske uzorke. Mjeseci bez indikatora nemaju IC. Izvorni cosine koristi interni score/50 − 1. IC ostaje statistika dioničkog signala pri svim prikazima strana i troškova. HAC koristi tri laga.')
    template = template.replace('<th>Spearman IC</th>', '<th>Spearman IC</th><th>HAC t · IC</th>')
    template = template.replace('<th scope="col">Pozitivni mj.</th>', '<th scope="col">Pozitivni mj.</th><th scope="col">Aktivni / svi mj.</th>')
    template = template.replace('<section class="panel ic-panel"><details>', '<section class="panel ic-panel"><details open>')
    template = re.sub(r'<noscript>.*?</noscript>', '<noscript>Statistike su dostupne u <a href="statistics_summary.csv">CSV sažetku</a>.</noscript>', template, flags=re.S)
    template = template.replace('<section class="selection-bar">', '__STATISTICS_CONTROLS__<section class="selection-bar">')
    methodology = '''<section class="panel methodology" id="methodology"><details><summary><h2>Metodologija i datoteke</h2></summary><div style="padding:24px">
    <p>Mjesečne formacije iz unaprijed izračunatih kompozita. Gornjih 20% dionica kupuje se, a donjih 20% shorta, uz +100% i −100% ciljnih težina. Rangovi i podobnost određuju se prije budućih ishoda. Ulaz je zatvaranje sljedeće sesije; izlaz zatvaranje sesije nakon idućeg kraja mjeseca.</p>
    <p>Long i short prikazi zasebno se mjesečno rebalansiraju i svaki ima 100% notionala. Borrow se obračunava samo na short pozicije. Aktivni portfelji nemaju dodatnu kamatu na cash/kolateral. Prije raspoloživosti indikatora i izvan režima primjenjuje se RF iz istog dnevnog arhiva i istih prozora držanja.</p>
    <p>Troškovi se računaju na cjelokupan apsolutni promet, s driftom težina i završnim zatvaranjem. Neto režimske krivulje koriste mjesečne prinose kontinuirane strategije; dodatni promet režimskog ulaska/izlaska nije ponovno procijenjen. Short dostupnost i naknade ovdje su pretpostavke. Nepoznati izlazi imaju osnovni nulti prinos ili zasebno označen nepovoljan scenarij.</p>
    <p>Dimenzije imaju različite univerzume i preklapajuće sastavnice; usporedba ne izolira doprinos jedne karakteristike. Nema sektorske neutralizacije ni zajedničke regresijske FMP procjene. Petogodišnji rast je projektni proxy promjene omjera i počinje 2012. Izvori mogu sadržavati revizije i procijenjene datume objave.</p>
    <p>IC se računa samo na poznatim budućim ishodima, uz najmanje 100 opažanja. Nepoznati ishodi ne uklanjaju dionice iz formacijskih portfelja. Spearman je rangovna korelacija; cosine koristi interni kompozit, odnosno centrirane formacijske rangove. Svi nalazi su istraživački; težine nisu prilagođene rezultatima.</p>
    <p><a href="statistics_summary.csv">Sažetak</a> · <a href="monthly_ic.csv">Mjesečni IC</a> · <a href="ic_summary.csv">IC sažetak</a> · <a href="statistics_monthly_returns.csv">Mjesečni portfelji</a> · <a href="statistics_regime_cash_summary.csv">Režimi</a> · <a href="statistics_manifest.json">Manifest</a> · <a href="report.html">Indikatori</a></p>
    <div id="regimeDefinitions"></div></div></details></section>'''
    css = (ASSETS / 'portfolio.css').read_text(encoding='utf-8') + '\n#correlationMatrix{overflow-x:auto}.correlation-table{min-width:540px}.statistics-controls{padding:20px;margin-bottom:20px}.statistics-controls select{max-width:450px;margin:8px 0 16px;display:block}'
    js = local_javascript()
    for scenario in analysis.CONFIG['scenarios']:
        for policy in analysis.CONFIG['missing_policies']:
            part = monthly[monthly.scenario.eq(scenario['name']) & monthly.missing_policy.eq(policy)]
            maps = {leg: part[part.leg.eq(leg)].pivot(index='signal_date', columns='signal', values='return') for leg in analysis.CONFIG['legs']}
            available = part[part.leg.eq('long_short')].pivot(index='signal_date', columns='signal', values='indicator_available')
            records, ic_values = [], []
            for i, row in enumerate(calendar.itertuples()):
                readiness = {s: bool(available.loc[row.signal_date, s]) for s in analysis.SIGNALS}
                def values(leg):
                    return {**{s: float(maps[leg].loc[row.signal_date, s]) if readiness[s] else None for s in analysis.SIGNALS}, 'risk_free': float(row.risk_free)}
                records.append(dict(i=i, date=row.signal_date.strftime('%Y-%m-%d'), entry=row.entry_date.strftime('%Y-%m-%d'),
                    exit=row.exit_date.strftime('%Y-%m-%d'), month=row.entry_date.strftime('%Y-%m'), common=bool(row.common_sample),
                    rf=float(row.risk_free), available={**readiness, 'risk_free': True}, r=values('long_short'),
                    legs={leg: values(leg) for leg in ['long', 'short']},
                    states={r['id']: None if pd.isna(getattr(row, r['id'] + '_state')) else int(getattr(row, r['id'] + '_state')) for r in REGIMES}))
                ic_row = []
                for signal in analysis.SIGNALS:
                    if (row.signal_date, signal) not in ic.index:
                        ic_row.append(None)
                    else:
                        observation = ic.loc[(row.signal_date, signal)]
                        ic_row.append([None if pd.isna(observation[key]) else float(observation[key]) for key in
                            ['n', 'n_observed', 'ic_spearman', 'ic_cosine_raw', 'ic_cosine_rank_centered']])
                ic_values.append(ic_row)
            cost_note = 'Bruto, bez troškova.' if scenario['name'] == 'gross_unconstrained' else scenario['label'] + '; dodatni promet režimskog uključivanja nije procijenjen.'
            payload = dict(series=series, regimes=REGIMES, months=records,
                commonMonths=int(calendar.common_sample.sum()), scenario=scenario['name'], missingPolicy=policy,
                costNote=cost_note, chartBasis=cost_note,
                legs=[dict(id='long_short', label='Long–short'), dict(id='long', label='Long · 100%'), dict(id='short', label='Short · 100%')],
                ic=dict(signals=[dict(id=s, label=SPECS[s]['label']) for s in analysis.SIGNALS], values=ic_values))
            cost_options = ''.join(f'<option value="{filename(s["name"], policy)}" {"selected" if s["name"] == scenario["name"] else ""}>{html.escape(s["label"])}</option>' for s in analysis.CONFIG['scenarios'])
            policy_options = ''.join(f'<option value="{filename(scenario["name"], p)}" {"selected" if p == policy else ""}>{label}</option>' for p, label in [('flat_unknown', 'Osnovna pretpostavka: prinos 0'), ('adverse_unknown', 'Nepovoljan ishod za svaku stranu')])
            controls = f'''<section class="panel statistics-controls"><label for="costScenario"><strong>Troškovi</strong></label><select id="costScenario">{cost_options}</select><label for="missingPolicy"><strong>Nepoznati izlazi</strong></label><select id="missingPolicy">{policy_options}</select><p>{html.escape(cost_note)} Prikaz: {'osnovna pretpostavka nepoznatog izlaza' if policy == 'flat_unknown' else 'nepovoljni nepoznati ishodi'}.</p><label for="legMode"><strong>Strana portfelja</strong></label><select id="legMode" aria-label="Strana portfelja"></select><p id="legNote"></p></section>'''
            extra_js = "\nfor(const id of ['costScenario','missingPolicy'])document.getElementById(id).addEventListener('change',e=>{location.href=e.target.value+location.hash;});"
            page = template.replace('Bruto rezultati', html.escape(scenario['label']))
            for key, value in {'__REPORT_CSS__': css, '__REPORT_JS__': js + extra_js,
                '__REPORT_METHODOLOGY__': methodology, '__STATISTICS_CONTROLS__': controls,
                '__REPORT_DATA__': json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')}.items():
                page = page.replace(key, value)
            assert '__REPORT_' not in page
            (OUT / filename(scenario['name'], policy)).write_text(page, encoding='utf-8')
    if update_indicator_report:
        from build_quality_dimensions_report import build_report
        build_report()
    print('Built six standalone statistics dashboards' + (' and updated the indicator report.' if update_indicator_report else '.'), flush=True)


if __name__ == '__main__':
    build()
