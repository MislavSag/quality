"""Standalone indicator methodology, coverage and latest cross-section browser."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/quality_dimensions'


def build_report():
    config = json.loads((ROOT / 'config/quality_dimensions.json').read_text(encoding='utf-8'))
    specs = {**config['dimensions'], **config['comparators']}
    descriptions = {
        'profitability': ('Veća profitabilnost prema četiri postojeće mjere. Godišnji FF pokazatelj određuje uži potpuni univerzum.', 'Prethodni kompozit s četiri sastavnice; rezultat reproduciran.'),
        'earnings_quality': ('Niži novčani i operativni accruals te manji udio neto operativne imovine u ukupnoj imovini.', 'Projektni kompozit; nije točna replika Sloana ili S&P-a.'),
        'financial_safety': ('Manji dug i pritisak kratkoročnog refinanciranja te veće pokriće kamata.', 'Fundamentalni proxy sigurnosti. Nema bete ni kalibrirane vjerojatnosti bankrota. Nedefinirano pokriće kamata ostaje nedostajuće.'),
        'earnings_stability': ('Manja volatilnost kvartalne dobiti, operativnog novčanog toka i prodaje, skalirana prosječnom imovinom.', 'Postojeći prozor od 20 uzastopnih fiskalnih kvartala, najmanje 12 konačnih opažanja po mjeri. Nije QMJ-ova volatilnost ROE-a.'),
        'quality_growth': ('Poboljšanje ROA-a, ROE-a, GP/imovina, CFO/imovina i bruto marže u odnosu na isti mjesec pet godina ranije.', 'Proxy promjene omjera. Ne uključuje QMJ-ovu rezidualnu dobit po dionici i ne zamjenjuje nedostajuću petogodišnju povijest kraćim prozorom.'),
        'investment_discipline': ('Manji jednogodišnji rast imovine te manja povećanja neto dugotrajne materijalne imovine i zaliha prema početnoj imovini.', 'Proxy konzervativnog investiranja. Slabije investiranje samo po sebi ne znači optimalno poslovanje.'),
        'payout_dilution': ('Viši dividendni prinos, niže neto izdavanje kapitala kroz novčane tokove i niži rast prijavljenog broja dionica.', 'Prinosi uključuju tržišnu kapitalizaciju, pa i cjenovnu dimenziju. Promjene broja dionica ovise o osnovici dobavljača i splitovima. Nedostajuće dividende nisu nula.'),
        'profitability_broad': ('Veća profitabilnost prema FCF/imovina, CFO/imovina i GP/imovina, bez užeg godišnjeg FF ulaza.', 'Postojeći široki usporedni kompozit s tri sastavnice; ne zamjenjuje osnovni indikator s četiri sastavnice.')
    }
    specs = {key: {**value, 'interpretation': descriptions[key][0], 'status': descriptions[key][1]}
             for key, value in specs.items()}
    summary = pd.read_csv(OUT / 'dimension_summary.csv')
    coverage = pd.read_csv(OUT / 'coverage.csv')
    latest = pd.read_csv(OUT / 'latest_scores.csv')
    metadata = pd.read_csv(OUT / 'component_metadata.csv')
    payload = {'specs': specs, 'summary': json.loads(summary.to_json(orient='records')),
               'coverage': json.loads(coverage.to_json(orient='records')),
               'latest': json.loads(latest.to_json(orient='records')),
               'metadata': json.loads(metadata.to_json(orient='records'))}
    statistics_available = (OUT / 'statistics_summary.csv').exists() and (OUT / 'ic_summary.csv').exists()
    if statistics_available:
        payload['statistics'] = json.loads(pd.read_csv(OUT / 'statistics_summary.csv').to_json(orient='records'))
        payload['icSummary'] = json.loads(pd.read_csv(OUT / 'ic_summary.csv').to_json(orient='records'))
    options = ''.join(f'<option value="{name}">{html.escape(spec["label"])}</option>'
                      for name, spec in specs.items())
    exclusion_labels = {
        'FUND_WORKING_CAPITAL_ACCRUALS_YOY': 'Oko 0,1% pokrivenosti podobnih dionica u 2007., 2012. i 2020.; zato ne ograničava osnovni indikator kvalitete dobiti.',
        'FUND_CFO_CONVERSION_TTM': 'CFO/neto dobit može pogrešno rangirati gubitaše i nestabilan je oko nulte dobiti; izostavljen iz osnovnog kompozita.',
        'FUND_DEBT_PAYDOWN_YIELD_TTM': 'Oko 0,2% pokrivenosti u 2007.; izostavljen kako se formula payout indikatora ne bi mijenjala kroz vrijeme.',
        'FUND_REPURCHASE_YIELD_TTM': 'Točan negativ neto izdavanja kapitala; uključivanje oba polja dvostruko bi brojalo isti ulaz.',
        'FUND_SHAREHOLDER_YIELD_TTM': 'Zbroj dividendnog i repurchase prinosa; ne uključuje se uz vlastite sastavnice.'
    }
    exclusions = ''.join(f'<li><code>{html.escape(name)}</code>: {html.escape(reason)}</li>'
                         for name, reason in exclusion_labels.items())
    document = '''<!doctype html><html lang="hr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231b62aa'/%3E%3Ctext x='7' y='24' fill='white' font-size='25'%3EQ%3C/text%3E%3C/svg%3E">
<title>Quality · Kompozitni indikatori dimenzija</title>
<style>
:root{--ink:#172e46;--muted:#5b7185;--blue:#1b62aa;--line:#dce5ed}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:30px 22px 70px}header{margin-bottom:26px}.brand{font-size:12px;letter-spacing:.16em;font-weight:750;color:var(--blue)}
h1{font-size:clamp(26px,4vw,40px);line-height:1.2;margin:12px 0}h2{font-size:20px;margin:0 0 14px}p{margin:10px 0}a{color:var(--blue)}
.panel{background:white;border:1px solid var(--line);border-radius:14px;padding:23px;margin:18px 0}.muted{color:var(--muted)}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end}label{display:block;font-size:13px;color:var(--muted);flex:1;min-width:180px}
select,input{display:block;width:100%;font:inherit;padding:10px;border:1px solid #c4d3df;border-radius:8px;background:white;color:var(--ink)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:22px}.metric{background:#f0f5fb;padding:14px;border-radius:8px}
.metric strong{display:block;font-size:23px}.metric span{font-size:12px;color:var(--muted)}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}td.number{text-align:right;font-variant-numeric:tabular-nums}
code{font-size:12px;overflow-wrap:anywhere}.note{border-left:4px solid #c38b21;background:#fffaed;padding:13px 18px}.chart{width:100%;height:auto;min-height:180px}
.score{display:inline-block;min-width:80px;background:linear-gradient(to right,#dbeaff var(--score),transparent 0);padding:1px 7px;border-radius:4px}
li{margin:9px 0}footer{font-size:12px;color:var(--muted);margin-top:25px}@media(max-width:650px){main{padding:20px 12px}.panel{padding:17px}.metrics{grid-template-columns:repeat(2,1fr)}th,td{padding:7px}.scroll table{min-width:660px}}
</style></head><body><main>
<header><div class="brand">QUALITY / RESEARCH</div><h1>Dimenzije · raniji širi univerzum</h1>
<p class="muted">Sedam dimenzija kvalitete i široki usporedni indikator profitabilnosti. Mjesečni presjek dionica, isti kriteriji podobnosti kao u prethodnoj analizi.</p>
<p><a href="../combined_quality/report.html"><strong>IWV quality faktor: ALL7 i CORE5</strong></a> · <a href="../quality_literature/report.html">Literatura i metodologije</a> · <a href="../joint_fmp/report.html">Dosadašnji FMP profitabilnosti</a></p></header>
<section class="panel"><div class="controls"><label>Dimenzija<select id="dimension">__OPTIONS__</select></label></div>
<div class="metrics"><div class="metric"><strong id="latestN"></strong><span>Dionica na posljednji datum</span></div>
<div class="metric"><strong id="latestCoverage"></strong><span>Udio tada podobnog univerzuma</span></div>
<div class="metric"><strong id="months"></strong><span>Mjeseci s indikatorom</span></div>
<div class="metric"><strong id="first"></strong><span>Prvi raspoloživi presjek</span></div></div></section>
__STATISTICS_SECTION__
<section class="panel"><h2 id="label"></h2><p id="interpretation"></p><p class="muted" id="status"></p>
<div class="scroll"><table><thead><tr><th>Sastavnica</th><th>Smjer</th><th>Težina</th><th>Definicija ulaza</th></tr></thead><tbody id="components"></tbody></table></div></section>
<section class="panel"><h2>Povijesna pokrivenost</h2><p class="muted">Broj dionica sa scoreom (plavo) i broj podobnih dionica (sivo). X-os prikazuje vrijeme.</p><svg id="coverageChart" class="chart" viewBox="0 0 1040 260" role="img" aria-label="Povijesna pokrivenost kompozitnog indikatora"></svg></section>
<section class="panel"><h2 id="latestTitle"></h2><p class="muted">Score od 0 do 100: više znači povoljnije obilježje unutar vlastitog potpunog presjeka. Prikazano do 40 dionica; nedostajući score ostaje prazan u izvozu.</p>
<div class="controls"><label>Pretraga tickera ili SID-a<input id="search" placeholder="npr. AAPL" autocomplete="off"></label><label>Poredak<select id="order"><option value="desc">Najviši score prvo</option><option value="asc">Najniži score prvo</option></select></label></div>
<div class="scroll"><table><thead><tr><th>Ticker</th><th>Security ID</th><th>Score</th><th>Sastavnice</th></tr></thead><tbody id="stocks"></tbody></table></div></section>
<section class="panel"><h2>Kako se kombiniraju sastavnice</h2>
<p>Za svaku dimenziju i datum uzimaju se samo tada podobne dionice sa svim njezinim konačnim sastavnicama. Smjer pokazatelja okreće se prema boljem obilježju. Percentil je <code>(prosječni rang − 0,5) / N</code>; score je 100 puta prosjek percentila s jednakim težinama. Interni signal je <code>score / 50 − 1</code>. Presjek mora imati najmanje 100 dionica.</p>
<p>Težine nisu prilagođavane budućim prinosima. Svaka dimenzija ima vlastiti potpuni univerzum; scoreovi nisu apsolutne ocjene niti su nužno izravno usporedivi između različitih univerzuma.</p>
<p class="note">Petogodišnji rast je promjena prethodno opaženih omjera profitabilnosti, bez rezidualnih dobitaka i prilagodbe po dionici iz QMJ-a. Početnih 60 mjeseci nema score. Stabilnost koristi postojeće 20-kvartalne mjere i nije volatilnost ROE-a.</p>
<p>Nema sektorske neutralizacije. Mjere duga, accruals i ulaganja zahtijevaju poseban tretman financijskih poduzeća i nekretnina za industrijsku replikaciju. Sigurnost i stabilnost ostaju odvojene; pri budućem ukupnom scoreu ne treba ih neprimjetno dvostruko ponderirati. Payout prinosi uključuju i cjenovnu dimenziju. Postojani gubici također mogu imati nisku volatilnost.</p>
<p>Procijenjeni datumi objava i naknadne revizije izvornog dobavljača ostaju ograničenje. Nijedan nedostajući ulaz nije dopunjen nulom.</p></section>
<section class="panel"><h2>Izbor ulaza nakon provjere pokrivenosti</h2><ul>__EXCLUSIONS__</ul></section>
<section class="panel"><h2>Podaci i dokumentacija</h2><p><a href="monthly_dimension_scores.parquet">Mjesečni scoreovi — Parquet</a> · <a href="latest_scores.csv">Posljednji presjek — CSV</a> · <a href="component_metadata.csv">Sastavnice i formule</a> · <a href="coverage.csv">Pokrivenost</a> · <a href="component_coverage.csv">Pokrivenost sastavnica</a> · <a href="score_correlations_summary.csv">Korelacije scoreova</a> · <a href="manifest.json">Manifest</a></p></section>
<footer>Metodologija 12. studenoga 2026. · Izvještaj radi lokalno, bez vanjskih skripti i mrežnih zahtjeva.</footer>
</main><script id="qualityData" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('qualityData').textContent);
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
__STATISTICS_JS__
function render(){const dim=$('dimension').value,spec=data.specs[dim],summary=data.summary.find(x=>x.dimension===dim);
$('label').textContent=spec.label;$('interpretation').textContent=spec.interpretation;$('status').textContent=spec.status;
$('latestN').textContent=summary.latest_scored_n.toLocaleString('hr-HR');$('latestCoverage').textContent=(100*summary.latest_coverage).toFixed(1)+'%';$('months').textContent=summary.scored_months;$('first').textContent=summary.first_scored_date?.slice(0,10)||'—';
$('components').innerHTML=data.metadata.filter(x=>x.dimension===dim).map(x=>`<tr><td><code>${esc(x.component)}</code></td><td>${x.direction>0?'Više je bolje':'Niže je bolje'}</td><td>${(100*x.weight).toFixed(1)}%</td><td>${esc(x.formula)}</td></tr>`).join('');
const series=data.coverage.filter(x=>x.dimension===dim),max=Math.max(...series.map(x=>x.eligible_n))*1.1,x=i=>55+950*i/(series.length-1),y=v=>220-185*v/max;
let svg='';for(let j=0;j<=4;j++){let v=max*j/4;svg+=`<line x1="55" y1="${y(v)}" x2="1005" y2="${y(v)}" stroke="#e2e9f0"/><text x="47" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#5b7185">${Math.round(v)}</text>`;}
for(const year of [2007,2010,2015,2020,2026]){let i=series.findIndex(z=>z.signal_date.startsWith(String(year)));if(i>=0)svg+=`<text x="${x(i)}" y="244" text-anchor="middle" font-size="12" fill="#5b7185">${year}</text>`;}
for(const [field,color] of [['eligible_n','#aebcc8'],['scored_n','#1b62aa']])svg+=`<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${series.map((z,i)=>x(i)+','+y(z[field])).join(' ')}"/>`;
$('coverageChart').innerHTML=svg;
$('latestTitle').textContent='Dionice · '+summary.latest_date.slice(0,10);
const query=$('search').value.toLowerCase(),direction=$('order').value==='asc'?1:-1;
const rows=data.latest.filter(z=>Number.isFinite(z[dim+'_score'])&&(!query||(z.ticker+' '+z.security_id).toLowerCase().includes(query))).sort((a,b)=>direction*(a[dim+'_score']-b[dim+'_score'])).slice(0,40);
$('stocks').innerHTML=rows.map(z=>`<tr><td>${esc(z.ticker)}</td><td><code>${esc(z.security_id)}</code></td><td class="number"><span class="score" style="--score:${z[dim+'_score']}%">${z[dim+'_score'].toFixed(2)}</span></td><td>${z[dim+'_component_count']}/${Object.keys(spec.components).length}</td></tr>`).join('')||'<tr><td colspan="4">Nema raspoloživih rezultata.</td></tr>';
window.qualityDimensionSnapshot={dimension:dim,summary,displayed:rows.map(z=>({security_id:z.security_id,score:z[dim+'_score']}))};if(data.statistics)renderStatistics(dim);}
for(const id of ['dimension','order'])$(id).addEventListener('change',render);$('search').addEventListener('input',render);render();
</script></body></html>'''
    document = document.replace('__OPTIONS__', options).replace('__EXCLUSIONS__', exclusions)
    if statistics_available:
        section, javascript = statistics_widgets()
        document = document.replace('__STATISTICS_SECTION__', section).replace('__STATISTICS_JS__', javascript)
    else:
        document = document.replace('__STATISTICS_SECTION__', '').replace('__STATISTICS_JS__', '')
    document = document.replace('__DATA__', json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/'))
    (OUT / 'report.html').write_text(document, encoding='utf-8')


def statistics_widgets():
    section = '''<section class="panel" id="statistics"><h2>Statistike odabrane dimenzije</h2>
<p class="muted">Mjesečni portfelji gornjeg i donjeg kvintila, isti troškovni model kao u analizi profitabilnosti. Svaki indikator ima vlastiti potpuni univerzum.</p>
<div class="controls"><label>Razdoblje statistika<select id="statsPeriod"><option value="full">Cijeli kalendar + RF prije indikatora</option><option value="available_history">Od prvog indikatora</option><option value="common_history">Zajedničko razdoblje od 2012.</option><option value="early">Formacije prije 2017.</option><option value="late">Formacije od 2017.</option></select></label>
<label>Troškovi<select id="statsScenario"><option value="gross_unconstrained">Bruto</option><option value="cost5_borrow0p5">5 bps + 0,5% posudba</option><option value="cost10_borrow3">10 bps + 3% posudba</option></select></label>
<label>Strana portfelja<select id="statsLeg"><option value="long_short">Long–short +100% / −100%</option><option value="long">Long +100%</option><option value="short">Short −100%</option></select></label>
<label>Nepoznati izlazi<select id="statsMissing"><option value="flat_unknown">Osnovna pretpostavka: prinos 0</option><option value="adverse_unknown">Nepovoljan ishod za svaku stranu</option></select></label></div>
<div class="metrics" id="statsMetrics"></div><p id="statsSample" class="muted"></p>
<div class="scroll"><table><thead><tr><th>Signalna statistika</th><th>Prosjek</th><th>HAC t (3 laga)</th><th>Mjeseci</th></tr></thead><tbody id="statsIC"></tbody></table></div>
<p class="muted">IC je Spearmanova korelacija s budućim prinosom. Izvorni cosine koristi interni kompozit <code>score/50 − 1</code>; druga cosine mjera koristi centrirane rangove kompozita. IC opisuje izvorni dionički signal i ne mijenja se odabirom long/short strane ili troškova.</p>
<p><a id="detailedStatistics" href="statistics.html"><strong>Grafovi, tržišni režimi, usporedba više serija i long/short prikaz →</strong></a></p>
<h2>Usporedba svih dimenzija</h2><div class="scroll"><table><thead><tr><th>Dimenzija</th><th>CAGR</th><th>Volatilnost</th><th>Sharpe vs RF</th><th>Najveći pad</th><th>Aktivni / svi mjeseci</th></tr></thead><tbody id="statsComparison"></tbody></table></div>
<p class="note">U mjesecima bez indikatora ostvaruje se RF. Aktivni portfelji nemaju dodatnu kamatu na cash ili kolateral. Razlike u prinosima uključuju i razlike u univerzumima. Za režimske neto krivulje promet uključivanja i isključivanja nije ponovno izračunat.</p>
<p><a href="statistics_summary.csv">Statistike portfelja — CSV</a> · <a href="monthly_ic.csv">Mjesečni IC i cosine</a> · <a href="ic_summary.csv">IC sažetak</a> · <a href="statistics_monthly_returns.csv">Mjesečni portfeljni rezultati</a></p></section>'''
    javascript = '''const fstat=(v,n=2)=>Number.isFinite(v)?v.toLocaleString('hr-HR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—';
const pstat=v=>Number.isFinite(v)?fstat(100*v)+'%':'—';
function renderStatistics(dim){
const period=$('statsPeriod').value,scenario=$('statsScenario').value,leg=$('statsLeg').value,policy=$('statsMissing').value;
const subset=data.statistics.filter(z=>z.period===period&&z.scenario===scenario&&z.leg===leg&&z.missing_policy===policy),row=subset.find(z=>z.signal===dim);
if(!row)return;
const metrics=[['CAGR',pstat(row.cagr)],['Sharpe prema RF-u',fstat(row.sharpe_rf)],['Godišnja volatilnost',pstat(row.annual_vol)],['Najveći mjesečni drawdown',pstat(row.monthly_max_drawdown)],['Aktivni / svi mjeseci',row.active_months+' / '+row.months],['Cash mjeseci',row.cash_months],['Prosječni mjesečni promet · aktivno',pstat(row.mean_turnover_active)],['Pozitivni mjeseci',pstat(row.positive_month_fraction)]];
$('statsMetrics').innerHTML=metrics.map(([label,value])=>`<div class="metric"><strong>${value}</strong><span>${label}</span></div>`).join('');
$('statsSample').textContent=row.first_entry+' – '+row.last_exit+' · Prva aktivna formacija: '+row.first_active_signal_date+(leg==='short'?' · Pozitivan prinos short strane znači dobit od shortanja.':'');
const ic=data.icSummary.find(z=>z.signal===dim&&z.period===(period==='available_history'?'full':period));
const definitions=[['ic_spearman','Spearman IC'],['ic_cosine_raw','Cosine · interni kompozit'],['ic_cosine_rank_centered','Cosine · centrirani rangovi']];
$('statsIC').innerHTML=definitions.map(([key,label])=>`<tr><td>${label}</td><td>${fstat(ic?.['mean_'+key],4)}</td><td>${fstat(ic?.[key+'_hac_t'])}</td><td>${ic?.months??'—'}</td></tr>`).join('');
$('statsComparison').innerHTML=Object.keys(data.specs).map(signal=>{const z=subset.find(v=>v.signal===signal);return z?`<tr style="${signal===dim?'background:#edf4ff;font-weight:600':''}"><td>${esc(data.specs[signal].label)}</td><td>${pstat(z.cagr)}</td><td>${pstat(z.annual_vol)}</td><td>${fstat(z.sharpe_rf)}</td><td>${pstat(z.monthly_max_drawdown)}</td><td>${z.active_months} / ${z.months}</td></tr>`:'';}).join('');
const file=scenario==='gross_unconstrained'&&policy==='flat_unknown'?'statistics.html':'statistics__'+scenario+'__'+policy+'.html';
const last=new Date(row.last_exit+'T12:00:00Z');last.setUTCMonth(last.getUTCMonth()-1);const to=last.toISOString().slice(0,7);
$('detailedStatistics').href=file+'#'+new URLSearchParams({series:dim,leg,from:row.first_entry.slice(0,7),to,lines:dim+',profitability_broad'});
window.qualityDimensionSnapshot.statistics={...row,ic};}
for(const id of ['statsPeriod','statsScenario','statsLeg','statsMissing'])$(id).addEventListener('change',()=>renderStatistics($('dimension').value));
'''
    return section, javascript


if __name__ == '__main__':
    build_report()
