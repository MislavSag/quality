"""Combined quality overview plus offline, multi-series regime dashboards."""
from __future__ import annotations

import html
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

import analyze_combined_quality as analysis
import build_quality_dimension_statistics_report as dashboards

ROOT, OUT = analysis.ROOT, analysis.OUT
COLORS = ['#365eea', '#209582', '#a87943', '#677fc8', '#64a99c', '#c79f6f',
          '#954bc2', '#c44f77', '#668a34', '#8962aa', '#c17994', '#91a95b']
SHORT = ['ALL7', 'CORE5', 'CORE5 = ALL7', 'ALL7 sort/FMP', 'CORE5 sort/FMP', 'CORE5 = ALL7/FMP',
         'ALL7 FMP', 'CORE5 FMP', 'CORE5 FMP = ALL7', 'ALL7 FMP β/vol', 'CORE5 FMP β/vol', 'CORE5 FMP = ALL7 β/vol']


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def methodology():
    return '''<section class="panel methodology" id="methodology"><details open><summary><h2>Kako je konstruiran quality faktor?</h2></summary><div style="padding:24px">
    <p><strong>Univerzum: povijesne IWV sastavnice iz paketa etfconstituents.</strong> Koristi se posljednji target_snapshot_date dostupan do zatvaranja formacijskog datuma, uz source_snapshot_date koji nije kasniji od target datuma. Članstvo se povezuje samo točnim punim SID-em i zahtijeva identity_usable=true. Ne koriste se današnje sastavnice kroz povijest, spajanje tickerom ni SID prefiksom. Usable retci s objavljenom težinom nula ostaju članovi jer težina može biti zaokružena.</p>
    <p>Prvi IWV snapshot je 28. lipnja 2009., a prva formacija 30. lipnja 2009. Prije toga nema rekonstruiranog članstva; puni kalendar u tim mjesecima drži RF. Prethodni fundamentalni podaci mogu se koristiti za petogodišnju promjenu, bez zahtjeva da je dionica i pet godina ranije bila član IWV-a. Zadržani su postojeći filtri raspoloživog USD izvještaja, kapitalizacije ≥100 mil. USD, sirove cijene ≥5 USD, ADV21 ≥1 mil. USD i cjenovnog ugovora. IWV određuje članstvo, a njegove težine ne određuju pondere strategije.</p>
    <p>Radi usporedbe promjene portfeljnog univerzuma zadržani su postojeći tržišni režimi i SPY benchmark. Širina, disperzija i agregatna nelikvidnost i dalje opisuju prethodni širi podobni panel; nisu ponovno definirani na ALL7, CORE5 ni IWV dionicama. SPY volatilnost/trend i SPY–TLT korelacija također su nepromijenjeni.</p>
    <p><strong>ALL7:</strong> profitabilnost (široki tri), kvaliteta dobiti, financijska sigurnost, stabilnost poslovanja, petogodišnji rast profitabilnosti, investicijska disciplina i payout/razrjeđivanje. Svaka dimenzija ima 1/7 težine. <strong>CORE5:</strong> bez rasta i payouta, svaka od pet dimenzija ima 1/5 težine. Formule i ponderi nisu odabrani prema rezultatima prinosa.</p>
    <p>Svaki mjesec zahtijevaju se sve sirove sastavnice. Na konačnom zajedničkom univerzumu ponovno se izračunaju usmjereni percentilni rangovi sastavnica, jednako ponderiraju unutar dimenzije, a prosjek dimenzije pretvara u normalni rangovni z-score s nultom sredinom i jediničnom populacijskom devijacijom. Kombinirani interni signal prosjek je dimenzijskih z-scoreova. Prikazani score 0–100 jest njegov percentil. IC/cosine izvorni koristi interni signal, prije završnog percentila.</p>
    <p><strong>Kvintili:</strong> gornjih 20% long +100%, donjih 20% short −100%, jednaki ponderi unutar strane. Signali se očitavaju na kraju mjeseca; ulaz je zatvaranje sljedeće sesije, izlaz zatvaranje sesije nakon sljedećeg kraja mjeseca. Budući ishod ne određuje podobnost.</p>
    <p><strong>Zajednički FMP:</strong> WLS regresija budućih viškova prinosa na sve dimenzije zajedno, konstantu i kontrole za log veličine, value, momentum i log nelikvidnosti. Ciljne dimenzijske izloženosti već su normalizirane. Kontrole su normalni rangovi, centrirani kapitalizacijskim ponderima i skalirani jednakoponderiranom populacijskom devijacijom. WLS težina je korijen kapitalizacije. QR daje sirove FMP težine za jediničnu izloženost svakoj dimenziji.</p>
    <p>Sirove težine dimenzija kombiniraju se s jednakim budžetima izloženosti, netiraju na razini dionice i tek zatim ukupno skaliraju na bruto izloženost 2. Zbog skaliranja pojedinačna dimenzijska izloženost više nije 1/7 odnosno 1/5. Beta/vol varijanta dodaje betu i rezidualnu volatilnost; može ukloniti namjerni obrambeni dio qualityja. Obje varijante koriste isti skup dionica s dostupnih svih šest kontrola. Nema procjene rezidualnog rizika budućim podacima ni optimizacije pondera prema prinosima.</p>
    <p><strong>Usporedbe:</strong> CORE5 na ALL7 uzorku izolira promjenu dimenzija pri istim dionicama; kvintili na FMP uzorku odvajaju učinak konstrukcije portfelja od dostupnosti kontrola. Usporedbe vlastitih univerzuma to ne izoliraju. Tablica IC-a prikazuje samo dioničke scoreove, a ne IC regresijskog portfelja.</p>
    <p>Long i short zasebno imaju 100% notionala, vlastiti drift težina, promet i troškove. Short prinos pozitivan je kad short pozicije zarađuju. Aktivni mjeseci nemaju kamatu na cash/kolateral ni financiranje. Prije indikatora i izvan režima drži se točan RF za prozor držanja. Neto režimske krivulje koriste prinose kontinuirane strategije; dodatni promet pri režimskom uključivanju/isključivanju nije ponovno obračunat.</p>
    <p>Scenariji: bruto; 5 bps po apsolutnom prometu uz 0,5% godišnje posudbe; 10 bps uz 3%. Uključeno je završno zatvaranje. Dostupnost shorta i stvarne naknade nisu poznate. Nepoznati izlazi koriste osnovni nulti prinos ili nepovoljan ishod za pojedinu stranu (−100% underlying na longu, +100% na shortu). Sirovi regresijski koeficijenti zasebno su dijagnostika s nultim nepoznatim underlying prinosima. Odbijeni ulazi mogu narušiti teorijsku neutralnost FMP-a.</p>
    <p>Projektni proxy, bez sektorske neutralizacije i bez tvrdnje o točnoj replici QMJ-a ili Barre. Revizije izvještaja, procijenjeni datumi objave, vendorove prilagođene cijene i nepotpuni delisting podaci ostaju ograničenja. Sve smo razdoblje već istraživali: podjela prije/poslije 2017. nije netaknuti OOS.</p>
    <p><a href="report.html">Pregled i zajedničke usporedbe</a> · <a href="iwv_universe_coverage.csv">IWV članstvo i pokrivenost CSV</a> · <a href="iwv_snapshot_calendar.csv">Datumi korištenih snapshotova</a> · <a href="statistics_summary.csv">Statistike CSV</a> · <a href="matched_comparisons.csv">Uparene usporedbe CSV</a> · <a href="monthly_ic.csv">IC/cosine CSV</a> · <a href="statistics_monthly_returns.csv">Mjesečni prinosi CSV</a> · <a href="fmp_diagnostics.csv">FMP dijagnostika CSV</a> · <a href="manifest.json">Manifest</a> · <a href="validation.json">Provjera izračuna</a> · <a href="../combined_quality__broad_20261112/report.html">Prethodni širi univerzum</a></p><div id="regimeDefinitions"></div></div></details></section>'''


def build_dashboards():
    specs = {s: {**v, 'components': {d: 1 for d in analysis.CFG[v['bundle']]}} for s, v in analysis.SPECS.items()}
    context = SimpleNamespace(ROOT=ROOT, OUT=OUT, SIGNALS=analysis.SIGNALS, CONFIG=analysis.CONFIG)
    dashboards.analysis, dashboards.OUT, dashboards.SPECS = context, OUT, specs
    dashboards.COLORS, dashboards.SHORT = COLORS, SHORT
    dashboards.build(update_indicator_report=False)
    for path in OUT.glob('statistics*.html'):
        page = path.read_text(encoding='utf-8')
        page = page.replace("'profitability'", "'sort_all7'").replace("'profitability_broad'", "'sort_core5'").replace("'earnings_quality'", "'fmp_all7'")
        page = page.replace('Quality · Statistike dimenzija', 'Quality · IWV zajednički faktor')
        page = page.replace('<h1>Quality: statistike dimenzija</h1><p>Mjesečni kvintilni portfelji, IC i cosine, long/short strane i tržišni režimi.</p>',
            '<h1>Quality: IWV univerzum</h1><p>ALL7 i CORE5 · povijesne IWV sastavnice · kvintili i zajednička regresijska procjena FMP-a.</p>')
        page = page.replace('ZASEBNE DIMENZIJE', 'QUALITY VARIJANTE')
        page = page.replace('Zasebni indikatori i zasebni portfelji. Ovdje se ne kombiniraju u ukupni quality portfelj.',
            'Jednaki ponderi dimenzija. Odvojene varijante portfelja i zajednički kontrolni uzorci.')
        page = page.replace('Svaka dimenzija koristi vlastiti potpuni univerzum.',
            'ALL7 i CORE5 koriste tada dostupno IWV članstvo, uz potpune karakteristike i filtre investabilnosti.')
        page = page.replace('Sastavnice, pokrivenost i presjek dionica', 'Sažetak, uparene usporedbe i scoreovi dionica')
        page = page.replace('${s.componentCount} mjere', '${s.componentCount} dimenzija')
        page = page.replace('Dimenzije imaju različite potpune univerzume.',
            'Provjeri oznake uzorka; izravne kontrolirane usporedbe dostupne su u pregledu.')
        page = page.replace('Mjesečni prosjeci kompozita u odabranim režimskim mjesecima, uz vlastite formacijske uzorke. Mjeseci bez indikatora nemaju IC. Izvorni cosine koristi interni score/50 − 1. IC ostaje statistika dioničkog signala pri svim prikazima strana i troškova. HAC koristi tri laga.',
            'IC i cosine odnose se samo na šest kombiniranih dioničkih scoreova, prije izrade portfelja. FMP nema zaseban IC u ovoj tablici. Izvorni cosine koristi prosjek standardiziranih dimenzija. Mjeseci bez scorea nemaju IC; HAC koristi tri laga. Odabir strane/troškova ne mijenja IC.')
        # Only stock scores have cross-sectional IC. Do not fabricate one for FMP.
        match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S)
        payload = json.loads(match.group(1))
        keep = [i for i, s in enumerate(payload['ic']['signals']) if s['id'] in analysis.SORT_SIGNALS]
        payload['ic']['signals'] = [payload['ic']['signals'][i] for i in keep]
        payload['ic']['values'] = [[r[i] for i in keep] for r in payload['ic']['values']]
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
        page = page[:match.start(1)] + encoded + page[match.end(1):]
        page, count = re.subn(r"^    \$\('legNote'\)\.textContent=state\.leg==='long_short'\?'Gornjih 20%[^\n]*\n[^\n]*\+' Vlastiti mjesečni promet i troškovi; u neaktivnim mjesecima RF\.';",
            "    $('legNote').textContent=(state.focal.startsWith('fmp_')?'Zajednički regresijski FMP; netirane težine svih dimenzija. ':'Kvintilni portfelj kombiniranog quality scorea. ')+(state.leg==='long_short'?'Long +100% / short −100%.':state.leg==='long'?'Zaseban long +100%.':'Zaseban short −100%; pozitivan prinos znači short dobit.')+' Vlastiti promet i troškovi; u neaktivnim mjesecima RF.';",
            page, count=1, flags=re.M)
        assert count == 1, 'Dynamic strategy description'
        page, count = re.subn(r'<section class="panel methodology" id="methodology">.*?</details></section>', lambda _: methodology(), page, count=1, flags=re.S)
        assert count == 1
        page = page.replace('quality_dimensions_${DATA.scenario}', 'combined_quality_iwv_${DATA.scenario}')
        page = page.replace("const headers=['signal_date'", "const headers=['universe','signal_date'")
        page = page.replace('rows=currentStrategyRows.map(m=>[m.date,', "rows=currentStrategyRows.map(m=>['IWV_historical',m.date,")
        path.write_text(page, encoding='utf-8')


def build_overview():
    summary = pd.read_csv(OUT / 'statistics_summary.csv')
    ic = pd.read_csv(OUT / 'ic_summary.csv')
    pairs = pd.read_csv(OUT / 'matched_comparisons.csv')
    coverage = pd.read_csv(OUT / 'coverage.csv')
    latest = pd.read_csv(OUT / 'latest_scores.csv')
    diagnostics = pd.read_csv(OUT / 'fmp_diagnostics.csv')
    factor_summary = pd.read_csv(OUT / 'factor_summary.csv')
    universe_coverage = pd.read_csv(OUT / 'iwv_universe_coverage.csv')
    metadata = {s: {**v, 'dimensions': len(analysis.CFG[v['bundle']])} for s, v in analysis.SPECS.items()}
    payload = dict(specs=metadata, summary=summary.to_dict('records'), ic=ic.to_dict('records'),
        comparisons=pairs.to_dict('records'), coverage=coverage.to_dict('records'),
        latest=latest.to_dict('records'), diagnostics=diagnostics.to_dict('records'), factorSummary=factor_summary.to_dict('records'),
        universeCoverage=universe_coverage.to_dict('records'))
    encoded = json.dumps(json_clean(payload), ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    options = ''.join(f'<option value="{s}">{html.escape(v["label"])}</option>' for s, v in metadata.items())
    scenarios = ''.join(f'<option value="{s["name"]}">{s["label"]}</option>' for s in analysis.CONFIG['scenarios'])
    page = '''<!doctype html><html lang="hr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23365eea'/%3E%3Ctext x='7' y='24' fill='white' font-size='25'%3EQ%3C/text%3E%3C/svg%3E">
<title>Quality · ALL7 i CORE5</title><style>
:root{--ink:#182b43;--muted:#617386;--blue:#365eea;--line:#dce5ed}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.6 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:30px 22px 65px}h1{font-size:clamp(28px,4vw,44px);line-height:1.2;margin:12px 0}h2{font-size:21px;margin:0 0 14px}p{margin:10px 0}a{color:var(--blue)}.brand{font-size:12px;letter-spacing:.15em;color:var(--blue);font-weight:750}.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:24px;margin:20px 0}.muted{color:var(--muted)}.note{border-left:4px solid #d3a246;background:#fff9eb;padding:15px 18px}.controls{display:flex;flex-wrap:wrap;gap:14px}label{flex:1;min-width:180px;font-size:13px;color:var(--muted)}select,input{display:block;font:inherit;color:var(--ink);width:100%;padding:10px;background:#fff;border:1px solid #c4d3df;border-radius:8px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}.metric{background:#f0f4fb;border-radius:9px;padding:15px}.metric strong{display:block;font-size:26px}.metric span{color:var(--muted);font-size:12px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}tr.selected{background:#f0f4ff}.button{display:inline-block;padding:11px 18px;background:var(--blue);color:#fff;border-radius:8px;text-decoration:none}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.tag{font-size:12px;background:#eef2f9;border-radius:5px;padding:3px 7px}footer{font-size:12px;color:var(--muted)}@media(max-width:700px){main{padding:20px 12px}.panel{padding:17px}.metrics{grid-template-columns:repeat(2,1fr)}.two{display:block}.scroll table{min-width:750px}}
</style></head><body><main><header><div class="brand">QUALITY / RESEARCH</div><h1>Quality: ALL7 i CORE5</h1><p class="muted">Jednaki ponderi dimenzija · mjesečni kvintili · zajednički regresijski FMP · upareni uzorci.</p><p><a href="../quality_dimensions/report.html">Pojedine dimenzije</a> · <a href="../quality_literature/report.html">Literatura</a> · <a href="../joint_fmp/report.html">Prethodni FMP profitabilnosti</a></p></header>
<div class="two"><section class="panel"><h2>ALL7</h2><p>Sedam dimenzija, svaka <strong>1/7</strong>: profitabilnost, kvaliteta dobiti, financijska sigurnost, stabilnost, petogodišnji rast profitabilnosti, investicijska disciplina te payout i razrjeđivanje.</p><p class="muted">Povijest od 2012. Širi trokomponentni indikator profitabilnosti.</p></section><section class="panel"><h2>CORE5</h2><p>Bez petogodišnjeg rasta i payouta. Pet dimenzija, svaka <strong>1/5</strong>. Zasebno je prikazan vlastiti uzorak i isti skup dionica kao kod ALL7.</p><p class="muted">Povijest od 2007. Sve sastavnice obvezne, bez dopunjavanja nulom.</p></section></div>
<section class="panel"><div class="controls"><label>Varijanta<select id="signal">__OPTIONS__</select></label><label>Razdoblje<select id="period"><option value="available_history">Raspoloživa povijest</option><option value="common_history">Zajednički početak: 2012.</option><option value="full">Cijeli kalendar, RF prije indikatora</option><option value="early">Prije 2017.</option><option value="late">Od 2017.</option></select></label></div><div class="controls" style="margin-top:14px"><label>Troškovi<select id="scenario">__SCENARIOS__</select></label><label>Strana<select id="leg"><option value="long_short">Long–short</option><option value="long">Long · 100%</option><option value="short">Short · 100%</option></select></label><label>Nepoznati izlazi<select id="policy"><option value="flat_unknown">Osnovno: underlying prinos 0</option><option value="adverse_unknown">Nepovoljan ishod za svaku stranu</option></select></label></div><div class="metrics" id="metrics"></div><p id="coverageNote" class="muted"></p><p id="strategyNote"></p><a class="button" id="dashboard" href="statistics.html">Grafovi, više serija i režimski filtri →</a></section>
<section class="panel"><h2>Usporedba svih varijanti</h2><p class="muted">Poštuje odabrano razdoblje, stranu, troškove i nepoznate izlaze. Kod raspoložive povijesti datumi se mogu razlikovati; za kontroliranu usporedbu koristi tablicu ispod.</p><div class="scroll"><table><thead><tr><th>Varijanta</th><th>CAGR</th><th>Vol.</th><th>Sharpe / RF</th><th>Drawdown</th><th>Aktivni / svi mj.</th><th>Promet / mj.</th></tr></thead><tbody id="comparisonBody"></tbody></table></div></section>
<section class="panel"><h2>Uparene usporedbe</h2><p class="muted">Obje strane para uvijek imaju iste aktivne mjesece. Oznaka „iste dionice” dodatno zahtijeva isti formacijski univerzum. CAGR se ponovno računa na zajedničkim datumima. Δ je godišnji aritmetički prosjek mjesečne razlike, uz HAC t-statistiku (3 laga).</p><label style="max-width:300px">Razdoblje uparenih testova<select id="pairPeriod"><option value="full">Cijela zajednička povijest</option><option value="early">Prije 2017.</option><option value="late">Od 2017.</option></select></label><div class="scroll"><table><thead><tr><th>Usporedba A − B</th><th>Mj.</th><th>CAGR A</th><th>CAGR B</th><th>Δ prosjek/god.</th><th>HAC t</th><th>Korelacija</th></tr></thead><tbody id="pairBody"></tbody></table></div></section>
<section class="panel"><h2>IC i cosine kombiniranog dioničkog signala</h2><p class="muted">FMP varijante ovdje prikazuju IC pripadajućeg kombiniranog scorea na istom univerzumu. To nije IC regresijskog portfelja. Troškovi i izbor long/short strane ne mijenjaju IC.</p><div class="scroll"><table><thead><tr><th>Score</th><th>Spearman IC</th><th>HAC t</th><th>Cosine · interni signal</th><th>Cosine · centrirani rangovi</th><th>Mj.</th></tr></thead><tbody id="icBody"></tbody></table></div></section>
<section class="panel"><h2>Posljednji presjek dionica</h2><p id="latestNote" class="muted"></p><div class="controls"><label>Pretraga tickera / security_id<input id="search" placeholder="npr. AAPL"></label><label>Redoslijed<select id="order"><option value="desc">Najviši quality score</option><option value="asc">Najniži quality score</option></select></label></div><div class="scroll"><table><thead><tr><th>Ticker</th><th>Security ID</th><th>Quality score 0–100</th><th>Interni z-signal</th></tr></thead><tbody id="stocks"></tbody></table></div><p class="muted">Do 100 rezultata. Za FMP ovo je pripadajući score; stvarni FMP ponderi ovise i o zajedničkoj regresiji.</p></section>
<div class="note">Istraživački rezultati na već promatranom uzorku. Ponderi nisu prilagođeni prinosima. Pozitivan bruto rezultat ne dokazuje premiju nakon troškova. U cash mjesecima ostvaruje se RF; dodatni promet pri režimskom prebacivanju nije uključen.</div>
__METHOD__<footer>Samostalan HTML · podaci ugrađeni · bez internetske veze</footer></main><script id="overview-data" type="application/json">__DATA__</script><script>
(()=>{'use strict';const D=JSON.parse(document.getElementById('overview-data').textContent),$=id=>document.getElementById(id);const fmt=(x,n=2)=>Number.isFinite(x)?x.toLocaleString('hr-HR',{minimumFractionDigits:n,maximumFractionDigits:n}):'—',pct=x=>Number.isFinite(x)?fmt(x*100)+'%':'—',esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={all7_vs_core5_same_stocks:'ALL7 − CORE5 · iste dionice',all7_vs_core5_own_stocks:'ALL7 − CORE5 · vlastiti univerzumi',all7_fmp_vs_sort:'ALL7 FMP − kvintili · iste dionice',core5_fmp_vs_sort:'CORE5 FMP − kvintili · iste dionice',all7_vs_core5_fmp_same_stocks:'ALL7 FMP − CORE5 FMP · iste dionice',all7_extra_beta_vol_controls:'ALL7 FMP beta/vol − osnovni FMP',core5_extra_beta_vol_controls:'CORE5 FMP beta/vol − osnovni FMP'};
function render(){const s=$('signal').value,period=$('period').value,scenario=$('scenario').value,leg=$('leg').value,policy=$('policy').value,spec=D.specs[s];const selected=D.summary.filter(r=>r.period===period&&r.scenario===scenario&&r.leg===leg&&r.missing_policy===policy),row=selected.find(r=>r.signal===s);const fields=[['CAGR',pct(row.cagr)],['Sharpe prema RF-u',fmt(row.sharpe_rf)],['Godišnja volatilnost',pct(row.annual_vol)],['Drawdown · mjesečni',pct(row.monthly_max_drawdown)],['Aktivni / svi mjeseci',row.active_months+' / '+row.months],['Cash / RF mjeseci',row.cash_months],['Prosječan promet / aktivni mj.',pct(row.mean_turnover_active)],['Pozitivni mjeseci',pct(row.positive_month_fraction)]];$('metrics').innerHTML=fields.map(([k,v])=>`<div class="metric"><strong>${v}</strong><span>${k}</span></div>`).join('');
const coverage=D.coverage.filter(r=>r.universe===spec.universe),last=coverage.at(-1);$('coverageNote').textContent=`Formacijski univerzum: ${spec.universe}. Posljednji datum ${last.signal_date}: ${last.scored_n} dionica od ${last.eligible_n} podobnih. Odabrani prozor držanja: ${row.first_entry} – ${row.last_exit}.`;
$('strategyNote').textContent=spec.kind==='fmp'?'Zajednička regresija dimenzija; sirove FMP težine kombinirane i netirane prije skaliranja. '+(spec.controls==='defensive'?'Dodane kontrole bete i rezidualne volatilnosti.':'Kontrole: veličina, value, momentum i nelikvidnost.'):'Gornji i donji kvintil kombiniranog scorea, jednaki ponderi unutar strane.';
$('comparisonBody').innerHTML=Object.keys(D.specs).map(id=>{const r=selected.find(x=>x.signal===id);return `<tr data-signal="${id}" class="${id===s?'selected':''}"><td>${D.specs[id].label}</td><td data-metric="cagr" data-value="${r.cagr}">${pct(r.cagr)}</td><td>${pct(r.annual_vol)}</td><td>${fmt(r.sharpe_rf)}</td><td>${pct(r.monthly_max_drawdown)}</td><td>${r.active_months} / ${r.months}</td><td>${pct(r.mean_turnover_active)}</td></tr>`;}).join('');
const pairs=D.comparisons.filter(r=>r.period===$('pairPeriod').value&&r.scenario===scenario&&r.leg===leg&&r.missing_policy===policy);$('pairBody').innerHTML=pairs.map(r=>`<tr data-comparison="${r.comparison}"><td>${names[r.comparison]}</td><td>${r.months}</td><td>${pct(r.cagr_a)}</td><td>${pct(r.cagr_b)}</td><td>${pct(r.annualized_mean_difference)}</td><td>${fmt(r.difference_hac_t)}</td><td>${fmt(r.return_correlation)}</td></tr>`).join('');
let scoreId=spec.kind==='sort'?s:Object.keys(D.specs).find(id=>D.specs[id].kind==='sort'&&D.specs[id].bundle===spec.bundle&&D.specs[id].universe===spec.universe);const icPeriod=['full','available_history'].includes(period)?'full':period,ic=D.ic.find(r=>r.signal===scoreId&&r.period===icPeriod);$('icBody').innerHTML=`<tr><td>${D.specs[scoreId].label}</td><td>${fmt(ic.mean_ic_spearman,4)}</td><td>${fmt(ic.ic_spearman_hac_t)}</td><td>${fmt(ic.mean_ic_cosine_raw,4)}</td><td>${fmt(ic.mean_ic_cosine_rank_centered,4)}</td><td>${ic.months}</td></tr>`;
const f=period==='available_history'?row.first_active_signal_date.slice(0,7):period==='common_history'?'2012-01':period==='early'?'2007-01':period==='late'?'2017-01':'2007-01';let entry=new Date(f+'-01T12:00:00Z');entry.setUTCMonth(entry.getUTCMonth()+1);const from=entry.toISOString().slice(0,7),to=period==='early'?'2017-01':'2026-08';const filename=scenario==='gross_unconstrained'&&policy==='flat_unknown'?'statistics.html':`statistics__${scenario}__${policy}.html`;$('dashboard').href=filename+'#'+new URLSearchParams({series:s,leg,from,to,lines:[s,'sort_core5','fmp_all7'].filter((x,i,a)=>a.indexOf(x)===i).join(',')});
const query=$('search').value.trim().toLowerCase(),direction=$('order').value==='asc'?1:-1;const stocks=D.latest.filter(r=>Number.isFinite(r[s+'_score'])&&(!query||String(r.ticker).toLowerCase().includes(query)||String(r.security_id).toLowerCase().includes(query))).sort((a,b)=>direction*(a[s+'_score']-b[s+'_score']));$('latestNote').textContent=`${D.latest[0].signal_date} · ${stocks.length} dionica prema odabranom filtru.`;$('stocks').innerHTML=stocks.slice(0,100).map(r=>`<tr><td>${esc(r.ticker)}</td><td>${esc(r.security_id)}</td><td>${fmt(r[s+'_score'])}</td><td>${fmt(r[s+'_signal'],4)}</td></tr>`).join('');window.combinedQualitySnapshot={signal:s,period,scenario,leg,policy,statistics:row,matched:pairs,scoreSignal:scoreId,ic,latestCount:stocks.length};}
for(const id of ['signal','period','scenario','leg','policy','pairPeriod','order'])$(id).addEventListener('change',render);$('search').addEventListener('input',render);render();})();
</script></body></html>'''
    factor_panel = '''<section class="panel" id="factorPanel"><h2>Zajednička regresija: procijenjeni prinosi dimenzija</h2><p class="muted">Sirovi koeficijenti za jediničnu dimenzijsku izloženost, godišnji aritmetički prosjek i HAC t (3 laga). Ovo nisu CAGR ni prinosi portfelja skaliranih na bruto 2. Osnovni nepoznati underlying prinosi su nula; troškovi i izbor strane ne mijenjaju ove regresijske koeficijente.</p><label style="max-width:400px">Regresijski model<select id="factorModel"></select></label><div class="scroll"><table><thead><tr><th>Dimenzija</th><th>Prosjek/god.</th><th>HAC t</th><th>Mj.</th><th>Poznati ishodi</th></tr></thead><tbody id="factorBody"></tbody></table></div></section>'''
    page = page.replace('<section class="panel"><h2>Posljednji presjek dionica</h2>', factor_panel + '<section class="panel"><h2>Posljednji presjek dionica</h2>')
    factor_js = '''
const factorNames=__FACTOR_NAMES__;
const factorOptions=Object.entries(D.specs).filter(([id,v])=>v.kind==='fmp');$('factorModel').innerHTML=factorOptions.map(([id,v])=>`<option value="${id}">${v.label}</option>`).join('');
function renderFactors(){const period=['full','available_history'].includes($('period').value)?'full':$('period').value;const rows=D.factorSummary.filter(r=>r.signal===$('factorModel').value&&r.period===period&&factorNames[r.factor]);$('factorBody').innerHTML=rows.map(r=>`<tr data-factor="${r.factor}"><td>${factorNames[r.factor]}</td><td data-coefficient="${r.annualized_mean_coefficient}">${pct(r.annualized_mean_coefficient)}</td><td>${fmt(r.coefficient_hac_t)}</td><td>${r.months}</td><td>${pct(r.mean_observed_fraction)}</td></tr>`).join('');}
$('factorModel').addEventListener('change',renderFactors);$('period').addEventListener('change',renderFactors);$('signal').addEventListener('change',()=>{if(D.specs[$('signal').value].kind==='fmp')$('factorModel').value=$('signal').value;renderFactors();});renderFactors();
'''.replace('__FACTOR_NAMES__', json.dumps({d: analysis.dimensions.SPECS[d]['label'] for d in analysis.CFG['all7']}, ensure_ascii=False))
    page = page.replace("$('search').addEventListener('input',render);render();})();", "$('search').addEventListener('input',render);render();" + factor_js + "})();")
    fixed = summary[(summary.period == 'common_history') & (summary.leg == 'long_short') & (summary.missing_policy == 'flat_unknown')]
    def pct_text(value):
        return f'{100 * value:.2f}%'.replace('.', ',')
    test_rows = []
    for signal in ['sort_all7', 'sort_core5', 'fmp_all7', 'fmp_core5']:
        values = fixed[fixed.signal.eq(signal)].set_index('scenario')
        cells = ''.join('<td>' + pct_text(values.loc[scenario['name'], 'cagr']) + '</td>' for scenario in analysis.CONFIG['scenarios'])
        test_rows.append('<tr><td>' + metadata[signal]['label'] + '</td>' + cells + '</tr>')
    comparison = pairs[(pairs.comparison == 'all7_vs_core5_same_stocks') & (pairs.period == 'full') &
        (pairs.scenario == 'gross_unconstrained') & (pairs.leg == 'long_short') & (pairs.missing_policy == 'flat_unknown')].iloc[0]
    t_text = f'{comparison.difference_hac_t:.2f}'.replace('.', ',')
    fixed_note = '''<section class="panel"><h2>IWV: prvi rezultati</h2><p>Usporedba koristi isti početak 2012. i long–short portfelje, uz osnovne nepoznate izlaze. Svi portfelji ograničeni su na povijesno IWV članstvo, ali zahtjevi za karakteristikama stvaraju različite poduzorke; tablica uparenih usporedbi niže dodatno kontrolira dionice.</p><div class="scroll"><table><thead><tr><th>Varijanta</th><th>Bruto CAGR</th><th>5 bps + 0,5% posudba</th><th>10 bps + 3% posudba</th></tr></thead><tbody>__FIXED_ROWS__</tbody></table></div><p class="muted">Fiksni pregled od formacije siječanj 2012. do srpanj 2026.; ne mijenja se filtrima niže. HAC t mjesečne razlike ALL7 − CORE5 na istim dionicama iznosi __PAIR_T__. Razliku treba procjenjivati uz troškove i nesigurnost na ovom već promatranom uzorku.</p></section>'''.replace('__FIXED_ROWS__', ''.join(test_rows)).replace('__PAIR_T__', t_text)
    page = page.replace('<div class="two">', fixed_note + '<div class="two">', 1)
    last = universe_coverage.iloc[-1]
    latest_counts = coverage[coverage.signal_date.eq(coverage.signal_date.max())].set_index('universe')
    universe_panel = f'''<section class="panel" id="universePanel"><h2>IWV članstvo i pokrivenost</h2><p><strong>Povijesni snapshotovi iz etfconstituents</strong>, zadnji tada dostupan snapshot, točan puni SID i provjeren identitet. Prva IWV formacija je 30. 6. 2009.; raniji mjeseci su RF. CORE5 koristi povijest od tada, a ALL7 od 2012. Nedostajuće članstvo ne popunjava se budućim snapshotom.</p><div class="scroll"><table><thead><tr><th>Presjek {last.signal_date}</th><th>Broj dionica</th></tr></thead><tbody><tr><td>Objavljeni IWV članovi</td><td data-universe="holdings">{int(last.holdings_n)}</td></tr><tr><td>Članovi s provjerenim identitetom</td><td data-universe="usable">{int(last.holdings_identity_usable_n)}</td></tr><tr><td>Prisutni u našem mjesečnom panelu</td><td data-universe="panel">{int(last.panel_matched_members_n)}</td></tr><tr><td>Nakon postojećih filtara investabilnosti</td><td data-universe="eligible">{int(last.iwv_eligible_n)}</td></tr><tr><td>Potpuni ALL7</td><td data-universe="all7">{int(latest_counts.loc['all7','scored_n'])}</td></tr><tr><td>Potpuni CORE5</td><td data-universe="core5">{int(latest_counts.loc['core5','scored_n'])}</td></tr></tbody></table></div><p class="muted">Snapshot na posljednji datum: {last.target_snapshot_date}; stvarni izvor {last.source_snapshot_date}. Ponderi IWV-a ne koriste se kao ponderi strategije. <a href="iwv_universe_coverage.csv">Mjesečna pokrivenost</a> · <a href="iwv_snapshot_calendar.csv">Lineage snapshotova</a> · <a href="../combined_quality__broad_20261112/report.html">Arhivirani rezultati prethodnog šireg univerzuma</a>.</p></section>'''
    page = page.replace('<div class="two">', universe_panel + '<div class="two">', 1)
    changed = pd.read_csv(OUT / 'universe_change_comparison.csv')
    changed = changed[(changed.period == 'common_history') & (changed.leg == 'long_short') &
        (changed.scenario == 'gross_unconstrained') & (changed.missing_policy == 'flat_unknown')].set_index('signal')
    change_rows = ''.join(f'<tr><td>{metadata[s]["label"]}</td><td>{pct_text(changed.loc[s,"cagr_broad"])}</td><td>{pct_text(changed.loc[s,"cagr_iwv"])}</td><td>{changed.loc[s,"difference_hac_t"]:.2f}</td></tr>'
        for s in ['sort_all7', 'sort_core5', 'fmp_all7', 'fmp_core5'])
    change_panel = '''<section class="panel"><h2>Učinak promjene univerzuma</h2><p class="muted">Bruto long–short, osnovni nepoznati izlazi, zajednički aktivni datumi od 2012. Oba izračuna koriste istu definiciju scorea i portfelja. IWV mijenja članstvo i sve rangove koji se iz njega računaju. HAC t odnosi se na mjesečnu razliku IWV − prethodni širi univerzum.</p><div class="scroll"><table><thead><tr><th>Varijanta</th><th>Širi univerzum · CAGR</th><th>IWV · CAGR</th><th>HAC t razlike</th></tr></thead><tbody>__ROWS__</tbody></table></div><p><a href="universe_change_comparison.csv">Sve uparene usporedbe univerzuma, strana i troškova</a></p></section>'''.replace('__ROWS__', change_rows)
    page = page.replace('<div class="two">', change_panel + '<div class="two">', 1)
    page = page.replace('<title>Quality · ALL7 i CORE5</title>', '<title>Quality · IWV · ALL7 i CORE5</title>')
    page = page.replace('<h1>Quality: ALL7 i CORE5</h1>', '<h1>Quality: IWV univerzum</h1>')
    page = page.replace('Povijest od 2007. Sve sastavnice obvezne', 'Povijest IWV univerzuma od lipnja 2009. Sve sastavnice obvezne')
    page = page.replace('>Pojedine dimenzije</a>', '>Pojedine dimenzije · raniji širi univerzum</a>')
    for key, value in {'__OPTIONS__': options, '__SCENARIOS__': scenarios, '__METHOD__': methodology(), '__DATA__': encoded}.items():
        page = page.replace(key, value)
    (OUT / 'report.html').write_text(page, encoding='utf-8')


def update_links():
    # Update source generators, then their current HTML. Numeric results stay intact.
    source = ROOT / 'scripts/build_quality_dimensions_report.py'
    old = '<p><a href="../quality_literature/report.html">Literatura i metodologije</a>'
    new = '<p><a href="../combined_quality/report.html"><strong>IWV quality faktor: ALL7 i CORE5</strong></a> · <a href="../quality_literature/report.html">Literatura i metodologije</a>'
    text = source.read_text(encoding='utf-8')
    if old in text:
        source.write_text(text.replace(old, new), encoding='utf-8')
    main = ROOT / 'results/quality_dimensions/report.html'
    text = main.read_text(encoding='utf-8')
    if old in text:
        main.write_text(text.replace(old, new), encoding='utf-8')
    for path in [source, main]:
        text = path.read_text(encoding='utf-8')
        text = text.replace('Zajednički quality faktor: ALL7 i CORE5', 'IWV quality faktor: ALL7 i CORE5')
        text = text.replace('<h1>Kompozitni indikatori dimenzija</h1>', '<h1>Dimenzije · raniji širi univerzum</h1>')
        path.write_text(text, encoding='utf-8')
    for name, key in [('manifest.json', 'output_sha256'), ('statistics_manifest.json', 'outputs')]:
        path = main.parent / name
        content = json.loads(path.read_text(encoding='utf-8'))
        content[key]['report.html'] = analysis.sha(main)
        content['combined_quality_report'] = '../combined_quality/report.html'
        content['combined_quality_link_note'] = 'Only report navigation changed; numeric outputs preserved. Earlier UI receipt describes its original HTML; combined_quality/ui_validation.json verifies the current link.'
        path.write_text(json.dumps(content, indent=2), encoding='utf-8')


def main():
    build_dashboards()
    build_overview()
    update_links()
    print('Built combined quality overview, six interactive dashboards, and existing-report link.', flush=True)


if __name__ == '__main__':
    main()
