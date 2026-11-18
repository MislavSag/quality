"""Apply project labels without changing predictor names or experiment data."""
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'scripts/report_assets/portfolio.html'
    t=p.read_text(encoding='utf-8')
    t=t.replace('Profitabilnost · Istraživački izvještaj','Quality · Profitabilna komponenta')
    t=t.replace('Profitabilnost, početak izvještaja','Quality, početak izvještaja')
    t=t.replace('>P<span>↗</span></span>PROFIT<span','>Q<span>↗</span></span>QUALITY<span')
    t=t.replace('PROFIT / RESEARCH','QUALITY / RESEARCH')
    t=t.replace("fill='white'%3EP%3C/text", "fill='white'%3EQ%3C/text")
    t=t.replace('<h1>Portfelj profitabilnosti</h1><p>Četiri faktora, različita tržišna okruženja. Istraži kada rade bolje.</p>',
                '<h1>Quality: profitabilna komponenta</h1><p>Dosadašnji portfelj profitabilnosti u širem projektu quality faktora.</p>')
    banner='<p class="panel-footnote">Profitabilnost je jedna dimenzija kvalitete. Ove rezultate još ne tumačimo kao test potpunog quality faktora. <a href="../quality_literature/report.html">Pregled literature i metodologija ↗</a></p>'
    t=re.sub(r'<p class="panel-footnote">Profitabilnost je jedna dimenzija.*?</p>',banner,t,count=1)
    p.write_text(t,encoding='utf-8')
    p=ROOT/'scripts/build_joint_fmp_report.py';t=p.read_text(encoding='utf-8')
    t=t.replace("base_template.replace('Profitabilnost · Istraživački izvještaj','Zajednička regresija · FMP')", "base_template.replace('Quality · Profitabilna komponenta','Quality · FMP profitabilne komponente')")
    t=t.replace("'<h1>Portfelj profitabilnosti</h1><p>Četiri faktora, različita tržišna okruženja. Istraži kada rade bolje.</p>'", "'<h1>Quality: profitabilna komponenta</h1><p>Dosadašnji portfelj profitabilnosti u širem projektu quality faktora.</p>'")
    t=t.replace('<h1>Profitabilnost iz zajedničke regresije</h1>','<h1>Quality: FMP profitabilne komponente</h1>')
    p.write_text(t,encoding='utf-8')
    changes={
        'analyze_quality.py':[('Početna analiza profitabilnosti','Quality: početna analiza profitabilne komponente')],
        'analyze_factor_portfolio.py':[('Portfelj četiri profitability faktora','Quality: portfelj četiriju faktora profitabilnosti')],
        'analyze_regimes.py':[('<title>Profitabilnost kroz režime</title>','<title>Quality: profitabilnost kroz režime</title>'),('<h1>Profitabilnost kroz tržišne režime</h1>','<h1>Quality: profitabilna komponenta kroz režime</h1>')],
        'backtest_monthly_long_short.py':[('Mjesečni long-short profitabilnosti','Quality: mjesečni long-short profitabilne komponente')],
        'build_quality_profitability_bundle.py':[('<h1>PROFIT_BUNDLE4 — mjesečni indikator profitabilnosti</h1>','<h1>Quality: PROFIT_BUNDLE4 — profitabilna komponenta</h1>')]}
    for name,pairs in changes.items():
        p=ROOT/'scripts'/name;t=p.read_text(encoding='utf-8')
        for old,new in pairs:t=t.replace(old,new)
        p.write_text(t,encoding='utf-8')
    with zipfile.ZipFile(ROOT/'data/reference/quality_rename_20261112/original_project_sources.zip') as z:
        readme=z.read('README.md').decode('utf-8')
    readme=readme.replace('# Profitability\n','# Quality\n',1)
    readme=readme.replace('Početna mjesečna analiza postojećih fundamentalnih pokazatelja i ekstenzije\nprofitabilnosti objavljene 10. studenoga 2026.', '''Projekt istraživanja **quality faktora**. Dosadašnji izračuni obuhvaćaju
profitabilnu komponentu kvalitete, uključujući mjesečne long–short portfelje,
zajedničke presječne regresije i factor-mimicking portfelje. Potpuni quality
faktor s dodatnim dimenzijama još nije implementiran.

U Positronu otvori mapu `C:/Users/Mislav/projects_r/quality`, odnosno workspace
`C:/Users/Mislav/projects_r/quality/quality.code-workspace`.
Skripte, podaci, rezultati i Git metapodaci fizički su u toj mapi.
Projekt ne ovisi o starim putanjama pod `prof`.
Identitet projekta evidentiran je u `config/project.json`.

Pregled literature nalazi se u [Quality: literatura i metodologije](results/quality_literature/report.html).
Uspoređuje akademske radove, stručne preglede i industrijske metodologije te
razlikuje postojeće izračune od planiranih dimenzija kvalitete.

Nazivi izvornih Parqueta, vendorovih stupaca, povijesnih signalnih oznaka
`PROFIT_BUNDLE4` i regresijskih termina `profitability` zadržavaju izvorno
računovodstveno značenje. Ekstenzija profitabilnosti objavljena je 10. studenoga
2026. Promjena naziva projekta ne predstavlja novi backtest.''')
    replacements=[('analyze_profitability.py','analyze_quality.py'),('build_profitability_bundle.py','build_quality_profitability_bundle.py'),
        ('validate_profitability_bundle.py','validate_quality_profitability_bundle.py'),('config/profitability_bundle.json','config/quality_profitability_bundle.json'),
        ('`results/profitability_bundle/','`results/quality_profitability_bundle/'),('Playwright CLI sesiju `profitability_report`','Playwright CLI sesiju `quality_report`'),
        ('## Profitability bundle','## Quality: bundle profitabilne komponente')]
    for old,new in replacements:readme=readme.replace(old,new)
    readme+='''

## Promjena naziva projekta 12. studenoga 2026.

Glavna skripta je `scripts/analyze_quality.py`; stari `analyze_profitability.py`
je kompatibilni ulaz. Bundle profitabilne komponente koristi
`build_quality_profitability_bundle.py`, `validate_quality_profitability_bundle.py`
i `config/quality_profitability_bundle.json`. Rezultati su dostupni i kroz
`results/quality_profitability_bundle`, koji vodi na istu postojeću pohranu.
Stare konfiguracije i eksperimentalni identifikatori ostaju dostupni.

Izvorni kod i konfiguracije prije promjene naziva arhivirani su u
`data/reference/quality_rename_20261112/original_project_sources.zip`. Provjera
`results/project_rename_validation.json` uspoređuje hash svih prethodnih CSV i
Parquet rezultata te provjerava nove i stare ulazne module. Povijesni manifesti
ostaju zapisi izvornog eksperimenta. Promjena brenda i navigacije zasebno je
evidentirana u `results/project_rename_manifest.json`.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    for name in ['initial_analysis/report.html','regimes/report.html','monthly_long_short/report.html','profitability_bundle/report.html','factor_portfolio/report_static.html']:
        p=ROOT/'results'/name;t=p.read_text(encoding='utf-8')
        t=re.sub(r'<p>Dosada.*?quality_literature/report.html.*?</p>',
            '<p>Dosadašnja analiza profitabilne komponente. Potpuni quality faktor još nije implementiran. <a href="../quality_literature/report.html">Literatura i metodologije</a>.</p>',t,count=1,flags=re.S)
        p.write_text(t,encoding='utf-8')
    print('Quality labels and UTF-8 documentation updated')


if __name__=='__main__':main()
