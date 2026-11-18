"""Build an offline, cited map of quality-factor research and project coverage."""
from collections import Counter
import html
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/quality_literature'
SOURCE=ROOT/'data/reference/quality_literature.json'


def esc(value):return html.escape(str(value),quote=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    registry=json.loads(SOURCE.read_text(encoding='utf-8'))
    sources=registry['sources']
    assert len({s['id'] for s in sources})==len(sources)
    schema=set(pq.read_schema('H:/strategies/characteristics/fundamental_characteristics.parquet').names)
    fields={
        'Profitabilnost':['FUND_RETURN_ON_EQUITY_TTM','FUND_RETURN_ON_ASSETS_TTM','FUND_GROSS_PROFIT_TO_ASSETS_TTM','FUND_CFO_TO_ASSETS_TTM','FUND_FCF_TO_ASSETS_TTM'],
        'Kvaliteta dobiti':['FUND_CASHFLOW_ACCRUALS_TTM','FUND_WORKING_CAPITAL_ACCRUALS_YOY','FUND_CFO_CONVERSION_TTM'],
        'Financijska sigurnost':['FUND_BOOK_LEVERAGE','FUND_DEBT_TO_ASSETS','FUND_NET_DEBT_TO_ASSETS','FUND_INTEREST_COVERAGE_TTM'],
        'Investiranje i kapitalna disciplina':['FUND_ASSET_GROWTH_YOY','FUND_NET_EQUITY_ISSUANCE_YIELD_TTM','FUND_DEBT_PAYDOWN_YIELD_TTM','FUND_SHAREHOLDER_YIELD_WITH_DEBT_TTM'],
        'Rast i stabilnost':['FUND_SALES_GROWTH_YOY']}
    inventory=[dict(dimension=d,column=c,present=c in schema,
                    audit='Provjerena prisutnost polja; formula, povijesna pokrivenost i raspoloživost nisu ovdje ponovno provjerene.')
               for d,names in fields.items() for c in names]
    pd.DataFrame(inventory).to_csv(OUT/'existing_field_inventory.csv',index=False)
    pd.DataFrame([{k:('|'.join(v) if isinstance(v,list) else v) for k,v in s.items()} for s in sources]).to_csv(OUT/'sources.csv',index=False)
    cards=[]
    for s in sorted(sources,key=lambda s:(s['priority'],s['category'],-s['year'])):
        tags=''.join(f'<span class="tag">{esc(d)}</span>' for d in s['dimensions'])
        doi=f' · DOI <a href="https://doi.org/{esc(s["doi"])}">{esc(s["doi"])}</a>' if s['doi'] else ''
        related=f' · <a href="{esc(s["related_url"])}">Dodatni službeni izvor</a>' if s.get('related_url') else ''
        searchable=' '.join([s['title'],s['authors'],s['summary'],*s['dimensions']]).lower()
        cards.append(f'''<article class="source-card" id="{s['id']}" data-category="{esc(s['category'])}" data-dimensions="{esc('|'.join(s['dimensions']))}" data-search="{esc(searchable)}">
        <div class="meta">{esc(s['category'])} · {s['year']} · Prioritet čitanja {s['priority']}</div>
        <h2><a href="{esc(s['url'])}">{esc(s['title'])}</a></h2><p class="citation">{esc(s['authors'])}<br>{esc(s['venue'])}{doi}{related}</p>
        <div class="tags">{tags}</div><p>{esc(s['summary'])}</p><p><strong>Za projekt:</strong> {esc(s['project_use'])}</p>
        <p class="caution"><strong>Ograničenje:</strong> {esc(s['caution'])}</p><details><summary>Opseg pregleda</summary><p>{esc(s['review_basis'])}</p></details></article>''')
    inventory_rows=''.join(f'<tr><td>{esc(d)}</td><td>'+''.join(f'<code>{esc(c)}</code><br>' for c in names)+f'</td><td>{"Polja prisutna" if all(c in schema for c in names) else "Dio polja nedostaje"}</td></tr>' for d,names in fields.items())
    categories=sorted({s['category'] for s in sources})
    dimensions=sorted({d for s in sources for d in s['dimensions']})
    methods=[('Objavljeni QMJ','Profitabilnost + rast profitabilnosti + sigurnost','asness2019'),
             ('MSCI Quality','ROE + nizak D/E + stabilnost dobiti; izmjena obveznih ulaza od 2025.','msci2025'),
             ('S&P Quality','ROE + accruals + financijski leverage','sp2025'),
             ('FTSE Quality','ROA + Δ asset turnover + accruals + CFO/dug; sektorske iznimke','ftse2026')]
    method_rows=''.join(f'<tr><td><a href="#{sid}">{esc(name)}</a></td><td>{esc(definition)}</td></tr>' for name,definition,sid in methods)
    doc=f'''<!doctype html><html lang="hr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23365eea'/%3E%3Ctext x='7' y='24' fill='white' font-family='Arial' font-size='24'%3EQ%3C/text%3E%3C/svg%3E"><title>Quality · Literatura i metodologije</title>
    <style>*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fb;color:#203047;font:15px/1.65 system-ui,sans-serif}}header{{background:white;border-bottom:1px solid #e0e6ef;padding:20px max(20px,calc((100% - 1120px)/2))}}main{{max-width:1160px;margin:30px auto;padding:0 20px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:21px;line-height:1.35}}a{{color:#315ccf;text-underline-offset:3px}}.brand{{letter-spacing:2px;font-weight:700}}.muted,.meta,.citation{{color:#63748d}}.panel,.source-card{{background:white;border:1px solid #e0e6ef;border-radius:14px;padding:24px;margin:20px 0}}.scope{{border-left:4px solid #365eea}}.source-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.source-card{{margin:0;overflow-wrap:anywhere}}.source-card[hidden]{{display:none}}.tag{{display:inline-block;background:#edf2fc;color:#47648f;border-radius:6px;padding:3px 8px;font-size:12px;margin:3px}}.meta{{font-size:12px}}.caution{{background:#fff7e6;padding:12px;border-radius:8px}}.filters{{display:flex;gap:14px;flex-wrap:wrap;align-items:end}}label{{display:flex;flex-direction:column;gap:5px;flex:1;min-width:190px}}input,select,button{{font:inherit;border:1px solid #ccd6e5;border-radius:7px;padding:10px;background:white;color:inherit;max-width:100%}}button{{cursor:pointer}}table{{width:100%;border-collapse:collapse;text-align:left}}td,th{{padding:12px;border-bottom:1px solid #e4e9f1;vertical-align:top}}.table-scroll{{overflow:auto}}code{{font-size:12px;overflow-wrap:anywhere}}li{{margin:9px 0}}details{{color:#63748d}}footer{{color:#63748d;padding:25px 0}}@media(max-width:760px){{.source-grid{{grid-template-columns:1fr}}h1{{font-size:28px}}.panel,.source-card{{padding:18px}}main{{padding:0 12px}}}}</style></head><body>
    <header><span class="brand">QUALITY / RESEARCH</span></header><main><p class="muted">Pregled izvora · 12. studenoga 2026.</p><h1>Quality faktor: literatura, definicije i podaci</h1>
    <p>Pregled obuhvaća {len(sources)} izvora: akademske radove, stručne preglede i industrijske metodologije. {esc(registry['scope'])}</p>
    <section class="panel scope"><h2>Opseg projekta nakon promjene naziva</h2><p>Projekt se sada zove Quality. Dosadašnji testovi analiziraju profitabilnu komponentu: FCF, CFO, bruto dobit i povezane pokazatelje. Promjena naziva ne znači da je već izračunat širi quality faktor.</p>
    <p><a href="../joint_fmp/report.html">FMP izvještaj s long i short prikazom</a> · <a href="../factor_portfolio/report.html">Dosadašnji portfelj faktora</a> · <a href="sources.csv">Popis izvora CSV</a> · <a href="existing_field_inventory.csv">Postojeća polja CSV</a></p></section>
    <section class="panel"><h2>Različite definicije kvalitete</h2><div class="table-scroll"><table><thead><tr><th>Pristup</th><th>Dimenzije i potvrđena verzija</th></tr></thead><tbody>{method_rows}</tbody></table></div>
    <p>Ne postoji jedinstven quality score. Smjer povezanosti s budućim prinosima treba provjeriti zasebno od ekonomske procjene kvalitete poslovanja. Empirijsku raspravu donosi <a href="#hsu2019">Hsu, Kalesnik i Kose (2019)</a>.</p></section>
    <section class="panel"><h2>Što predlažemo za ovaj projekt</h2><p>Ovo je istraživački prijedlog na temelju pregleda izvora; nove kombinacije još nisu izračunate.</p><ol>
    <li>Zadržati postojeću profitabilnost kao zaseban stup i referentni rezultat.</li>
    <li>Usporediti QMJ-like model s jasno navedenim proxyjima, industrijski model s malo deskriptora i model profitabilnosti, accruals, investiranja te payouta/razrjeđivanja. Njihove definicije i rezultate prikazivati odvojeno.</li>
    <li>Pondere prvo zadati po skupinama pokazatelja. Dodavanje više sličnih profitabilnih mjera ne bi smjelo neprimjetno povećati težinu profitabilnosti.</li>
    <li>Razlikovati puni quality portfelj i rezidualni quality nakon neutralizacije. Ako sigurnost uključimo u score, neutralizacija bete i volatilnosti uklanja dio namjerno ciljane kvalitete.</li>
    <li>Provjeriti zaduženost i negativni kapital, sezonalnost, duljinu povijesti za stabilnost dobiti, revizije izvještaja te datume raspoloživosti. Banke i osiguravatelji zahtijevaju poseban tretman.</li>
    <li>Postojeće FMP, long, short, režimske i troškovne usporedbe koristiti nakon provjere novih ulaza. Ne optimizirati izbor definicije na već pregledanim prinosima.</li></ol></section>
    <section class="panel"><h2>Početna inventura naših podataka</h2><p>Provjerena je prisutnost naziva stupaca u postojećem fundamentalnom Parquetu. Ova provjera ne potvrđuje povijesnu pokrivenost, točnu formulu ni point-in-time ispravnost svake nove mjere.</p>
    <div class="table-scroll"><table><thead><tr><th>Skupina</th><th>Postojeći stupci</th><th>Provjera</th></tr></thead><tbody>{inventory_rows}</tbody></table></div>
    <p>Petogodišnji rast profitabilnosti, stabilnost dobiti i potpuni F-score nisu ovim korakom rekonstruirani. Postojeći FUND_QMJ_LIKE_PROFITABILITY_FY ostaje mjera profitabilnosti, a ne puni QMJ.</p></section>
    <section class="panel"><h2>Pretraži izvore</h2><div class="filters"><label>Vrsta izvora<select id="category"><option value="">Sve vrste</option>{''.join(f'<option>{esc(c)}</option>' for c in categories)}</select></label><label>Dimenzija<select id="dimension"><option value="">Sve dimenzije</option>{''.join(f'<option>{esc(d)}</option>' for d in dimensions)}</select></label><label>Naslov, autor ili pojam<input id="query" type="search" placeholder="npr. accruals, QMJ, leverage"></label><button id="reset">Očisti</button></div><p id="count" aria-live="polite">Prikazano {len(sources)} od {len(sources)} izvora</p></section>
    <div class="source-grid">{''.join(cards)}</div><footer>Primarni izvori povezani su uz svaki zapis. Opseg stvarno pregledanog materijala naveden je pod „Opseg pregleda”. Nisu preuzimani zaštićeni puni tekstovi niti su postojeći prinosi ponovno procjenjivani.</footer></main>
    <script>const cards=[...document.querySelectorAll('.source-card')],ids=['category','dimension','query'];function filter(){{const [c,d,q]=ids.map(id=>document.getElementById(id).value.trim().toLocaleLowerCase('hr'));let n=0;for(const card of cards){{const match=(!c||card.dataset.category.toLocaleLowerCase('hr')===c)&&(!d||card.dataset.dimensions.toLocaleLowerCase('hr').split('|').includes(d))&&(!q||card.dataset.search.includes(q));card.hidden=!match;if(match)n++;}}document.getElementById('count').textContent=`Prikazano ${{n}} od ${{cards.length}} izvora`;}}for(const id of ids)document.getElementById(id).addEventListener('input',filter);document.getElementById('reset').addEventListener('click',()=>{{for(const id of ids)document.getElementById(id).value='';filter();}});</script></body></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    counts=dict(Counter(s['category'] for s in sources))
    (OUT/'review_manifest.json').write_text(json.dumps(dict(review_date=registry['review_date'],sources=len(sources),categories=counts,
        selection='targeted primary-source review; not exhaustive or a systematic review',
        field_inventory='schema names only; coverage and formula validation remain separate work'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(sources=len(sources),categories=counts),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
