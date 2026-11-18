"""Offline FMP reports, reusing the existing calendar/regime visualization."""
import html
import json
import re

import numpy as np
import pandas as pd

from analyze_joint_fmp import OUT, ROOT, CFG, PROFIT, FACTORS, sha
from analyze_quality import hac_mean
from build_interactive_report import ASSETS, REGIMES, cash_stats

METHOD_NAMES={'zscore':'Winsorizirani z-score','rank_normal':'Normalizirani rangovi'}
WEIGHT_NAMES={'sqrt_cap':'WLS: korijen kapitalizacije','inverse_specific_variance':'WLS: inverzna specifična varijanca'}
LABELS={'fcf_assets_ttm':'FCF / imovina','cfo_assets_ttm':'CFO / imovina','gp_assets_ttm':'Bruto dobit / imovina'}


def filename(model,scenario):
    return 'report.html' if model==CFG['primary'] and scenario=='gross' else model+'__'+scenario+'.html'


def build():
    monthly=pd.read_csv(OUT/'monthly_returns.csv',parse_dates=['signal_date','entry_date','exit_date'])
    sides=pd.read_csv(OUT/'side_monthly_returns.csv',parse_dates=['signal_date','entry_date','exit_date'])
    side_receipt=json.loads((OUT/'side_manifest.json').read_text())
    assert side_receipt['inputs'][str(OUT/'weights.parquet')]==sha(OUT/'weights.parquet'), 'Stale side weights'
    source=pd.read_csv(ROOT/'results/factor_portfolio/portfolio_monthly.csv',parse_dates=['signal_date','entry_date','exit_date'])
    rf=pd.read_csv(ROOT/'results/factor_portfolio/risk_free_windows.csv',parse_dates=['signal_date','entry_date','exit_date'])
    features=pd.read_csv(ROOT/'results/regimes/regime_features.csv',parse_dates=['signal_date','entry_date','exit_date'])
    coefficients=pd.read_csv(OUT/'factor_returns.csv',parse_dates=['signal_date'])
    diagnostics=pd.read_csv(OUT/'diagnostics.csv',parse_dates=['signal_date'])
    summaries=pd.read_csv(OUT/'summary.csv')
    series=[
        dict(id='portfolio3_gross2',label='FMP: tri faktora, netirana košarica gross 200%',short='FMP 3 · gross 2',color='#365eea'),
        dict(id='portfolio3',label='FMP: tri faktora bez reskaliranja košarice',short='FMP 3 · netirano',color='#8293b5'),
        dict(id='fcf_assets_ttm',label='Zajednički FMP: FCF',short='FMP FCF',color='#4c9f89',component=True),
        dict(id='cfo_assets_ttm',label='Zajednički FMP: CFO',short='FMP CFO',color='#b89b68',component=True),
        dict(id='gp_assets_ttm',label='Zajednički FMP: bruto dobit',short='FMP GP',color='#b988a4',component=True),
        dict(id='sort_common_basket3_gross2',label='Kvintili: tri faktora, isti univerzum, gross 200%',short='Kvintili 3 · gross 2',color='#778bb3'),
        dict(id='sort_common_basket3',label='Kvintili: tri faktora bez reskaliranja košarice',short='Kvintili 3 · netirano',color='#abb6ca'),
        dict(id='sort_fcf_assets_ttm',label='Kvintili: FCF, isti univerzum',short='Sort FCF',color='#8eb5a2'),
        dict(id='sort_cfo_assets_ttm',label='Kvintili: CFO, isti univerzum',short='Sort CFO',color='#b7ab8b'),
        dict(id='sort_gp_assets_ttm',label='Kvintili: bruto dobit, isti univerzum',short='Sort GP',color='#c1a0b4'),
        dict(id='portfolio4',label='FMP tri faktora + javni RMW, bruto',short='FMP 4 bruto',color='#7d56c4'),
        dict(id='original_portfolio4',label='Raniji portfelj četiri faktora, bruto',short='Raniji 4',color='#a1adbf'),
        dict(id='public_rmw',label='Javni RMW, bruto',short='RMW bruto',color='#b17b52',component=True),
        dict(id='risk_free',label='Risk-free / cash',short='RF',color='#74869a')]
    for s in series:
        s['legs']=['long_short'] if s['id'] in ['portfolio4','original_portfolio4','public_rmw','portfolio3','sort_common_basket3'] else ['long_short','long','short']
    next(s for s in series if s['id']=='portfolio3_gross2')['sideLabel']='FMP: netirana košarica triju faktora'
    next(s for s in series if s['id']=='portfolio3_gross2')['sideShort']='FMP 3'
    next(s for s in series if s['id']=='sort_common_basket3_gross2')['sideLabel']='Kvintili: netirana košarica triju faktora, isti univerzum'
    next(s for s in series if s['id']=='sort_common_basket3_gross2')['sideShort']='Kvintili 3'
    legs=[dict(id='long_short',label='Long–short'),dict(id='long',label='Long · 100%'),dict(id='short',label='Short · 100%')]
    base_template=(ASSETS/'portfolio.html').read_text(encoding='utf-8')
    base_template=base_template.replace('Quality · Profitabilna komponenta','Quality · FMP profitabilne komponente')
    base_template=base_template.replace('<h1>Quality: profitabilna komponenta</h1><p>Dosadašnji portfelj profitabilnosti u širem projektu quality faktora.</p>',
        '<h1>Quality: FMP profitabilne komponente</h1><p>Transformirane karakteristike → zajednička procjena prinosa → factor-mimicking portfelji.</p>')
    base_template=base_template.replace('Jednake kapitalne težine.<br>Mjesečni rebalans, bez optimizacije.',
        'Tri lokalna FMP-a procijenjena zajedno. Svaki cilja gross 200%. Košaricu prikazujemo nakon netiranja, s ponovnim skaliranjem na gross 200% i bez njega. Javni RMW je vanjski benchmark.')
    base_template=base_template.replace('STRUKTURA PORTFELJA','KOMPONENTE ZA USPOREDBU')
    base_template=re.sub(r'<section class="panel ic-panel">.*?</section>','__FMP_DIAGNOSTICS__',base_template,flags=re.S)
    js=(ASSETS/'portfolio.js').read_text(encoding='utf-8')
    js=js.replace("focal:'portfolio4'","focal:'portfolio3_gross2'")
    js=js.replace("visible:new Set(['portfolio4','public_rmw'])","visible:new Set(['portfolio3_gross2','sort_common_basket3_gross2','public_rmw'])")
    js=js.replace('<strong>25%</strong>',"<strong>${s.id==='public_rmw'?'Javni':'FMP'}</strong>")
    js=js.replace("function renderIC() {","function renderIC() {\n    if (!$('icBody')) return;")
    # Four-way comparison remains gross even in the local assumed-cost report.
    js=js.replace("${s.id==='portfolio4'?'<span class=\"table-badge\">25% × 4</span>':''}",
                  "${s.id==='portfolio4'?'<span class=\"table-badge\">25% × 4 · bruto</span>':''}")
    all_regimes=[];regression_summary=[];comparison=[]
    models=[m+'__'+w for m in CFG['transformations'] for w in CFG['weighting']]
    for model in models:
        method,weighting=model.split('__')
        title=METHOD_NAMES[method]+' · '+WEIGHT_NAMES[weighting]
        diag=diagnostics[diagnostics.model==model]
        raw_table=[]
        for signal in PROFIT:
            part=coefficients[(coefficients.model==model)&(coefficients.factor==signal)]
            mean,t=hac_mean(part.factor_return)
            d=diag[diag.signal==signal]
            record=dict(model=model,signal=signal,annual_unit_exposure_mean=12*mean,mean_hac_t=t,
                        average_raw_gross=d.raw_gross.mean(),average_factor_exposure_at_gross2=d.gross2_factor_exposure.mean(),
                        average_largest_stock_weight=d.max_abs_weight.mean(),average_effective_positions=d.effective_positions.mean())
            regression_summary.append(record)
            raw_table.append({'Faktor':LABELS[signal],'Godišnji prosjek koeficijenta (%)':100*12*mean,
                'HAC t':t,'Prosječni raw gross':d.raw_gross.mean(),
                'Izloženost pri gross 2':d.gross2_factor_exposure.mean(),
                'Najveća težina, prosjek (%)':100*d.max_abs_weight.mean()})
        regression_html=pd.DataFrame(raw_table).to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}')
        for scenario in ['gross','cost10_borrow3']:
            selected_sides=sides[(sides.model==model)&(sides.scenario==scenario)].pivot(index='signal_date',columns=['leg','signal'],values='return')
            common_sides=sides[(sides.model=='sort_common')&(sides.scenario==scenario)].pivot(index='signal_date',columns=['leg','signal'],values='return')
            side_maps={}
            for leg in ['long','short']:
                side_maps[leg]=pd.DataFrame(index=selected_sides.index)
                for signal in PROFIT:
                    side_maps[leg][signal]=selected_sides[leg,signal]
                    side_maps[leg]['sort_'+signal]=common_sides[leg,signal]
                for signal in ['portfolio3','portfolio3_gross2']:
                    side_maps[leg][signal]=selected_sides[leg,'basket3']
                for signal in ['sort_common_basket3','sort_common_basket3_gross2']:
                    side_maps[leg][signal]=common_sides[leg,'basket3']
            part=monthly[(monthly.model==model)&(monthly.scenario==scenario)&(monthly.missing_policy=='flat_unknown')]
            matrix=part.pivot(index='signal_date',columns='signal',values='return')
            calendar=source[['signal_date','entry_date','exit_date']].set_index('signal_date')
            matrix=calendar.join(matrix).rename(columns={'basket3':'portfolio3','basket3_gross2':'portfolio3_gross2'})
            common=monthly[(monthly.model=='sort_common')&(monthly.scenario==scenario)&(monthly.missing_policy=='flat_unknown')].pivot(index='signal_date',columns='signal',values='return')
            matrix['sort_common_basket3']=common.basket3
            matrix['sort_common_basket3_gross2']=common.basket3_gross2
            for signal in PROFIT:matrix['sort_'+signal]=common[signal]
            # This literal basket is explicitly gross; the cost page does not
            # misrepresent the public RMW as an executable net sleeve.
            gross=monthly[(monthly.model==model)&(monthly.scenario=='gross')&(monthly.missing_policy=='flat_unknown')].pivot(index='signal_date',columns='signal',values='return')
            for signal in ['public_rmw','original_portfolio4']:
                matrix[signal]=source.set_index('signal_date')['portfolio4' if signal=='original_portfolio4' else signal]
            matrix['portfolio4']=(gross[PROFIT].sum(axis=1)+matrix.public_rmw)/4
            merged=matrix.reset_index().merge(rf,on=['signal_date','entry_date','exit_date'],validate='one_to_one')
            merged=merged.merge(features,on=['signal_date','entry_date','exit_date'],validate='one_to_one')
            merged['risk_free']=merged.risk_free
            assert len(merged)==235 and np.isfinite(merged[[s['id'] for s in series]]).all().all()
            records=[]
            for i,row in enumerate(merged.itertuples()):
                records.append(dict(i=i,date=row.signal_date.strftime('%Y-%m-%d'),entry=row.entry_date.strftime('%Y-%m-%d'),
                    exit=row.exit_date.strftime('%Y-%m-%d'),month=row.entry_date.strftime('%Y-%m'),common=bool(row.common_sample),rf=float(row.risk_free),
                    r={s['id']:float(getattr(row,s['id'])) for s in series},
                    legs={leg:{**{key:float(value) for key,value in side_maps[leg].loc[row.signal_date].items()},'risk_free':float(row.risk_free)} for leg in ['long','short']},
                    states={r['id']:None if pd.isna(getattr(row,r['id']+'_state')) else int(getattr(row,r['id']+'_state')) for r in REGIMES}))
            for regime in [None,*REGIMES]:
                for state in ([None] if regime is None else [0,1]):
                    active=np.ones(len(merged),dtype=bool) if regime is None else (merged.common_sample&merged[regime['id']+'_state'].eq(state))
                    for s in series:
                        values=merged[s['id']].where(active,merged.risk_free)
                        all_regimes.append(dict(model=model,scenario=scenario,signal=s['id'],leg='long_short',regime='all' if regime is None else regime['id'],state=state,
                            active_months=int(np.sum(active)),cash_months=int(np.sum(~np.asarray(active))),
                            **cash_stats(values,merged.risk_free,merged.entry_date,merged.exit_date)))
                    for leg in ['long','short']:
                        for signal in [*side_maps[leg].columns,'risk_free']:
                            raw=merged.risk_free if signal=='risk_free' else merged.signal_date.map(side_maps[leg][signal])
                            values=raw.where(active,merged.risk_free)
                            all_regimes.append(dict(model=model,scenario=scenario,signal=signal,leg=leg,regime='all' if regime is None else regime['id'],state=state,
                                active_months=int(np.sum(active)),cash_months=int(np.sum(~np.asarray(active))),
                                **cash_stats(values,merged.risk_free,merged.entry_date,merged.exit_date)))
            nav=' '.join(f'<a href="{filename(m,scenario)}" style="margin-right:16px">{html.escape(METHOD_NAMES[m.split("__")[0]]+" / "+("√cap" if m.endswith("sqrt_cap") else "1/spec.var"))}</a>' for m in models)
            costs=' '.join(f'<a href="{filename(model,s)}" style="margin-right:16px">{label}</a>' for s,label in [('gross','Bruto'),('cost10_borrow3','10 bps + 3% posudba')])
            assumption='Bruto, bez troškova.' if scenario=='gross' else 'Lokalni FMP i kvintilni portfelji: 10 bps po prometu i 3% godišnja posudba. RMW i oba četverofaktorska prikaza ostaju BRUTO. Režimske krivulje preslikavaju mjesečne prinose stalno aktivne troškovne strategije; promet režimske strategije nije ponovno izračunat.'
            note=f'''<section class="panel" style="padding:20px;margin-bottom:20px"><h2 style="font-size:18px">{html.escape(title)}</h2><p>{assumption}</p><p>{nav}</p><p>{costs}</p><p>Bez sektorske neutralizacije. Izvan odabranog režima primjenjuje se RF. Troškovi samog režimskog uključivanja nisu procijenjeni.</p></section>
            <section class="panel" style="padding:20px;margin-bottom:20px"><label for="legMode"><strong>Strana portfelja</strong></label><select id="legMode" aria-label="Strana portfelja" style="display:block;width:100%;max-width:340px;margin-top:10px"></select><p id="legNote" style="line-height:1.6"></p><p>Za usporedbu long i short linija: odaberi stranu, klikni „Dodaj usporedbu”, zatim promijeni stranu. RF se primjenjuje u cash režimu u svim prikazima.</p></section>'''
            raw_section=f'''<section class="panel" style="padding:24px"><h2>Regresijski koeficijenti i skaliranje FMP-a</h2><p>Opis cijelog uzorka, ne mijenja se s gornjim filtrima. Koeficijenti su prinosi po jediničnoj transformiranoj izloženosti; nisu CAGR portfelja gross 200%.</p><div class="table-scroll">{regression_html}</div><p>HAC s tri laga; vremenska podjela prije/od 2017. dostupna je u summary.csv i predstavlja stabilnost, ne netaknuti OOS.</p></section>'''
            methodology='''<section class="panel methodology" id="methodology"><details><summary><h2>Metodologija, provjere i datoteke</h2></summary><div style="padding:24px">
            <p>Sva tri signala i šest kontrola procjenjuju se istodobno: tržišna konstanta, FCF/imovina, CFO/imovina, GP/imovina, log veličine, book-to-market, momentum 12–1, SPY beta 63 dana, log rezidualne volatilnosti i log Amihudove nelikvidnosti. Sektori su izostavljeni jer povijesne klasifikacije nisu potvrđene.</p>
            <p>Potpuni formacijski uzorak određuju samo tada dostupne karakteristike. Buduća dostupnost prinosa ne uklanja dionice. Z-score: winsorizacija 1/99%, kapitalizacijski ponderirana sredina i jednako ponderirana populacijska SD. Alternativa: normalni kvantili prosječnih rangova, zatim ista standardizacija. Nema imputacije karakteristika.</p>
            <p>Regresija: r − RF = X f + ε. WLS operator A = (X′WX)⁻¹X′W računa se QR dekompozicijom. Redak A za faktor daje njegov raw FMP. X′h = e: jedinica ciljnom faktoru, nula ostalim stupcima, uključujući tržišnu konstantu. Raw PnL jednak je koeficijentu uz isti tretman nepoznatih prinosa. Nula izloženosti transformiranom beta stupcu nije jamstvo nulte ostvarene tržišne bete.</p>
            <p>Težine W: korijen kapitalizacije ili inverzna procjena specifične varijance. Specifična varijanca je EWMA kvadrata reziduala prethodnih mjesečnih regresija na kontrole (poluživot 12 mjeseci, najmanje 12 poznatih reziduala). Rezidual se dodaje tek kad je exit_date ≤ signal_date. Za kraću povijest koristi se 63d SPY rezidualna varijanca / 12. Pod varijance je 10% medijana presjeka; inverzne težine ograničene su na 1/99 percentil. To je istraživačka aproksimacija, nije komercijalni Barra model.</p>
            <p>Pojedinačni faktori skalirani su na gross 200%, što mijenja jediničnu izloženost. Tri FMP-a u košarici imaju po 1/3 kapitala; pozicije se netiraju pa gross košarice može biti niži od 200%. Prikazane su i košarice dodatno reskalirane na gross 200% na svakom datumskom formiranju, samo prema tada poznatim težinama. Kvintilna kontrola koristi isti potpuni univerzum i isto pravilo skaliranja. Četverofaktorska usporedba zadržava bruto jednake prinose triju pojedinačnih FMP-ova i javnog RMW-a.</p>
            <p>Formiranje: kraj mjeseca, ulaz: sljedeće zatvaranje, izlaz: zatvaranje nakon sljedećeg kraja mjeseca. Nepoznat prinos u osnovnoj verziji ima pretpostavljen nulti prinos i zatvaranje; zasebna osjetljivost pretpostavlja −100% za nepoznate long ishode i +100% za short. Odbijena ulazna cijena i praktično skaliranje mogu narušiti točnu FMP neutralnost; teorijske težine i identitet ostaju zasebno spremljeni.</p>
            <p>Aktivni portfelji koriste nultu kamatu na cash/kolateral i nulto financiranje, dosljedno ranijim portfeljima. U režimskom cashu koristi se povijesni RF iz istih dnevnih prozora kao RMW. Sharpe oduzima RF. Troškovi režimskog uključivanja nisu dodani; drawdown je mjesečni. Izvještaji su revidirani vendorovi podaci s procijenjenim datumima objave; short dostupnost nije potvrđena.</p>
            <p>Odabir Long ili Short predstavlja zasebno mjesečno rebalansiran portfelj s +100% odnosno −100% NAV-a na formaciji, uz iste relativne težine odgovarajuće strane netiranog portfelja. Short prikazuje stvarni predznak dobiti od shortanja: pad cijena donosi pozitivan prinos. Troškovi trgovanja i završnog zatvaranja računaju se za svaku stranu zasebno, a posudba samo za short. Odbijeni ulazi se ne nadomještaju. Odabir jedne strane gubi FMP neutralnost. Ne uključujemo kamatu na short kolateral ili dodatni RF u aktivnim mjesecima. Zasebno ukamaćene strane i troškovi nisu aditivna dekompozicija CAGR-a long–short portfelja.</p>
            <div id="regimeDefinitions" class="definition-grid"></div><p>__LINKS__</p></div></details></section>'''
            links=['summary.csv','side_summary.csv','side_monthly_returns.csv','side_manifest.json','factor_returns.csv','regression_summary.csv','diagnostics.csv','coverage.csv','monthly_returns.csv','regime_cash_summary.csv','weights.parquet','design.parquet','manifest.json','validation.json']
            methodology=methodology.replace('__LINKS__',' · '.join(f'<a href="{f}">{f}</a>' for f in links))
            template=base_template.replace('<section class="selection-bar">',note+'<section class="selection-bar">')
            if scenario!='gross': template=template.replace('Bruto rezultati','Pretpostavljeni troškovi')
            template=template.replace('<a href="report_static.html">statični izvještaj</a>','<a href="summary.csv">CSV sažetak</a>')
            template=template.replace('__FMP_DIAGNOSTICS__',raw_section)
            payload=dict(series=series,legs=legs,chartBasis='Mjesečni rebalans. Primjenjuju se pretpostavke prinosa i troškova navedene uz odabrani model.',regimes=REGIMES,months=records,commonMonths=int(merged.common_sample.sum()),ic=dict(signals=[],values=[[] for _ in records]))
            encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<',r'\u003c')
            document=template.replace('__REPORT_CSS__',(ASSETS/'portfolio.css').read_text(encoding='utf-8')).replace('__REPORT_JS__',js).replace('__REPORT_DATA__',encoded).replace('__REPORT_METHODOLOGY__',methodology)
            assert '__REPORT_' not in document
            (OUT/filename(model,scenario)).write_text(document,encoding='utf-8')
            if scenario=='gross':matrix.to_csv(OUT/(model+'__comparison.csv'))
    pd.DataFrame(regression_summary).to_csv(OUT/'regression_summary.csv',index=False)
    pd.DataFrame(all_regimes).to_csv(OUT/'regime_cash_summary.csv',index=False)
    (OUT/'report_manifest.json').write_text(json.dumps(dict(code_sha256=sha(__file__),models=models,standalone_pages=8,
        asset_hashes={f:sha(ASSETS/f) for f in ['portfolio.html','portfolio.css','portfolio.js']},
        source_hashes={f:sha(OUT/f) for f in ['monthly_returns.csv','side_monthly_returns.csv','side_manifest.json','factor_returns.csv','diagnostics.csv']}),indent=2))
    print('Built eight standalone FMP reports:',OUT/'report.html',flush=True)


if __name__=='__main__':build()
