"""Independent, isolated audit: never rewrites Claude's source or results.

Reproduce the saved baseline, compare identical calendar windows, enforce the
requested ACTUAL holding limit, and measure execution/dividend sensitivities.
These are diagnostics, not a newly selected strategy or a clean OOS test.
"""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tintor_claude_audit/20261117_v1'
ZSE=Path('C:/Users/Mislav/Documents/GitHub/zse-research')
ORIGINAL=ROOT/'scripts/tintor_claude'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def stats(nav,initial,begin,end,base_date=None):
    s=nav.sort_index().loc[begin:end]
    history=nav.sort_index().loc[nav.index< s.index[0]]
    base=history.iloc[-1] if len(history) else initial
    base_date=history.index[-1] if len(history) else pd.Timestamp(begin)
    w=s/base;years=(s.index[-1]-base_date).days/365.25
    return dict(cagr=float(w.iloc[-1]**(1/years)-1),max_dd=float((w/w.cummax().clip(lower=1)-1).min()),
                total=float(w.iloc[-1]-1),first=str(s.index[0].date()),last=str(s.index[-1].date()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[*ORIGINAL.glob('*.py'),ROOT/'config/tintor_claude.json',
           *[ROOT/'data/tintor_claude'/n for n in ['prices.parquet','signals.parquet','dividends.parquet','filings.parquet']],
           ROOT/'results/tintor_claude/final/summary.csv',ROOT/'results/tintor_claude/final/nav.csv',Path(__file__)]
    before={str(p):sha(p) for p in files}
    (OUT/'inputs.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    sys.path.insert(0,str(ORIGINAL));base=load_module('claude_tintor_original_audit',ORIGINAL/'backtest.py')
    # Readable snapshot of exactly the original run function with three explicit
    # diagnostic controls. No edits to the original project implementation.
    code=inspect.getsource(base.run)
    old='fill_days: int | None = None, dividends: bool = True, label: str = "") -> dict:'
    assert old in code
    code=code.replace(old,'fill_days: int | None = None, dividends: bool = True, label: str = "",\n        actual_limit: int | None = None, retain_missing_quotes: bool = False) -> dict:',1)
    old='''            if px != px:
                val = pos[i] * last_px.get(i, 0.0) * (1 - ex["delisting_haircut"])'''
    new='''            if px != px:
                if retain_missing_quotes:
                    continue  # Keep owned; last mark provisional, no invented sale/cash.
                val = pos[i] * last_px.get(i, 0.0) * (1 - ex["delisting_haircut"])'''
    assert old in code;code=code.replace(old,new,1)
    old='''                    if side == "B":
                        fp_ = px * (1 + slippage)'''
    new='''                    if side == "B":
                        if actual_limit is not None and i not in pos and len(pos) >= actual_limit:
                            q = 0.0  # An unsold old holding still occupies a slot.
                        fp_ = px * (1 + slippage)'''
    assert old in code;code=code.replace(old,new,1)
    (OUT/'diagnostic_run_function.py').write_text(code,encoding='utf-8')
    ns=base.__dict__.copy();exec(compile(code,str(OUT/'diagnostic_run_function.py'),'exec'),ns);run=ns['run']
    m=base.Market();saved=pd.read_csv(ROOT/'results/tintor_claude/final/nav.csv',parse_dates=['date']).set_index('date')
    targets={name:base.build_targets(m.sig,name,8)[0] for name in ['kvaliteta','kvaliteta+vrednovanje','ew_svi']}
    cases=[('original_quality','kvaliteta',{}),('original_value','kvaliteta+vrednovanje',{}),('original_ew','ew_svi',{}),
           ('limit8_quality','kvaliteta',dict(actual_limit=8)),('limit8_value','kvaliteta+vrednovanje',dict(actual_limit=8)),
           ('limit8_5pct_5days_quality','kvaliteta',dict(actual_limit=8,cap=.05,fill_days=5)),
           ('no_dividends_quality','kvaliteta',dict(dividends=False)),
           ('retain_missing_ew','ew_svi',dict(retain_missing_quotes=True))]
    rows=[];position_rows=[];results={}
    for i,(case,strategy,kw) in enumerate(cases,1):
        res=run(m,targets[strategy],'2019-01-01','2026-11-16',label=case,**kw);results[case]=res
        if case.startswith('original_'):
            reference=saved[f'{strategy}|zavrsno'].dropna()
            assert res['nav'].index.equals(reference.index)
            assert np.allclose(res['nav'].nav,reference,rtol=1e-11,atol=1e-6),case
        if kw.get('actual_limit'):assert res['nav'].n_pos.le(kw['actual_limit']).all()
        d=res['nav'];d.to_csv(OUT/f'{case}_daily.csv')
        for window,start,end in [('claude_final','2019-01-01','2026-11-16'),('matched','2021-01-01','2026-10-30')]:
            sel=d.loc[start:end]
            rows.append(dict(source='Claude',case=case,strategy=strategy,window=window,**stats(d.nav,100000,start,end),
                             mean_positions=float(sel.n_pos.mean()),max_positions=int(sel.n_pos.max()),
                             fraction_over8=float(sel.n_pos.gt(8).mean()),mean_exposure=float((1-sel.cash/sel.nav).mean())))
        print(f'{i}/{len(cases)} {case}: final NAV {d.nav.iloc[-1]:.2f}; max positions {d.n_pos.max()}',flush=True)
    zrun=ZSE/'data/derived/tintor_fundamental/20261116_v3'
    for case,strategy in [('ours_quality','quality'),('ours_value','quality_valuation_proxy'),('ours_ew','ew_complete')]:
        d=pd.read_parquet(zrun/f'{strategy}__net50__c100000__lastmark_daily.parquet').set_index('date')
        sel=d.loc['2021-01-01':'2026-10-30']
        rows.append(dict(source='Codex',case=case,strategy=strategy,window='matched',**stats(d.nav_eur,100000,'2021-01-01','2026-10-30'),
                         mean_positions=float(sel.n_positions.mean()),max_positions=int(sel.n_positions.max()),
                         fraction_over8=float(sel.n_positions.gt(8).mean()),mean_exposure=float((1-sel.cash_eur/sel.nav_eur).mean())))
    for source,path in [('Claude',ROOT/'data/tintor_claude/raw/index_CROBEXtr.csv')]:
        idx=pd.read_csv(path,parse_dates=['date']).set_index('date').last_value.sort_index()
        rows.append(dict(source=source,case='CROBEXtr',strategy='CROBEXtr',window='matched',**stats(idx,1,'2021-01-01','2026-10-30')))
    zi=pd.read_parquet(ZSE/'data/derived/zse_qautoml/20261113_refresh_v1/indices.parquet')
    idx=zi[zi['index'].str.lower().eq('crobextr')].set_index('date').level.sort_index()
    rows.append(dict(source='Codex',case='CROBEXtr',strategy='CROBEXtr',window='matched',**stats(idx,1,'2021-01-01','2026-10-30')))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'comparison.csv',index=False)
    # Missing-quote cashouts are a flaw in the engine, but quantify where the
    # saved experiment actually exercised this branch before attributing gains.
    cashouts=[]
    for case in ['original_quality','original_value','original_ew']:
        e=results[case]['events'];e=e.loc[e.type.eq('delistiranje')]
        for r in e.itertuples():
            later=m.close.loc[m.close.index>r.date,r.isin].dropna()
            cashouts.append(dict(case=case,date=r.date,isin=r.isin,amount=r.amount,has_later_quote=bool(len(later)),
                                 next_quote_date=later.index[0] if len(later) else pd.NaT))
    pd.DataFrame(cashouts).to_csv(OUT/'automatic_cashouts.csv',index=False)
    dividends=pd.read_parquet(ROOT/'data/tintor_claude/dividends.parquet')
    dividends['period_end']=[pd.Timestamp(int(y),3*int(q),1)+pd.offsets.MonthEnd(0) for y,q in zip(dividends.fy,dividends.q)]
    dividends['before_quarter_end']=dividends.pay_date<dividends.period_end
    dividends[['ticker','fy','q','pay_date','period_end','amount','dps_adj','before_quarter_end']].to_csv(OUT/'dividend_timing.csv',index=False)
    # Security P&L for original quality in matched window; this variant has no
    # automatic cashouts, so independent reconstruction needs only actual fills.
    res=results['original_quality'];assert not res['events'].type.eq('delistiranje').any()
    fills=res['fills'].copy();fills['units']=np.where(fills.side.eq('B'),fills.shares_adj,-fills.shares_adj)
    init=fills.loc[fills.exec_date<'2021-01-01'].groupby('isin').units.sum()
    end=fills.loc[fills.exec_date<='2026-10-30'].groupby('isin').units.sum()
    f=fills.loc[fills.exec_date.between('2021-01-01','2026-10-30')]
    flow=pd.Series(np.where(f.side.eq('B'),-f.value-f.commission,f.value-f.commission),index=f.index).groupby(f['isin']).sum()
    div=res['events'].query("type=='dividenda'");div=div.loc[div.date.between('2021-01-01','2026-10-30')].groupby('isin').amount.sum()
    held=set(init.index)|set(end.index)
    for sid in held:
        p0=m.close.loc[:'2020-12-31',sid].dropna();p1=m.close.loc[:'2026-10-30',sid].dropna()
        begin=init.get(sid,0)*(p0.iloc[-1] if len(p0) else 0);ending=end.get(sid,0)*(p1.iloc[-1] if len(p1) else 0)
        pnl=ending-begin+flow.get(sid,0)+div.get(sid,0)
        position_rows.append(dict(isin=sid,symbol=m.sym.get(sid),pnl_eur=pnl,initial_value=begin,final_value=ending,dividends_eur=div.get(sid,0)))
    attribution=pd.DataFrame(position_rows).sort_values('pnl_eur',ascending=False)
    actual_delta=res['nav'].loc[:'2026-10-30','nav'].iloc[-1]-res['nav'].loc[:'2020-12-31','nav'].iloc[-1]
    assert np.isclose(attribution.pnl_eur.sum(),actual_delta,atol=.001)
    attribution.to_csv(OUT/'quality_pnl_by_security.csv',index=False)
    assert all(sha(p)==h for p,h in before.items()),'Original inputs changed during audit'
    validation=dict(passed=True,baselines_reproduced=3,actual_limit_verified=True,pnl_reconciles=True,
                    original_inputs_unchanged=True,dividends_before_quarter_end=int(dividends.before_quarter_end.sum()),
                    dividend_records=len(dividends),automatic_cashout_events=len(cashouts),cases=len(cases))
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(summary.loc[summary.window.eq('matched'),['source','case','cagr','max_dd','mean_positions','max_positions','fraction_over8','mean_exposure']].to_string(index=False))
    print(attribution.head(8).to_string(index=False));print(validation)

if __name__=='__main__':main()
