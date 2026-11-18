(() => {
  'use strict';
  const DATA = JSON.parse(document.getElementById('report-data').textContent);
  const $ = id => document.getElementById(id);
  const series = DATA.series, regimes = DATA.regimes, months = DATA.months;
  const byId = Object.fromEntries(series.map(s => [s.id,s]));
  const range = [months[0].month,months.at(-1).month];
  const state = {from:range[0],to:range[1],focal:'portfolio4',filters:{},leg:'long_short',
    chart:'equity',visible:new Set(['portfolio4','public_rmw']),saved:[],compare:'spy_vol21',regimeMetric:'cagr'};
  const comparisonColors=['#7d56c4','#209791','#cf7840','#a34f78','#698d36','#be5558'];
  const fmt = (n,d=2) => Number.isFinite(n) ? n.toLocaleString('hr-HR',{minimumFractionDigits:d,maximumFractionDigits:d}) : '—';
  const pct = (n,d=2) => Number.isFinite(n) ? `${fmt(n*100,d)}%` : '—';
  const date = s => new Date(s+'T12:00:00Z');
  const dateLabel = s => date(s).toLocaleDateString('hr-HR',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});
  const monthLabel = m => date(m+'-01').toLocaleDateString('hr-HR',{month:'short',year:'numeric',timeZone:'UTC'});
  const mean = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : NaN;
  const sd = a => a.length>1 ? Math.sqrt(a.reduce((x,y)=>x+(y-mean(a))**2,0)/(a.length-1)) : NaN;
  const hasRegime = () => regimes.some(r => state.filters[r.id] !== undefined);
  const focal = () => byId[state.focal];
  const colorClass = value => Number.isFinite(value) ? value>0?'positive':value<0?'negative':'' : '';
  const escape = text => String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const components=series.filter(s=>s.component);
  const legModes=DATA.legs || [{id:'long_short',label:'Long–short'}];
  const legId=selection=>selection.leg || 'long_short';
  const supportsLeg=(s,selection=state)=>!s.legs || s.legs.includes(legId(selection));
  const availableSeries=(selection=state)=>series.filter(s=>supportsLeg(s,selection));
  const returnsFor=(m,selection=state)=>legId(selection)==='long_short'?m.r:m.legs[legId(selection)];
  const legLabel=selection=>legModes.find(l=>l.id===legId(selection))?.label || 'Long–short';
  const seriesLabel=(s,selection=state,short=false)=>(DATA.legs && s.id!=='risk_free'?legLabel(selection)+' · ':'')+(short?(legId(selection)!=='long_short'&&s.sideShort?s.sideShort:s.short):legId(selection)!=='long_short'&&s.sideLabel?s.sideLabel:s.label);
  let currentRows=[],currentStrategyRows=[],currentMetrics=null,hoverIndex=null;

  function filterRows({ignore=null,forceCommon=false,baseline=false,selection=state}={}) {
    const common = forceCommon || Object.keys(selection.filters).length>0;
    return months.filter(m => m.month>=selection.from && m.month<=selection.to && (!common || m.common)
      && (baseline || regimes.every(r => r.id===ignore || selection.filters[r.id]===undefined || m.states[r.id]===selection.filters[r.id])))
      .map(m=>({...m,r:returnsFor(m,selection)}));
  }
  function strategyRows(selection=state) {
    const gated=Object.keys(selection.filters).length>0;
    return months.filter(m=>m.month>=selection.from && m.month<=selection.to).map(m=>{
      const active=!gated || (m.common && regimes.every(r=>selection.filters[r.id]===undefined || m.states[r.id]===selection.filters[r.id]));
      return {...m,active,cashReason:active?'':!m.common?'regime_unavailable':'outside_regime',
        r:active?returnsFor(m,selection):Object.fromEntries(availableSeries(selection).map(s=>[s.id,m.rf]))};
    });
  }
  function curve(rows,id,drawdown=false) {
    let value=1,peak=1;
    const result=[drawdown?0:1];
    for (const row of rows) {
      value*=1+row.r[id];peak=Math.max(peak,value);
      result.push(drawdown?value/peak-1:value);
    }
    return result;
  }
  function calculate(rows,id) {
    if (!rows.length) return {n:0,cagr:NaN,annualMean:NaN,vol:NaN,sharpe:NaN,dd:NaN,hit:NaN,total:NaN};
    const r=rows.map(m=>m.r[id]),avg=mean(r),vol=sd(r)*Math.sqrt(12),equity=curve(rows,id);
    const excess=rows.map(m=>m.r[id]-m.rf),excessVol=sd(excess)*Math.sqrt(12);
    const years=(date(rows.at(-1).exit)-date(rows[0].entry))/(365.25*86400000);
    return {n:r.length,cagr:years>0?equity.at(-1)**(1/years)-1:NaN,annualMean:avg*12,
      vol,sharpe:excessVol>1e-12?mean(excess)*12/excessVol:NaN,dd:Math.min(...curve(rows,id,true)),
      hit:r.filter(x=>x>0).length/r.length,total:equity.at(-1)-1};
  }
  function correlation(x,y) {
    if (x.length<3) return NaN;
    const mx=mean(x),my=mean(y),sx=x.reduce((s,v)=>s+(v-mx)**2,0),sy=y.reduce((s,v)=>s+(v-my)**2,0);
    return sx>0 && sy>0 ? x.reduce((s,v,i)=>s+(v-mx)*(y[i]-my),0)/Math.sqrt(sx*sy) : NaN;
  }
  function setValue(id,value,text) {
    $(id).textContent=text;
    $(id).dataset.value=Number.isFinite(value)?String(value):'';
  }
  function saveHash() {
    const p=new URLSearchParams();
    if (state.from!==range[0]) p.set('from',state.from);
    if (state.to!==range[1]) p.set('to',state.to);
    if (state.focal!=='portfolio4') p.set('series',state.focal);
    if (state.leg!=='long_short') p.set('leg',state.leg);
    p.set('lines',[...state.visible].join(','));
    if(state.saved.length) p.set('comparisons',JSON.stringify(state.saved));
    for (const r of regimes) if (state.filters[r.id]!==undefined) p.set(r.id,state.filters[r.id]);
    try {history.replaceState(null,'',location.pathname+location.search+(p.size?'#'+p.toString():''));} catch (_) { /* file preview may restrict history */ }
  }
  function readHash() {
    const p=new URLSearchParams(location.hash.slice(1));
    for (const key of ['from','to']) if (/^\d{4}-\d{2}$/.test(p.get(key)||'') && p.get(key)>=range[0] && p.get(key)<=range[1]) state[key]=p.get(key);
    if (byId[p.get('series')]) state.focal=p.get('series');
    if(legModes.some(l=>l.id===p.get('leg')))state.leg=p.get('leg');
    if(!supportsLeg(focal()))state.focal=availableSeries()[0].id;
    for (const r of regimes) if (['0','1'].includes(p.get(r.id))) state.filters[r.id]=Number(p.get(r.id));
    if(p.has('lines')) state.visible=new Set(p.get('lines').split(',').filter(id=>byId[id]));
    else state.visible.add(state.focal);
    try {
      const comparisons=JSON.parse(p.get('comparisons')||'[]');
      if(Array.isArray(comparisons)) state.saved=comparisons.filter(c=>c && byId[c.series] &&
        /^\d{4}-\d{2}$/.test(c.from) && /^\d{4}-\d{2}$/.test(c.to) && c.from>=range[0] && c.to<=range[1] && c.from<=c.to &&
        legModes.some(l=>l.id===legId(c)) && supportsLeg(byId[c.series],c) &&
        c.filters && typeof c.filters==='object' && !Array.isArray(c.filters) && Object.entries(c.filters).every(([id,v])=>regimes.some(r=>r.id===id) && [0,1].includes(v)))
        .map(c=>({series:c.series,from:c.from,to:c.to,leg:legId(c),filters:{...c.filters}}));
    } catch (_) { state.saved=[]; }
  }
  function initialize() {
    readHash();
    if(window.matchMedia('(max-width:900px)').matches) document.querySelector('.filter-panel').open=false;
    $('from').min=$('to').min=range[0];$('from').max=$('to').max=range[1];
    $('regimeFilters').innerHTML=regimes.map(r=>`<label for="filter-${r.id}">${r.label}<select id="filter-${r.id}" aria-label="${r.label}"><option value="all">Sva razdoblja</option><option value="1">${r.high}</option><option value="0">${r.low}</option></select></label>`).join('');
    $('focal').innerHTML=series.filter(s=>s.id!=='risk_free').map(s=>`<option value="${s.id}">${s.label}</option>`).join('');
    if($('legMode')){
      $('legMode').innerHTML=legModes.map(l=>`<option value="${l.id}">${l.label}</option>`).join('');
      $('legMode').addEventListener('change',e=>{
        state.leg=e.target.value;
        if(!supportsLeg(focal()))state.focal=availableSeries()[0].id;
        state.visible.add(state.focal);render();
      });
    }
    $('compareRegime').innerHTML=regimes.map(r=>`<option value="${r.id}">${r.label}</option>`).join('');
    $('allocation').innerHTML=components.map(s=>`<div class="allocation-row"><span class="allocation-name"><i class="series-dot" style="background:${s.color}"></i>${s.short}</span><strong>25%</strong></div>`).join('');
    $('commonCount').textContent=DATA.commonMonths;
    $('regimeDefinitions').innerHTML=regimes.map(r=>`<div class="definition"><h4>${r.label}</h4><p>${r.description}</p></div>`).join('');
    for (const r of regimes) $('filter-'+r.id).addEventListener('change',e=>{
      if(e.target.value==='all') delete state.filters[r.id];else state.filters[r.id]=Number(e.target.value);
      state.compare=r.id;render();
    });
    for(const key of ['from','to']) $(key).addEventListener('change',e=>{state[key]=e.target.value||range[key==='from'?0:1];render();});
    $('focal').addEventListener('change',e=>{state.focal=e.target.value;state.visible.add(state.focal);render();});
    $('compareRegime').addEventListener('change',e=>{state.compare=e.target.value;renderRegimeBars();});
    $('reset').addEventListener('click',()=>{state.from=range[0];state.to=range[1];state.filters={};render();});
    document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{
      state.filters={};
      if(button.dataset.preset!=='all') {const id=button.dataset.preset;state.filters[id]=id==='spy_vol21'?1:0;state.compare=id;}
      render();
    }));
    document.querySelectorAll('[data-chart]').forEach(button=>button.addEventListener('click',()=>{
      state.chart=button.dataset.chart;
      document.querySelectorAll('[data-chart]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
      renderChart();
    }));
    document.querySelectorAll('[data-regime-metric]').forEach(button=>button.addEventListener('click',()=>{
      state.regimeMetric=button.dataset.regimeMetric;
      document.querySelectorAll('[data-regime-metric]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
      renderRegimeBars();
    }));
    $('export').addEventListener('click',exportCSV);
    $('showAllSeries').addEventListener('click',()=>{state.visible=new Set(availableSeries().map(s=>s.id));renderLegend();renderChart();saveHash();});
    $('showFocalSeries').addEventListener('click',()=>{state.visible=new Set([state.focal]);renderLegend();renderChart();saveHash();});
    $('addComparison').addEventListener('click',()=>{
      const filters=Object.fromEntries(regimes.filter(r=>state.filters[r.id]!==undefined).map(r=>[r.id,state.filters[r.id]]));
      const comparison={series:state.focal,from:state.from,to:state.to,leg:state.leg,filters};
      const duplicate=state.saved.some(c=>c.series===comparison.series && c.from===comparison.from && c.to===comparison.to &&
        legId(c)===legId(comparison) &&
        regimes.every(r=>c.filters[r.id]===comparison.filters[r.id]));
      $('comparisonStatus').hidden=false;
      if(duplicate){$('comparisonStatus').textContent='Taj je odabir već dodan na graf.';return;}
      state.saved.push(comparison);$('comparisonStatus').textContent='Usporedba je sačuvana. Sada možeš promijeniti filtre ili glavnu seriju.';
      renderLegend();renderChart();saveHash();
    });
    document.querySelector('.brand').addEventListener('click',e=>{
      e.preventDefault();window.scrollTo({top:0,behavior:'smooth'});
    });
    document.querySelector('.method-link').addEventListener('click',e=>{
      e.preventDefault();$('methodology').querySelector('details').open=true;
      $('methodology').scrollIntoView({behavior:'smooth'});
    });
    let priorWidth=0;
    new ResizeObserver(entries=>{const w=Math.round(entries[0].contentRect.width);if(w!==priorWidth){priorWidth=w;requestAnimationFrame(renderChart);}}).observe($('chartArea'));
    render();
  }
  function renderControls() {
    $('focal').innerHTML=availableSeries().filter(s=>s.id!=='risk_free').map(s=>`<option value="${s.id}">${legId(state)!=='long_short'&&s.sideLabel?s.sideLabel:s.label}</option>`).join('');
    if($('legMode')){
      $('legMode').value=state.leg;
      $('legNote').textContent=state.leg==='long_short'?
        'Cijeli long–short portfelj prema odabranoj konstrukciji.':
        'Zasebna '+(state.leg==='long'?'long':'short')+' strana, normalizirana na 100% NAV-a prije odbijanja naloga. '+
        (state.leg==='short'?'Pozitivan prinos znači dobit od shortanja. ':'')+
        'Nakon odabira jedne strane FMP neutralnost više ne vrijedi. RMW i portfelji koji ga uključuju nemaju dostupnu podjelu strana. Dvije skale iste netirane košarice daju istu stranu pa je prikazana jednom.';
    }
    $('from').value=state.from;$('to').value=state.to;$('focal').value=state.focal;$('compareRegime').value=state.compare;
    for(const r of regimes){const el=$('filter-'+r.id);el.value=state.filters[r.id]===undefined?'all':String(state.filters[r.id]);el.classList.toggle('filtered',state.filters[r.id]!==undefined);}
    document.querySelectorAll('[data-preset]').forEach(b=>{
      const key=b.dataset.preset,keys=Object.keys(state.filters);
      b.classList.toggle('active',key==='all'?keys.length===0:keys.length===1 && keys[0]===key && state.filters[key]===(key==='spy_vol21'?1:0));
      b.setAttribute('aria-pressed',b.classList.contains('active'));
    });
    $('activeFilters').innerHTML=regimes.filter(r=>state.filters[r.id]!==undefined).map(r=>`<button data-remove="${r.id}" aria-label="Ukloni filtar: ${state.filters[r.id]?r.high:r.low}">${state.filters[r.id]?r.high:r.low}<span aria-hidden="true">×</span></button>`).join('');
    $('activeFilters').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{delete state.filters[b.dataset.remove];render();}));
  }
  function renderKPIs() {
    const rows=currentStrategyRows,conditional=hasRegime(),metrics=calculate(rows,state.focal);
    const base=calculate(strategyRows({...state,filters:{}}),state.focal),cashMonths=rows.length-currentRows.length;
    currentMetrics=metrics;
    const ret=metrics.cagr;
    $('returnLabel').textContent=conditional?'Faktor / RF · CAGR':'Godišnji prinos · CAGR';
    setValue('returnValue',ret,pct(ret));setValue('sharpeValue',metrics.sharpe,fmt(metrics.sharpe));
    setValue('volValue',metrics.vol,pct(metrics.vol));setValue('monthsValue',currentRows.length,String(currentRows.length));
    $('returnNote').textContent=conditional?'Faktor u režimu, RF u cash mjesecima':'Godišnja složena stopa prinosa';
    $('returnNote').className='kpi-note';
    if(conditional && rows.length){const delta=metrics.sharpe-base.sharpe;$('sharpeNote').textContent=`${delta>0?'+':''}${fmt(delta)} prema svim režimima`;$('sharpeNote').className='kpi-note '+colorClass(delta);}
    else{$('sharpeNote').textContent='Mjesečni višak prinosa prema RF';$('sharpeNote').className='kpi-note';}
    $('volNote').textContent=conditional?'Cijela strategija, uključujući cash':'Mjesečna SD × √12';
    $('monthsNote').textContent=conditional?`${cashMonths} cash · ${rows.length} ukupno`:`od ${rows.length} u datumskom rasponu`;
    $('sampleBadge').textContent=conditional?'Faktor + RF cash':'Cijeli odabir';
    $('selectionDescription').textContent=`${monthLabel(state.from)} – ${monthLabel(state.to)}`;
    let alert='';
    if(state.from>state.to) alert='Početni mjesec je nakon završnog. Promijeni datumski raspon.';
    else if(!rows.length) alert='Nema dostupnih mjeseci u datumskom rasponu.';
    else if(conditional && !currentRows.length) alert='Nema mjeseci u odabranom režimu. Cijelo razdoblje strategija drži RF cash.';
    else if(currentRows.length<24) alert=`Samo ${currentRows.length} faktorskih mjeseci. Procjene za aktivni režim mogu biti nestabilne; ostatak razdoblja drži RF cash.`;
    $('sampleAlert').hidden=!alert;$('sampleAlert').textContent=alert;
    $('export').disabled=!rows.length;
    $('addComparison').disabled=!rows.length;
  }
  function renderLegend() {
    $('seriesLegend').innerHTML=availableSeries().map(s=>`<label class="series-choice"><input type="checkbox" data-series="${s.id}" aria-label="Prikaži seriju: ${s.label}" ${state.visible.has(s.id)?'checked':''}><i class="series-dot" style="background:${s.color}"></i>${s.short}${hasRegime() && s.id!=='risk_free'?' + RF':''}</label>`).join('');
    $('seriesLegend').querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{const id=input.dataset.series;input.checked?state.visible.add(id):state.visible.delete(id);renderChart();saveHash();}));
    $('savedSeries').innerHTML=state.saved.map((c,i)=>`<div class="saved-series-chip"><i class="series-dot" style="background:${comparisonColors[i%comparisonColors.length]}"></i><span>${escape(comparisonLabel(c))}</span><button data-remove-comparison="${i}" aria-label="Ukloni usporedbu: ${escape(comparisonLabel(c))}">×</button></div>`).join('');
    $('savedSeries').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{state.saved.splice(Number(button.dataset.removeComparison),1);renderLegend();renderChart();saveHash();}));
  }
  function comparisonLabel(c) {
    const parts=regimes.filter(r=>c.filters[r.id]!==undefined).map(r=>c.filters[r.id]?r.high:r.low);
    return seriesLabel(byId[c.series],c,true)+' · '+(parts.length?parts.join(' + ')+' / RF':'Svi mjeseci')+' · '+c.from+' – '+c.to;
  }
  function chartSegments(rows,id,dd) {
    const values=curve(rows,id,dd),segments=[];
    rows.forEach((row,i)=>{
      if(i===0 || rows[i-1].exit!==row.entry) segments.push([{time:date(row.entry).getTime(),value:values[i],row:null,kind:'entry'}]);
      segments.at(-1).push({time:date(row.exit).getTime(),value:values[i+1],row,kind:'exit'});
    });
    return segments;
  }
  function renderChart() {
    if(!$('chart')) return;
    const dd=state.chart==='drawdown';
    const lines=availableSeries().filter(s=>state.visible.has(s.id)).map(s=>({...s,key:s.id,label:seriesLabel(s),short:seriesLabel(s,state,true),rows:currentStrategyRows,saved:false}));
    state.saved.forEach((c,i)=>lines.push({...byId[c.series],key:'comparison-'+i,label:comparisonLabel(c),
      color:comparisonColors[i%comparisonColors.length],rows:strategyRows(c),saved:true}));
    const plotted=lines.filter(line=>line.rows.length).map(line=>({...line,segments:chartSegments(line.rows,line.id,dd)}));
    $('chartTooltip').hidden=true;hoverIndex=null;
    $('chartTitle').textContent=dd?'Mjesečni drawdown kroz vrijeme':'Rast vrijednosti kroz vrijeme';
    const filtered=hasRegime() || state.saved.some(c=>Object.keys(c.filters).length);
    $('chartNote').textContent='Os x prikazuje stvarno kalendarsko vrijeme. Svaka krivulja počinje od 1 na svom prvom ulazu. '+
      (filtered?'U odabranom režimu primjenjuje se faktorski prinos, u ostalim mjesecima prinos RF-a. Ako režim nije dostupan, također se drži RF. Troškovi uključivanja i isključivanja nisu procijenjeni. ':(DATA.chartBasis||'Svi prinosi su bruto, uz mjesečni rebalans. '))+
      (state.saved.length?'Sačuvane usporedbe mijenjaju samo graf; pokazatelji i tablice prate glavni odabir.':'');
    if(!plotted.length){$('chart').innerHTML=`<div class="empty-chart">${!lines.length?'Odaberi barem jednu seriju iznad grafa.':'Nema podataka za odabrane serije i filtre.'}</div>`;return;}
    const W=Math.max(280,$('chartArea').clientWidth),H=W<450?275:305,L=45,R=14,T=17,B=35,pw=W-L-R,ph=H-T-B;
    const points=plotted.flatMap(line=>line.segments.flat()),all=points.map(p=>p.value);
    const min=Math.min(...all),max=Math.max(...all),padding=Math.max((max-min)*.12,dd?.015:.025),lo=min-padding,hi=dd?0:max+padding;
    const bounds=months.filter(m=>m.month>=state.from && m.month<=state.to);
    const lower=[...points.map(p=>p.time),...(bounds.length?[date(bounds[0].entry).getTime()]:[])];
    const upper=[...points.map(p=>p.time),...(bounds.length?[date(bounds.at(-1).exit).getTime()]:[])];
    const start=Math.min(...lower),end=Math.max(...upper),times=[...new Set(points.map(p=>p.time))].sort((a,b)=>a-b);
    const x=value=>L+(value-start)/(end-start||1)*pw;
    const y=value=>T+(hi-value)/(hi-lo||1)*ph;
    let svg=`<svg id="performanceChart" viewBox="0 0 ${W} ${H}" data-x-scale="calendar" data-domain-start="${new Date(start).toISOString().slice(0,10)}" data-domain-end="${new Date(end).toISOString().slice(0,10)}" role="img" tabindex="0" aria-label="${$('chartTitle').textContent}. Strelicama lijevo i desno pregledaj datume."><title>${$('chartTitle').textContent}</title>`;
    for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);svg+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#edf0f5"/><text x="${L-9}" y="${yy+3}" text-anchor="end" fill="#a1aabd" font-size="10">${dd?pct(v,0):fmt(v,2)}</text>`;}
    const longRange=end-start>540*86400000,ticks=[start],spacing=W<450?65:80;
    if(longRange){const yearSpan=(end-start)/(365.25*86400000),target=W<450?3:7;
      const step=[1,2,3,5,10,20].find(s=>s>=yearSpan/target)||20;
      for(let year=Math.ceil(new Date(start).getUTCFullYear()/step)*step;year<=new Date(end).getUTCFullYear();year+=step){const t=Date.UTC(year,0,1,12);if(t>start && t<end && x(t)-L>spacing && W-R-x(t)>spacing)ticks.push(t);}
    }else{const count=W<450?3:5;for(let i=1;i<count;i++){const t=start+(end-start)*i/count;if(x(t)-L>spacing && W-R-x(t)>spacing)ticks.push(t);}}
    ticks.push(end);
    for(const [i,t] of ticks.entries()){const dt=new Date(t),label=longRange?dt.getUTCFullYear():dt.toLocaleDateString('hr-HR',{month:'short',year:'numeric',timeZone:'UTC'});
      svg+=`<text data-axis="time" data-time="${dt.toISOString().slice(0,10)}" x="${x(t)}" y="${H-11}" text-anchor="${i===0?'start':i===ticks.length-1?'end':'middle'}" fill="#a1aabd" font-size="10">${label}</text>`;}
    svg+='<defs><linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#365eea" stop-opacity=".08"/><stop offset="100%" stop-color="#365eea" stop-opacity="0"/></linearGradient></defs>';
    for(const line of plotted){
      const path=line.segments.map(segment=>segment.map((p,i)=>`${i?'L':'M'}${x(p.time).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ')).join(' ');
      if(line.id==='portfolio4' && !line.saved && !dd){const area=line.segments.map(segment=>segment.map((p,i)=>`${i?'L':'M'}${x(p.time).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ')+` L${x(segment.at(-1).time)},${T+ph} L${x(segment[0].time)},${T+ph} Z`).join(' ');svg+=`<path d="${area}" fill="url(#areaGradient)"/>`;}
      svg+=`<path data-chart-series="${line.key}" data-signal="${line.id}" data-segments="${line.segments.length}" data-points="${line.segments.flat().length}" data-final-value="${line.segments.at(-1).at(-1).value}" d="${path}" fill="none" stroke="${line.color}" stroke-width="${line.id===state.focal || line.saved?2.4:1.7}" ${line.saved?'stroke-dasharray="6 3"':''} opacity="${line.id===state.focal || line.saved?1:.85}" stroke-linejoin="round" stroke-linecap="round"/>`;
    }
    svg+=`<line id="hoverLine" x1="0" x2="0" y1="${T}" y2="${T+ph}" stroke="#acb7cf" stroke-dasharray="3 3" visibility="hidden"/><g id="hoverDots"></g></svg>`;
    $('chart').innerHTML=svg;
    const el=$('performanceChart');
    function show(index){index=Math.max(0,Math.min(times.length-1,index));hoverIndex=index;const time=times[index],xx=x(time);
      $('hoverLine').setAttribute('x1',xx);$('hoverLine').setAttribute('x2',xx);$('hoverLine').setAttribute('visibility','visible');
      const active=plotted.map(line=>({line,point:line.segments.flat().find(p=>p.time===time)}));
      $('hoverDots').innerHTML=active.filter(item=>item.point).map(({line,point})=>`<circle cx="${xx}" cy="${y(point.value)}" r="3.5" fill="${line.color}" stroke="white" stroke-width="1.5"/>`).join('');
      $('chartTooltip').innerHTML=`<div class="tooltip-date">${dateLabel(new Date(time).toISOString().slice(0,10))}</div>`+active.map(({line,point})=>`<div class="tooltip-line"><span><i class="series-dot" style="background:${line.color}"></i>${escape(line.saved?line.label:line.short)}${point?.row && !point.row.active?' · cash RF':''}</span><strong>${point?(dd?pct(point.value):fmt(point.value,3)):'—'}</strong></div>`).join('');
      $('chartTooltip').hidden=false;
      const tipWidth=$('chartTooltip').offsetWidth;
      $('chartTooltip').style.left=Math.max(0,Math.min(W-tipWidth,xx+15>W-tipWidth?xx-tipWidth-15:xx+15))+'px';
      $('chartTooltip').style.top='22px';}
    el.addEventListener('pointermove',e=>{const px=(e.clientX-el.getBoundingClientRect().left)/el.getBoundingClientRect().width*W;let nearest=0,distance=Infinity;times.forEach((v,i)=>{const delta=Math.abs(x(v)-px);if(delta<distance){distance=delta;nearest=i;}});show(nearest);});
    el.addEventListener('pointerleave',()=>{$('chartTooltip').hidden=true;$('hoverLine').setAttribute('visibility','hidden');$('hoverDots').innerHTML='';});
    el.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();show((hoverIndex??0)+(e.key==='ArrowRight'?1:-1));}});
  }
  function renderTable() {
    const conditional=hasRegime();
    $('tableReturnLabel').textContent='CAGR';
    $('drawdownLabel').textContent='cijelo razdoblje držanja';
    $('comparisonBody').innerHTML=availableSeries().map(s=>{const m=calculate(currentStrategyRows,s.id),r=m.cagr;return `<tr class="${s.id===state.focal?'focal-row':''}" data-signal="${s.id}"><td><span class="table-series"><i class="series-dot" style="background:${s.color}"></i><strong>${seriesLabel(s)}${conditional && s.id!=='risk_free'?' / RF':''}</strong>${s.id==='portfolio4'?'<span class="table-badge">25% × 4</span>':''}</span></td><td class="${colorClass(r)}" data-metric="return" data-value="${Number.isFinite(r)?r:''}">${pct(r)}</td><td data-metric="vol" data-value="${Number.isFinite(m.vol)?m.vol:''}">${pct(m.vol)}</td><td data-metric="sharpe" data-value="${Number.isFinite(m.sharpe)?m.sharpe:''}">${fmt(m.sharpe)}</td><td class="negative" data-metric="dd" data-value="${Number.isFinite(m.dd)?m.dd:''}">${pct(m.dd)}</td><td>${pct(m.hit,1)}</td></tr>`;}).join('');
    $('tableNote').textContent='CAGR, volatilnost i drawdown obuhvaćaju cijeli datumski raspon. Sharpe koristi mjesečni višak prinosa prema RF-u.'+(conditional?' Izvan odabranog režima, uključujući nedostupne režimske podatke, drži se RF cash.':'');
  }
  function renderRegimeBars() {
    $('compareRegime').value=state.compare;$('regimeSeries').textContent=focal().label;
    const r=regimes.find(r=>r.id===state.compare);
    const groups=[0,1].map(level=>{const rows=strategyRows({...state,filters:{...state.filters,[r.id]:level}});return {level,rows,m:calculate(rows,state.focal)};});
    const max=Math.max(...groups.map(g=>Math.abs(g.m[state.regimeMetric])).filter(Number.isFinite),.0001);
    $('regimeBars').innerHTML=groups.map(g=>{const value=g.m[state.regimeMetric],selected=state.filters[r.id]===g.level,active=g.rows.filter(m=>m.active).length;return `<div class="regime-bar ${selected?'selected':''}" data-state="${g.level}"><div class="regime-bar-label"><span>${g.level?r.high:r.low}<small>${active} faktor / ${g.rows.length-active} cash</small></span><strong>${state.regimeMetric==='sharpe'?fmt(value):pct(value)}</strong></div><div class="bar-rail"><div class="bar-fill" style="width:${Number.isFinite(value)?Math.abs(value)/max*100:0}%;${value<0?'background:#d2a0aa':''}"></div></div></div>`;}).join('');
  }
  function renderCorrelation() {
    const components=availableSeries().filter(s=>s.component);
    let html='<table class="correlation-table"><thead><tr><th></th>'+components.map(s=>`<th scope="col">${s.short}</th>`).join('')+'</tr></thead><tbody>';
    for(const s of components){html+=`<tr><td>${s.short}</td>`;for(const t of components){const value=correlation(currentRows.map(m=>m.r[s.id]),currentRows.map(m=>m.r[t.id]));const alpha=Number.isFinite(value)?Math.abs(value)*.5:0;const background=Number.isFinite(value)?value<0?`rgba(193,109,140,${alpha*.55})`:`rgba(111,145,228,${alpha})`:'#f7f8fb';html+=`<td title="${s.label} × ${t.label}" style="background:${background};color:${value>.75?'#496697':'#7b8aa5'}">${fmt(value)}</td>`;}html+='</tr>';}
    $('correlationMatrix').innerHTML=html+'</tbody></table>';
  }
  function renderHeatmap() {
    $('heatmapTitle').textContent='Kalendar prinosa · '+seriesLabel(focal(),state,true);
    const selected=new Map(currentStrategyRows.map(m=>[m.i,m])),all=new Map(months.map(m=>[m.month,m]));
    const yearFrom=Number(state.from.slice(0,4)),yearTo=Number(state.to.slice(0,4));
    const labels=['Sij','Velj','Ožu','Tra','Svi','Lip','Srp','Kol','Ruj','Lis','Stu','Pro'];
    let html='<table class="heatmap-table"><thead><tr><th>Godina</th>'+labels.map(l=>`<th>${l}</th>`).join('')+'<th title="Složeni prinos faktora i casha u datumskom odabiru">Ukupno</th></tr></thead><tbody>';
    for(let year=yearFrom;year<=yearTo;year++){let total=1,n=0;html+=`<tr><td>${year}</td>`;
      for(let mo=1;mo<=12;mo++){const key=year+'-'+String(mo).padStart(2,'0'),row=all.get(key);
        if(!row){html+='<td class="unavailable" title="Nema opažanja">·</td>';continue;}
        if(!selected.has(row.i)){html+=`<td class="excluded" title="${monthLabel(key)} · izvan odabira">—</td>`;continue;}
        const held=selected.get(row.i),v=held.r[state.focal];total*=1+v;n++;
        const opacity=.12+Math.min(Math.abs(v)/.08,1)*.4;
        const bg=!held.active?'#edf1fa':v>=0?`rgba(89,173,143,${opacity})`:`rgba(204,115,129,${opacity})`;
        html+=`<td title="${escape(monthLabel(key)+' · '+pct(v)+' · '+(held.active?'faktor':'cash RF')+' · formacija '+dateLabel(row.date))}" style="background:${bg};color:${v>=0?'#397a65':'#a45a68'}">${fmt(v*100,1)}${!held.active?'<sup class="cash-mark">c</sup>':''}</td>`;}
      html+=`<td>${n?pct(total-1,1):'—'}</td></tr>`;}
    $('heatmap').innerHTML=html+'</tbody></table>';
  }
  function renderIC() {
    $('icBody').innerHTML=DATA.ic.signals.map((signal,j)=>{
      const records=currentRows.map(m=>DATA.ic.values[m.i][j]).filter(Boolean);
      const average=k=>mean(records.map(r=>r[k]).filter(Number.isFinite));
      const known=records.filter(r=>Number.isFinite(r[2])).length;
      return `<tr><td>${signal.label}</td><td>${fmt(average(2),4)}</td><td>${fmt(average(3),4)}</td><td>${fmt(average(4),4)}</td><td>${known}</td><td>${fmt(average(0),0)}</td></tr>`;
    }).join('');
  }
  function exportCSV() {
    const shown=availableSeries(),legColumn=DATA.legs?['portfolio_side']:[];
    const headers=['signal_date','entry_date','exit_date','holding_month','factor_active','cash_reason',...legColumn,...shown.map(s=>s.id),...regimes.map(r=>r.id+'_state'),...shown.filter(s=>s.id!=='risk_free').map(s=>'raw_'+s.id)];
    const rows=currentStrategyRows.map(m=>[m.date,m.entry,m.exit,m.month,m.active?1:0,m.cashReason,...(DATA.legs?[state.leg]:[]),...shown.map(s=>m.r[s.id]),...regimes.map(r=>m.states[r.id]??''),...shown.filter(s=>s.id!=='risk_free').map(s=>returnsFor(months[m.i])[s.id])]);
    const csv='\uFEFF'+[headers,...rows].map(row=>row.join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));
    const link=document.createElement('a');link.href=url;link.download=`quality_${state.from}_${state.to}_${hasRegime()?'factor_rf':'all'}${DATA.legs?'_'+state.leg:''}.csv`;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function render() {
    currentRows=filterRows();currentStrategyRows=strategyRows();renderControls();renderKPIs();renderLegend();renderChart();renderTable();renderRegimeBars();renderCorrelation();renderHeatmap();renderIC();saveHash();
    document.documentElement.dataset.ready='true';
  }
  // Numeric snapshot used for calculation verification; does not alter the UI or inputs.
  Object.defineProperty(window,'qualityReportSnapshot',{get:()=>({selected:currentRows.map(m=>m.date),metrics:currentMetrics,
    conditional:hasRegime(),series:state.focal,leg:state.leg,filters:{...state.filters},visible:[...state.visible],comparisons:state.saved,
    calendar:currentStrategyRows.map(m=>m.date),activeMonths:currentRows.length,cashMonths:currentStrategyRows.length-currentRows.length,
    unavailableCashMonths:currentStrategyRows.filter(m=>m.cashReason==='regime_unavailable').length})});
  Object.defineProperty(window,'profitabilityReportSnapshot',{get:()=>window.qualityReportSnapshot});
  initialize();
})();
