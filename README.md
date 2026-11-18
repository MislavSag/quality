# Quality

Izvještaj: [Ocjene kvalitete po Tintoru na američkom tržištu](https://mislavsag.github.io/quality/).
Mape `data/` i `results/` nisu u repozitoriju jer potječu od dobavljača podataka.

Projekt istraživanja **quality faktora**. Obuhvaća sedam dimenzija, zajedničke
ALL7 i CORE5 scoreove, mjesečne kvintilne portfelje i zajedničke regresijske
factor-mimicking portfelje. [Zajednički quality faktor](results/combined_quality/report.html)
koristi **povijesne IWV sastavnice iz paketa etfconstituents**, uz postojeće
filtre investabilnosti, te prikazuje rezultate, troškove i uparene uzorke.
CORE5 s IWV članstvom počinje u lipnju 2009., ALL7 u siječnju 2012.; ranije
formacije u punom kalendaru drže RF. Članstvo koristi posljednji tada dostupan
snapshot i točan puni SID s provjerenim identitetom. [Prethodni širi univerzum](results/combined_quality__broad_20261112/report.html)
sačuvan je kao arhivirana usporedba.
[Interaktivni quality izvještaj](results/combined_quality/statistics.html) omogućuje
više serija, long/short strane i režimske filtre uz RF u neaktivnim mjesecima.
[Pregled quality dimenzija](results/quality_dimensions/report.html) uključuje
scoreove, pokrivenost i statistike. [Detaljni statistički izvještaj](results/quality_dimensions/statistics.html)
omogućuje grafove, režime i usporedbu long/short strana.

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
2026. Promjena naziva projekta ne predstavlja novi backtest.

Pokretanje iz korijena projekta:

```powershell
python scripts/analyze_quality.py
```

Skripta čita postojeće Parquete iz `H:/strategies/characteristics` i
`H:/strategies/characteristics_profitability_20261110`. Izvori se ne mijenjaju.
Lokalni mjesečni cache sprema se u `data/derived`, a izvještaj i CSV rezultati
u `results/initial_analysis`.

Signal se očitava na kraju mjeseca; budući prinos mjeri se od zatvaranja
sljedeće sjednice do zatvaranja sjednice nakon sljedećeg kraja mjeseca.
Filtri pri formiranju: fundamentalni izvještaj u USD, tržišna kapitalizacija
najmanje 100 mil. USD, sirova cijena najmanje 5 USD i prosječan dnevni promet
u 21 sjednici najmanje 1 mil. USD. Buduća raspoloživost cijene nije uvjet
za ulazak u formacijski uzorak.

Rezultati uključuju rangovni IC, kvintile jednake težine, usporedbe na
zajedničkim uzorcima, tri podrazdoblja i mjesečne presječne regresije uz
value, veličinu, momentum, rast prihoda i rast imovine. Standardne pogreške
vremenskih prosjeka koriste HAC s tri laga.

IC se prikazuje u tri jasno odvojena oblika:

- `ic_spearman` (kompatibilni naziv `ic`): korelacija rangova signala i prinosa.
- `ic_cosine_raw`: skalarni produkt izvornog signala i izvornog budućeg prinosa,
  podijeljen produktom njihovih L2 normi, bez centriranja i prilagodbe riziku.
- `ic_cosine_rank_centered`: cosine centriranih rangova signala i izvornih
  budućih prinosa. Rangovi se centriraju na formacijskom uzorku prije
  izostavljanja nepoznatih ishoda. Prinos se ne centrira i ne rangira.

Sve koriste iste poznate ishode i najmanje 100 opažanja. Nulta norma daje
nedostajuću vrijednost. Necentrirani cosine može uključivati tržišnu komponentu
zbog nenulte sredine signala. Ove specifikacije nisu deklarirane kao točne
replike Paleologove obrade prinosa ili modela rizika.

Za dopunu ranije izračunatih rezultata bez ponovnog izračuna portfelja:

```powershell
python scripts/add_cosine_ic.py
python scripts/validate_initial_analysis.py
```

Usporedba je u `results/initial_analysis/ic_comparison.csv`; sačuvane su i
izvorne Spearmanove tablice u poddirektoriju `before_cosine`.

Ovo su eksplorativne bruto statistike na vendorovoj prilagođenoj cijeni.
Nepoznati delisting ishodi ostaju nepoznati. Troškovi trgovanja i shortanja
nisu modelirani; Q5−Q1 je statistička razlika, a ne verificirana izvršiva
strategija. Povijesne revizije, procijenjeni datumi objave i izvorni odabir
vrijednosnica ograničavaju zaključke.

Datoteka `manifest.json` evidentira ulaze, njihove veličine, vremena izmjene,
sažetke Parquet metapodataka, hash skripte i parametre analize.

## Mjesečni long–short

```powershell
python scripts/backtest_monthly_long_short.py
python scripts/validate_monthly_long_short.py
```

Pravila su u `config/monthly_long_short.json`. Tri odabrana signala koriste
gornjih i donjih 20%, jednake težine unutar strane, cilj +100%/−100% i isti
odgođeni ulaz kao početna analiza. Između mjesečnih rebalansa težine driftaju.
Rezultati su u `results/monthly_long_short/report.html`.

Obračun uključuje ulazne i završne troškove, promet prema driftanim pozicijama
te posudbu po prosječnoj početnoj i završnoj short vrijednosti. Troškovni
scenariji su pretpostavke, uključujući scenarij djelomične nedostupnosti i
ograničenja količine shorta. Ne koriste se potvrđene povijesne brokerove
količine ili naknade. Kamata na novac, kolateral i financiranje margine je nula.

Pozicija s nepoznatom ulaznom cijenom ne otvara se. Nepoznat izlaz u osnovnom
scenariju pretpostavlja prinos nula i zatvaranje, a zaseban nepovoljni scenarij
za iste pozicije uzima pad 100% za long i rast 100% za short. To su testovi
osjetljivosti; ne rekonstruiraju stvarni delisting događaj ili isplatu.
Drawdown se mjeri na mjesečnim krajevima držanja.

## Javni RMW benchmark

`python scripts/compare_public_rmw.py` uspoređuje bruto rezultate s javnim
američkim RMW faktorom Kennetha Frencha. Preuzete mjesečne i dnevne ZIP
datoteke su u `data/reference`, a rezultati i izvorni URL-ovi s hashom datoteka
u `results/public_comparison`.

Mjesečna usporedba pokriva veljaču 2007. – kolovoz 2026. Dodatno se javni
dnevni RMW slaže u identične prozore držanja od našeg ulaznog do izlaznog
zatvaranja. To je dijagnostička usporedba; konstrukcija javnog mjesečnog i
dnevnog faktora nije potpuno jednaka. Uspoređuju se bruto serije bez dodatnih
troškova posudbe i trgovanja.

## Profitabilnost kroz režime

```powershell
python scripts/analyze_regimes.py
python scripts/validate_regimes.py
```

Izvještaj je u `results/regimes/report.html`, a pravila su u `config/regimes.json`.
Šest režima koristi cijene poznate na formacijskom zatvaranju: SPY volatilnost
21 dan, SPY trend 126 dana, korelaciju SPY–TLT 63 dana, udio podobnih dionica
s pozitivnim 126d prinosom, robusnu disperziju 21d prinosa i medijan Amihudove
nelikvidnosti 21 dan. Širina je zamjena za MA200 iz priloga. Pragovi za tri
adaptivne podjele koriste samo prethodne mjesece, uz najmanje 12 opažanja.
Makro serije nisu uključene jer lokalna povijest nije point-in-time.

Zajednički uzorak ima 223 formacijska mjeseca od siječnja 2008. do srpnja
2026. Ponovno se koriste postojeći mjesečni long–short portfelji, tri scenarija
troškova i dvije pretpostavke nepoznatih izlaza. IC tablice pokrivaju svih 21
signala; primarni testovi prinosa odnose se na FCF/imovinu, CFO/imovinu i
bruto dobit/imovinu. Javni RMW služi kao usporedba u istim prozorima držanja.

Uvjetni godišnji prosjek je 12 puta prosječni mjesečni prinos, nije CAGR
strategije koja trguje samo u odabranom režimu. Kontinuirane regresije koriste
standardizaciju iz prošlosti, HAC tri kalendarska mjeseca i BH korekciju
višestrukog testiranja. Podjela prije/od 2017. provjerava vremensku stabilnost;
nije netaknut out-of-sample test. Nije optimizirana ni provedena režimska
alokacija. Provjere uključuju rekonstrukciju povijesnih prefiksa, promjenu
budućih podataka, neovisnu kontrolu izvora i HAC kovarijance te usklađivanje
uvjetnih prinosa i IC-a s prethodno objavljenim serijama.

## Quality: bundle profitabilne komponente

```powershell
python scripts/build_quality_profitability_bundle.py
python scripts/validate_quality_profitability_bundle.py
```

`PROFIT_BUNDLE4` je mjesečni indikator s četiri jednake težine: FCF/imovina,
CFO/imovina, bruto dobit/imovina i naš godišnji FF2015 OP/kapital. Posljednja
komponenta je računovodstveni signal povezan s RMW-om, a ne javna serija prinosa
RMW-a niti točna replika aktualne Frenchove definicije.

Score od 0 do 100 je prosjek percentilnih rangova puta 100; svaki percentil je
`(prosječni rang - 0.5) / N`. Rangovi koriste samo podobne dionice sa sva četiri
podatka na dan formacije. Sve četiri komponente moraju biti konačne; nema
imputacije niti preraspodjele težina ako komponenta nedostaje. Interni signal
`bundle4` jednak je `bundle4_score / 50 - 1` radi nulte sredine na formacijskom
uzorku. Pravila i fiksne težine su u `config/quality_profitability_bundle.json`.

U `results/quality_profitability_bundle/monthly_bundle_scores.parquet` nalaze se svi
mjesečni ključevi, pojedinačni percentilni rangovi, broj dostupnih komponenti
i scoreovi. Zadnji presjek je u `latest_bundle_scores.csv`, a pregled u
`report.html`. Izvorni panel i upstream paket nisu promijenjeni; dnevni bundle
ovim korakom nije izračunat.

Kontrolirane usporedbe koriste svaku od četiri komponente te bundle 3 na istom
univerzumu. Zaseban `bundle3` koristi širi univerzum sa sva tri TTM podatka.
IC, troškovi, nepovoljni nepoznati izlazi i povijesni režimi koriste prethodno
definirane metode. Usporedbe prije/od 2017. nisu netaknut out-of-sample test.
Nema režimskog uključivanja niti optimizacije težina.

Odvojena dijagnostička `four_factor_basket.csv` doslovno kombinira tri ranija
bruto long–short prinosa i spremljeni javni RMW, po 25% kapitala na početku
svakog mjesečnog prozora. Ta serija ne sudjeluje u rangiranju dionica; nije
provjeren izvršiv portfelj i nema procijenjene troškove javnog RMW-a.

## Odabrani portfelj četiriju faktora

Korisnik je odabrao portfelj prinosa četiriju faktora. Glavna specifikacija je
u `config/factor_portfolio.json`: po 25% kapitala u tri postojeća lokalna
long–short faktora (FCF, CFO, bruto dobit prema imovini) i javni RMW. Težine se
vraćaju na 25% svaki mjesec, a unutar mjeseca driftaju. Nema zamjene javnog
RMW-a lokalnim OP/kapital signalom. Prethodni indikator za rangiranje dionica
ostaje zaseban istraživački rezultat.

```powershell
python scripts/analyze_factor_portfolio.py
```

Glavni izvještaj je `results/factor_portfolio/report.html`, a mjesečni prinosi
su u `portfolio_monthly.csv`. Rezultati su bruto: troškovi javnog RMW-a i
ukupne implementacije nisu procijenjeni. Usporedbe komponenti, korelacije,
režimi, doprinosi varijanci i provjere računanja spremljeni su uz izvještaj.

Glavni `report.html` je samostalan interaktivan izvještaj s ugrađenim podacima,
CSS-om i JavaScriptom; otvara se lokalno bez servera ili internetske veze.
Filtri za datum i šest režima ažuriraju pokazatelje, usporedbu šest serija,
graf vrijednosti/drawdowna, korelacije, kalendar prinosa i IC tablicu 21 signala.
Odabrani mjeseci mogu se preuzeti u CSV-u, a režimski odabir čuva se u URL hashu.
Statična verzija ostaje u `report_static.html`.

Režimski filtri koriste isti zajednički uzorak od 223 mjeseca za odluku o
aktiviranju faktora. Više filtara znači njihov presjek. Aktualni prikaz
uključuje cijeli datumski raspon: faktor u odabranom režimu, RF cash u svim
ostalim mjesecima, uključujući početne mjesece bez dostupnog režima. CAGR,
volatilnost i drawdown obuhvaćaju cijelu putanju. Sharpe koristi višak prinosa
prema RF-u za iste prozore držanja. IC i korelacije izvornih faktora ostaju
uvjetni opisi samo aktivnih mjeseci.

Samo za obnovu HTML-a iz postojećih rezultata:

```powershell
python scripts/build_interactive_report.py
```

Izvori su u `scripts/report_assets/portfolio.html`, `portfolio.css` i
`portfolio.js`. Browser provjere u `scripts/validate_interactive_report.py`
uspoređuju brojeve s neovisnim pandas izračunima i provjeravaju filtere,
izvoz, IC i prikaz na različitim širinama. Koriste prethodno otvorenu
Playwright CLI sesiju `quality_report` i lokalni preview na portu 8767.

Graf uvijek koristi stvarno kalendarsko vrijeme i godine na x-osi. Aktualne
režimske varijante imaju kontinuirane krivulje s RF prinosima u cash mjesecima.
Kućice omogućuju prikaz više faktora, portfelja i čistog RF benchmarka.
Gumb `Dodaj usporedbu` čuva trenutačnu glavnu seriju, datumski raspon i filtre
kao dodatnu krivulju. Spremljene krivulje ostaju na grafu dok se glavni odabir
mijenja; ne mijenjaju pokazatelje i tablice. Krivulje se zasebno normaliziraju
na 1 na svojem prvom ulazu. Odabir serija i usporedbe čuvaju se u URL hashu.

RF se računa iz iste spremljene Frenchove dnevne arhive kao javni RMW:
produkt dnevnih `(1 + RF) - 1` za `entry_date < date <= exit_date`. Izgradnja
provjerava hash arhive i broj dnevnih opažanja prema spremljenim RMW prozorima;
RF se ne imputira nulom. Rezultati su u `risk_free_windows.csv` i
`regime_cash_summary.csv`. U aktivnim mjesecima ostaje izvorni bruto faktorski
prinos, bez dodatnog RF-a. Cash mjeseci u kalendaru imaju oznaku `c`, a CSV
izvoz sadrži cijelu putanju, oznaku aktivnosti, razlog casha, RF i izvorne
faktorske prinose. Troškovi prelaska između faktora i casha nisu procijenjeni.

## Zajednička regresija i factor-mimicking portfelji

```powershell
python scripts/analyze_joint_fmp.py
python scripts/validate_joint_fmp.py
python scripts/build_joint_fmp_sides.py
python scripts/validate_joint_fmp_sides.py
python scripts/build_joint_fmp_report.py
```

Nova analiza je odvojena od ranijih portfelja u `results/joint_fmp/report.html`.
Procjenjuje FCF/imovinu, CFO/imovinu i bruto dobit/imovinu istodobno s tržišnom
konstantom, log veličine, book-to-market, momentumom 12–1, SPY betom 63 dana,
log rezidualne volatilnosti i log Amihudove nelikvidnosti. Sektori nisu uključeni:
povijesne klasifikacije nisu potvrđene. Javni RMW ostaje vanjska serija prinosa.

Pravila su u `config/joint_fmp.json`. Četiri specifikacije kombiniraju winsorizirani
z-score (1/99 percentil) ili normalne kvantile rangova s WLS ponderima korijena
kapitalizacije ili inverzne specifične varijance. Standardizacija oduzima
kapitalizacijski ponderiranu sredinu i dijeli jednako ponderiranom populacijskom
SD. Jedinstveni potpuni uzorak određuju formacijske karakteristike, bez filtriranja
po budućem ishodu i bez imputacije karakteristika. Prosječno ima 1.654 dionice.

Za `r - RF = X f + epsilon`, QR dekompozicija daje operator
`A = (X' W X)^-1 X' W`. Redak za pojedini faktor daje raw FMP, čiji su prinos
i regresijski koeficijent jednaki uz istu pretpostavku o nepoznatim ishodima.
Provjerava se `X' h = e_j`: jedinica ciljanom stupcu, nula svim ostalima.
Težine X i W ne koriste buduće prinose. Stupci `raw_*` u `weights.parquet`
nisu portfelji s fiksnim grossom.

Specifična varijanca koristi EWMA kvadrata reziduala regresije na kontrole,
poluživot 12 mjeseci i najmanje 12 poznatih reziduala. Rezidual ulazi u povijest
tek kada `exit_date <= signal_date`, pa ni prethodni formacijski mjesec nije
automatski dostupan. Prije dovoljne povijesti koristi se SPY rezidualna varijanca
63 dana podijeljena s 12. Primjenjuje se pod varijance od 10% medijana presjeka i
ograničenje inverznih pondera na 1/99 percentil. To nije komercijalni Barra model.

Pojedinačni FMP portfelji skaliraju se na +100% long / −100% short, čime se
mijenja jedinična faktorska izloženost. `basket3` prosječi tri ciljna portfelja uz
netiranje pozicija; prosječan gross primarne verzije pada na oko 62%.
`basket3_gross2` dodatno skalira netiranu košaricu na gross 200%, prema težinama
poznatima pri formiranju. Ista se pravila primjenjuju na kvintilne kontrolne
portfelje iz istoga univerzuma. FCF i CFO FMP prinosi primarne verzije imaju
korelaciju oko −0,88, za razliku od pozitivne korelacije prijašnjih kvintila.

Izvještaj ima osam lokalnih stranica za četiri modela i dvije troškovne
pretpostavke, više usporednih linija, godine na x-osi i postojeće režimske filtre
s RF prinosom izvan aktivnog režima. Glavni prikaz je bruto netirana košarica
reskalirana na gross 200%. Četverofaktorske usporedbe s javnim RMW-om ostaju
bruto i na troškovnim stranicama. Režimski prikaz troškovnih prinosa preslikava
prinose stalno aktivne strategije: promet nakon promjene režima nije ponovno
izračunat.

`factor_returns.csv` čuva koeficijente svih deset stupaca; `design.parquet`
transformirane izloženosti i pondere; `weights.parquet` raw i skalirane FMP
težine; `diagnostics.csv` koncentraciju, kolinearnost, izloženost, rizik i
nepoznate ishode. `summary.csv` razlikuje cijelo razdoblje i podrazdoblja prije/od
2017., bruto/troškovne scenarije i dvije pretpostavke nepoznatih ishoda.

Provjere neovisno rekonstruiraju transformacije, uspoređuju koeficijente sa
statsmodels WLS-om, sve mjesečne FMP identitete i skalirane prinose te 470
sekvencijalnih prognoza specifičnog rizika. `validate_joint_fmp_report.py`
provjerava stvarni browser u prethodno otvorenoj Playwright CLI sesiji `joint_fmp`
na lokalnom previewu porta 8768. Odbijeni ulazni nalozi mogu narušiti preciznu
neutralnost praktičnog portfelja; teorijski FMP ostaje zasebno spremljen.
Nepoznati delisting ishodi, revizije izvještaja, pretpostavljeni short troškovi,
nulta aktivna kamata na cash/kolateral i izostanak sektora ostaju ograničenja.

Odabir **Strana portfelja** u FMP izvještaju nudi Long–short, Long 100% i
Short 100%. Zasebne strane koriste relativne težine pozitivnih odnosno negativnih
pozicija nakon netiranja, normalizirane na 100% NAV-a prije odbijanja ulaznih
naloga. Svaka strana ima vlastiti mjesečni rebalans, drift, promet i završno
zatvaranje. Posudba se obračunava samo na short; pozitivni short prinos označava
dobit od shortanja. U aktivnim mjesecima nema dodatne kamate na cash/kolateral,
a u cash režimu ostaje RF. Odabir jedne strane više nije neutralan FMP.

`side_monthly_returns.csv` i `side_summary.csv` čuvaju odvojene obračune.
Podjela RMW-a i kombinacija s RMW-om nije dostupna jer imamo samo ukupne
faktorske prinose; te serije izostavljaju se pri odabiru jedne strane. Dvije
skale iste netirane košarice imaju identične normalizirane strane pa se prikazuju
jednom. Odabir strane čuva se u URL-u, CSV izvozu i spremljenim usporedbama;
long i short mogu se istodobno dodati na graf. CAGR zasebnih strana nije aditivna
dekompozicija CAGR-a long–short portfelja. Neovisna provjera 18.800 mjesečnih
obračuna uključuje drugačiji solver troškova i NAV-a te predznak short dobiti.


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

## Kompozitni indikatori quality dimenzija

Konfiguracija je u `config/quality_dimensions.json`. Pokretanje:

```powershell
python scripts/build_quality_dimensions.py
python scripts/validate_quality_dimensions.py
```

[HTML pregled](results/quality_dimensions/report.html) omogućuje izbor dimenzije,
pregled povijesne pokrivenosti te pretragu i sortiranje posljednjeg presjeka.
Scoreovi su u `results/quality_dimensions/monthly_dimension_scores.parquet`,
po ključu `security_id, signal_date`, s istim mjesečnim ključevima kao izvorni
panel. Za svaku dimenziju postoje stupci `_score`, `_signal` i `_component_count`.

| Dimenzija | Sastavnice, usmjerene prema višem scoreu |
| --- | --- |
| `profitability` | FCF/imovina, CFO/imovina, GP/imovina, postojeći FF2015 OP/kapital |
| `earnings_quality` | Negativni cash-flow accruals, operativni accruals i NOA/imovina |
| `financial_safety` | Negativni dug/imovina, pozitivno pokriće kamata, negativni pritisak refinanciranja |
| `earnings_stability` | Negativna 20-kvartalna volatilnost dobiti, CFO-a i prodaje prema prosječnoj imovini |
| `quality_growth` | Petogodišnje promjene ROA-a, ROE-a, GP/imovina, CFO/imovina i bruto marže |
| `investment_discipline` | Negativni rast imovine i rast PPE-a plus zaliha prema početnoj imovini |
| `payout_dilution` | Dividendni prinos, negativno neto izdavanje kapitala i negativni rast broja dionica |

Široki `profitability_broad` ostaje zaseban usporedni indikator s tri sastavnice.
Svaki score je 100 puta prosjek usmjerenih percentila `(prosječni rang − 0,5)/N`,
uz jednake težine. Rangovi se računaju u vlastitom podobnom potpunom presjeku
dimenzije, uz najmanje 100 dionica. Sve sastavnice moraju biti konačne;
nema dopunjavanja nulom ni promjenjivih težina. Interni signal je `score/50 − 1`.

Rast je projektni proxy promjene omjera, a ne QMJ-ov rast rezidualne dobiti po
dionici. Koristi isti SID točno 60 kalendarskih mjeseci ranije, uključujući
tadašnje raspoložive vrijednosti i bez zahtjeva tadašnje investicijske podobnosti.
Petogodišnji indikator zato počinje 2012. Stabilnost koristi već postojeći
20-kvartalni prozor s najmanje 12 konačnih opažanja i nije volatilnost ROE-a.

Radni kapital i otplata duga izostavljeni su iz osnovnih sastavnica zbog slabe
rane pokrivenosti; CFO/neto dobit izostavljen je zbog tumačenja kod gubitaša.
Dividendne vrijednosti koje nedostaju ostaju nedostajuće. Za financijske sektore
nema posebnog povijesnog tretmana. Payout prinosi uključuju i cjenovnu dimenziju,
a sigurnost i stabilnost mogu se preklapati pri budućem ukupnom scoreu.

Izvještaj i manifest bilježe pokrivenost, formule i ograničenja izvora. Provjera
uspoređuje sve scoreove i rangove neovisnom implementacijom, reprodukciju
postojećeg profitability bundlea, kalendarske lagove i odabrane stavke izvornih
izvještaja. Zasebni portfeljni rezultati dimenzija opisani su u nastavku.

## Statistike i mjesečni portfelji quality dimenzija

```powershell
python scripts/analyze_quality_dimensions.py
python scripts/build_quality_dimension_statistics_report.py
python scripts/validate_quality_dimension_statistics.py
```

Konfiguracija je u `config/quality_dimension_statistics.json`. Za svaki od sedam
indikatora i široki profitability usporedni indikator kupuje se gornji kvintil,
a shorta donji, uz jednako ponderiranje unutar strana i ciljeve +100% / −100%.
Long i short dodatno se zasebno rebalansiraju na 100% notionala. Pozitivan prinos
short prikaza znači dobit od shortanja. Signal i podobnost ne ovise o dostupnosti
budućeg prinosa; neizvršivi ulazi obrađuju se tek pri formiranju pozicija.

Prikazuju se Spearman IC, cosine internog kompozita i cosine centriranih rangova,
uz HAC s tri laga. IC koristi samo poznate buduće ishode; računovodstvo portfelja
zadržava nepoznate ishode uz osnovni nulti prinos i zaseban nepovoljan scenarij.
Postoje bruto, 5 bps + 0,5% godišnje posudbe i 10 bps + 3% posudbe scenariji.
Troškovi uključuju drift težina, apsolutni promet i završno zatvaranje pozicija.

U mjesecima bez indikatora i izvan izabranog režima drži se RF. Aktivni portfelji
ne dobivaju dodatnu cash/kolateral kamatu. Neto režimske krivulje preslikavaju
kontinuirane mjesečne neto prinose; promet režimskog uključivanja i isključivanja
nije ponovno izračunat. Režimi koriste postojeće formacijske podatke bez novih
pragova prilagođenih rezultatima. Različiti univerzumi ostaju ograničenje usporedbe.

Osnovni izvještaj nudi cijeli kalendar, povijest od prvog indikatora, zajedničko
razdoblje od 2012. i podrazdoblja prije/od 2017. Detaljni izvještaji dodatno imaju
slobodan datumski raspon, režimske filtre, long/short izbor, više grafičkih serija,
spremljene usporedbe i CSV izvoz. Cash mjeseci jasno su odvojeni od aktivnih.

Rezultati su u `results/quality_dimensions/statistics_summary.csv`,
`statistics_monthly_returns.csv`, `statistics_regime_cash_summary.csv`,
`monthly_ic.csv` i `ic_summary.csv`. Formacijske pozicije spremljene su u
`statistics_formation_holdings.parquet`. `statistics_validation.json` bilježi
neovisnu provjeru kvintila, svih mjesečnih knjiga zasebnim solverom troškova,
RF-a, IC-a, sažetaka i reprodukcije ranijih profitability prinosa.
Nova analiza koristi sortirane portfelje; zajednički FMP dimenzija zaseban je
sljedeći istraživački korak.

## Tintor / ZSE fundamentalna strategija (Claude)

Povijesni test strategije kvalitete na Zagrebačkoj burzi prema formulama iz knjige
J. Tintora *Poslovna analiza*. [Izvještaj](results/tintor_claude/report.html),
[mapa formula](results/tintor_claude/formula_map.md) i
[ručna provjera izdavatelja](results/tintor_claude/manual_verification.md).
Parametri su u `config/tintor_claude.json`; podaci (`data/tintor_claude/`) nisu u Gitu.

```powershell
python scripts/tintor_claude/fetch.py master      # popis vrijednosnica, CROBEX, CROBEXtr
python scripts/tintor_claude/fetch.py prices      # dnevni cjenici ZSE od 2010.
python scripts/tintor_claude/fetch.py combine     # cjenici i EHO metapodaci u parquet
python scripts/tintor_claude/fetch.py meta        # nadopuna EHO metapodataka
python scripts/tintor_claude/fetch.py reports     # XLS/XLSX financijski izvještaji
python -P scripts/tintor_claude/parse_reports.py  # AOP retci iz obrazaca
python -P scripts/tintor_claude/fundamentals.py   # stavke po objavi, mapa AOP-a
python -P scripts/tintor_claude/build_prices.py   # panel cijena, korporativne akcije
python -P scripts/tintor_claude/signals.py        # presjeci po datumu signala, dividende
python -P scripts/tintor_claude/tintor.py         # provjera formula na knjižnim primjerima
python -P scripts/tintor_claude/backtest.py dev   # razvojno razdoblje
python -P scripts/tintor_claude/backtest.py final # završno razdoblje
python -P scripts/tintor_claude/verify.py 2026-11-05 HT,PODR,KOEI,ADPL
python -P scripts/tintor_claude/report.py
# izvoz knjiga za aplikaciju ZSE Analitika (zse-research), zatim tamo publish_zse_tintor_claude.py
python -P scripts/tintor_claude/export_app.py <zse-research> <zse-research>/data/derived/zse_datasets/versions/<BAZA> <RUN_ID>
```

`fetch.py meta` polazi od postojećih EHO metapodataka u
`C:/Users/Mislav/projects_r/zsefundamentals/data/files`. Knjiga se preuzima iz Azure kontejnera `poslovna-analiza` u `data/tintor_claude/book`.
