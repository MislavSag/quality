"""Compare local gross results with official public US RMW factor returns."""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/"data"/"reference"
OUT=ROOT/"results"/"public_comparison"
URL="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def read_factor(name,daily=False):
    path=REF/name
    with zipfile.ZipFile(path) as archive:
        text=archive.read(archive.namelist()[0]).decode("utf-8-sig")
    pattern=r"^\d{8}," if daily else r"^\d{6},"
    lines=[line for line in text.splitlines() if re.match(pattern,line)]
    data=pd.read_csv(io.StringIO("date,Mkt_RF,SMB,HML,RMW,CMA,RF\n"+"\n".join(lines)))
    data["date"]=pd.to_datetime(data.date.astype(str),format="%Y%m%d" if daily else "%Y%m")
    for col in data.columns[1:]:
        data[col]=data[col].where(data[col]>-99)/100
    assert not data.date.duplicated().any()
    return data


def stats(name,returns,window):
    r=np.asarray(returns,dtype=float)
    assert np.isfinite(r).all()
    equity=np.r_[1,np.cumprod(1+r)]
    return {"series":name,"window":window,"months":len(r),
        "cagr":equity[-1]**(12/len(r))-1,"annual_mean":r.mean()*12,
        "annual_vol":r.std(ddof=1)*np.sqrt(12),
        "sharpe_cash0":r.mean()/r.std(ddof=1)*np.sqrt(12),
        "monthly_max_drawdown":np.min(equity/np.maximum.accumulate(equity)-1)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    filename="F-F_Research_Data_5_Factors_2x3_CSV.zip"
    factors=read_factor(filename)
    factors["month"]=factors.date.dt.to_period("M")
    monthly=pd.read_csv(ROOT/"results"/"monthly_long_short"/"monthly_returns.csv",
                        parse_dates=["signal_date","entry_date","exit_date"])
    monthly=monthly[(monthly.scenario=="gross_unconstrained")&(monthly.missing_policy=="flat_unknown")].copy()
    monthly["month"]=monthly.entry_date.dt.to_period("M")
    results=[]
    comparisons=[]
    for signal,group in monthly.groupby("signal"):
        joined=group.merge(factors[["month","RMW"]],on="month",validate="one_to_one")
        assert len(joined)==len(group)==235
        results.append(stats(signal,joined["return"],"holding windows labelled by entry month"))
        comparisons.append(joined[["signal","signal_date","entry_date","exit_date","return","RMW"]])
    benchmark=factors[factors.month.isin(monthly.month.unique())]
    assert len(benchmark)==235
    results.append(stats("US_RMW_official_monthly",benchmark.RMW,"calendar months 2007-02 to 2026-08"))
    daily_name="F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    if (REF/daily_name).exists():
        daily=read_factor(daily_name,True)
        windows=monthly.drop_duplicates("signal_date").sort_values("signal_date")
        aligned=[]
        for row in windows.itertuples():
            values=daily[(daily.date>row.entry_date)&(daily.date<=row.exit_date)]
            assert len(values)>=17 and len(values)<=25
            aligned.append({"signal_date":row.signal_date,"entry_date":row.entry_date,
                "exit_date":row.exit_date,"RMW_daily_compounded":np.prod(1+values.RMW)-1,
                "n_daily_observations":len(values)})
        aligned=pd.DataFrame(aligned)
        results.append(stats("US_RMW_daily_compounded",aligned.RMW_daily_compounded,"exact local entry-close to exit-close windows"))
        aligned.to_csv(OUT/"rmw_aligned_windows.csv",index=False)
    pd.DataFrame(results).to_csv(OUT/"comparison.csv",index=False)
    pd.concat(comparisons,ignore_index=True).to_csv(OUT/"monthly_comparison.csv",index=False)
    sources=[]
    for name in [filename,daily_name]:
        p=REF/name
        if p.exists():
            sources.append({"url":URL+name,"local_file":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={"sources":sources,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_vintage":"202609 CRSP database per downloaded file header",
        "comparison":"gross with gross; no trading costs, borrow costs or cash interest added",
        "calendar_window_difference":"local strategy trades next-session closes; official monthly RMW uses calendar months",
        "daily_comparison":"official daily RMW is compounded within exact local close-to-close windows; not an exact replica of monthly RMW construction",
        "limitations":"Different definitions, universes, weights, rebalancing and accounting sources; no causal attribution of performance gap"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(results).to_string(index=False))


if __name__=="__main__":
    main()
