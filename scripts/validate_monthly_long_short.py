"""Hand-calculated portfolio cases and published monthly accounting checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_monthly_long_short import post_cost_nav, simulate, formation_books, ROOT, OUT

config=json.loads((ROOT/"config"/"monthly_long_short.json").read_text())
config["initial_nav"]=1.0


def book(date="2020-01-31",entry="2020-02-03",exit="2020-03-04"):
    return pd.DataFrame({"security_id":["A","B"],"ticker":["A","B"],
        "signal_date":pd.Timestamp(date),"entry_date":pd.Timestamp(entry),"exit_date":pd.Timestamp(exit),
        "side":[1,-1],"requested_weight":[1.,-1.],"entry_executable":True,
        "availability_u":[.8,.8],"PV_AVG_DOLLAR_VOLUME_21D":1e9,
        "forward_return":[.1,-.1],"return_observed":True}).set_index("security_id",drop=False)


gross=dict(config["scenarios"][0])
b=book()
x,_=simulate([b],gross,config)
assert np.isclose(x.nav_end.iloc[0],1.2)
assert np.isclose(x.turnover_full_notional.iloc[0],2.)
cost=dict(gross,trading_bps=10)
x,_=simulate([b],cost,config)
assert np.isclose(x.nav_end.iloc[0],(1.2-.002)/1.002)
borrow=dict(gross,borrow_rate=.03)
x,_=simulate([b],borrow,config)
assert np.isclose(x.borrow_cost.iloc[0],.95*.03*30/365)
assert np.isclose(x.nav_end.iloc[0],1.2-.95*.03*30/365)

# Rebalance using marked previous positions: [1.1,-0.9] -> [1.2,-1.2].
b2=book("2020-02-28","2020-03-04","2020-04-03")
x,_=simulate([b,b2],gross,config)
assert np.isclose(x.nav_end.iloc[-1],1.44)
assert np.isclose(x.turnover_full_notional.iloc[1],.4/1.2)

# Missing exits are explicit, and neither policy changes formation membership.
missing=b.copy()
missing.loc["A","return_observed"]=False
missing.loc["A","forward_return"]=np.nan
flat,_=simulate([missing],gross,config)
adverse,_=simulate([missing],gross,config,True)
assert np.isclose(flat.nav_end.iloc[0],1.1)
assert np.isclose(adverse.nav_end.iloc[0],.1)
assert flat.long_n.iloc[0]==adverse.long_n.iloc[0]==1

# Reject the short without selecting a replacement, and reduce long equally.
unavailable=b.copy()
unavailable.loc["B","availability_u"]=0
x,_=simulate([unavailable],dict(gross,unavailable_short_fraction=.1),config)
assert x.short_blocked_n.iloc[0]==1
assert x.long_gross_weight.iloc[0]==x.short_gross_weight.iloc[0]==0
assert x.nav_end.iloc[0]==1

capacity=b.copy()
capacity.loc["B","PV_AVG_DOLLAR_VOLUME_21D"]=20
x,_=simulate([capacity],dict(gross,short_adv_fraction_cap=.01),config)
assert np.isclose(x.short_gross_weight.iloc[0],.2)
assert np.isclose(x.long_gross_weight.iloc[0],.2)
assert np.isclose(x.nav_end.iloc[0],1.04)

formation=pd.concat([b.iloc[[0]]]*100,ignore_index=True)
formation["security_id"]=[f"S{i}" for i in range(100)]
formation["ticker"]=formation.security_id
formation["eligible"]=True
formation["entry_price"]=10.
formation["entry_contract_ok"]=True
formation["test_signal"]=np.arange(100)
first=formation_books(formation,"test_signal",config)[0]
formation["forward_return"]=np.nan
formation["return_observed"]=False
second=formation_books(formation,"test_signal",config)[0]
assert len(first)==len(second)==40
pd.testing.assert_series_equal(first.requested_weight,second.requested_weight)

monthly=pd.read_csv(OUT/"monthly_returns.csv",parse_dates=["signal_date","entry_date","exit_date"])
assert (monthly.signal_date<monthly.entry_date).all()
assert (monthly.entry_date<monthly.exit_date).all()
expected=monthly.nav_start-monthly.entry_trading_cost+monthly.gross_pnl-monthly.borrow_cost-monthly.final_liquidation_cost
np.testing.assert_allclose(monthly.nav_end,expected,atol=1e-6,rtol=1e-12)
np.testing.assert_allclose(monthly.long_pnl+monthly.short_pnl,monthly.gross_pnl,atol=1e-6,rtol=1e-12)
np.testing.assert_allclose(monthly.long_gross_weight,monthly.short_gross_weight,atol=1e-12)
assert (monthly.borrow_cost>=0).all()
assert (monthly.entry_trading_cost>=0).all()
assert not monthly.duplicated(["signal","scenario","missing_policy","signal_date"]).any()
for _,g in monthly.groupby(["signal","scenario","missing_policy"],sort=False):
    assert len(g)==235
    np.testing.assert_allclose(g.nav_start.iloc[1:],g.nav_end.iloc[:-1],atol=1e-6,rtol=1e-12)
    np.testing.assert_allclose((1+g["return"]).cumprod(),g.nav_end/g.nav_start.iloc[0],atol=1e-12)
    assert (g.final_liquidation_cost.iloc[:-1]==0).all()

receipt={"status":"passed","synthetic_cases":["gross return","entry and final trading costs",
    "borrow accrual","drifted turnover","unknown exits","blocked short and balanced scaling","short quantity capacity",
    "formation independent of future outcome availability"],
    "published_month_rows":len(monthly),"portfolio_paths":monthly.groupby(["signal","scenario","missing_policy"]).ngroups,
    "checks":"timing, self-financing accounting, leg P&L, neutral target, nonnegative costs, NAV continuity, compounding"}
(OUT/"validation.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
