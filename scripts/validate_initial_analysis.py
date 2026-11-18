"""Independent checks of source endpoints, timing, and selected IC calculations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
from analyze_quality import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "initial_analysis"
panel = pd.read_parquet(ROOT / "data" / "derived" / "monthly_panel.parquet")
calendar = pd.read_csv(OUT / "formation_calendar.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
monthly = pd.read_csv(OUT / "monthly_diagnostics.csv", parse_dates=["signal_date"])
assert (calendar.signal_date < calendar.entry_date).all()
assert (calendar.entry_date < calendar.exit_date).all()
assert not panel.duplicated(["security_id", "signal_date"]).any()
eligible = panel[panel.eligible]
assert (eligible.PV_CLOSE_RAW >= 5).all()
assert (eligible.PV_AVG_DOLLAR_VOLUME_21D >= 1e6).all()
assert (eligible.FUND_MARKET_CAP_USD_COMPAT >= 1e8).all()
assert (eligible.FUND_STATEMENT_AVAILABLE_DATE <= eligible.signal_date).all()

source = ds.dataset("H:/strategies/characteristics/price_volume_characteristics.parquet")
endpoints_checked = 0
for i in [0, len(calendar)//2, len(calendar)-1]:
    date = calendar.signal_date.iloc[i]
    candidates = eligible[(eligible.signal_date == date) & eligible.return_observed].sort_values("security_id")
    for _, row in candidates.iloc[[0, len(candidates)//2, -1]].iterrows():
        values = source.to_table(columns=["trade_date", "PV_CLOSE"], filter=(
            (ds.field("security_id") == row.security_id) & ds.field("trade_date").isin(
                [row.entry_date.date(), row.exit_date.date()]))).to_pandas(date_as_object=False).set_index("trade_date")
        expected = values.loc[row.exit_date, "PV_CLOSE"] / values.loc[row.entry_date, "PV_CLOSE"] - 1
        assert np.isclose(expected, row.forward_return, atol=1e-12, rtol=1e-12)
        endpoints_checked += 1

ic_checked = 0
cosine_checked = 0
for i in [0, len(calendar)//2, len(calendar)-1]:
    date = calendar.signal_date.iloc[i]
    for signal in ["fcf_assets_ttm", "gp_assets_ttm", "qmj_like_fy"]:
        observed = eligible[(eligible.signal_date == date) & eligible.return_observed].dropna(subset=[signal])
        expected = spearmanr(observed[signal], observed.forward_return).statistic
        saved = monthly[(monthly["sample"] == "own") & (monthly.signal == signal) & (monthly.signal_date == date)].ic.item()
        assert np.isclose(expected, saved, atol=1e-12, rtol=1e-12)
        ic_checked += 1
        formation = eligible[(eligible.signal_date == date) & np.isfinite(eligible[signal])]
        mask = formation.return_observed & np.isfinite(formation.forward_return)
        x = formation.loc[mask, signal].to_numpy()
        y = formation.loc[mask, "forward_return"].to_numpy()
        ranks = rankdata(formation[signal], method="average")
        centered_rank = (ranks-ranks.mean())[mask.to_numpy()]
        row = monthly[(monthly["sample"] == "own") & (monthly.signal == signal) & (monthly.signal_date == date)].iloc[0]
        assert np.isclose(1-cosine(x,y), row.ic_cosine_raw, atol=1e-12, rtol=1e-12)
        assert np.isclose(1-cosine(centered_rank,y), row.ic_cosine_rank_centered, atol=1e-12, rtol=1e-12)
        cosine_checked += 2

# Mathematical edge cases distinguish a cosine from a centered correlation.
x, y = np.array([1., 2., 4.]), np.array([-.1, .2, .3])
assert np.isclose(cosine_similarity(x,y), cosine_similarity(x*1e200,y*1e200))
assert np.isclose(cosine_similarity(x,y), -cosine_similarity(-x,y))
assert np.isnan(cosine_similarity([0,0,0],y))
assert np.isnan(cosine_similarity(x,[0,0,0]))
assert not np.isclose(cosine_similarity(x,y), np.corrcoef(x,y)[0,1])
assert np.isclose(cosine_similarity(x-x.mean(),y-y.mean()),np.corrcoef(x,y)[0,1])
assert np.isclose(cosine_similarity(rankdata(x)-2,rankdata(y)-2),spearmanr(x,y).statistic)
assert np.array_equal(monthly.ic_spearman.to_numpy(), monthly.ic.to_numpy(), equal_nan=True)

paired = monthly[monthly["sample"].str.startswith("pair_")]
for _, group in paired.groupby(["sample", "signal_date"]):
    assert group.n.nunique() == 1
    assert group.n_observed.nunique() == 1
receipt = {"status": "passed", "rows": len(panel), "eligible_rows": len(eligible),
    "timing_and_key_checks": "all monthly rows", "independent_source_endpoint_checks": endpoints_checked,
    "independent_spearman_checks": ic_checked, "paired_sample_checks": "all reported pairs",
    "independent_cosine_checks": cosine_checked, "cosine_edge_cases": "passed",
    "scope": "validates calculations and sampled lineage, not source accuracy or first-release accounting vintages"}
(OUT / "validation.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
