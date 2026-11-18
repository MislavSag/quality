"""Add cosine metrics without recalculating prices, quintiles or regressions."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

import analyze_quality as analysis


def main():
    out = analysis.OUT
    manifest = json.loads((out/"manifest.json").read_text())
    inputs = [analysis.fingerprint(p) for p in [analysis.PRICES, analysis.FUND, analysis.EXTFILE]]
    if inputs != manifest["inputs"]:
        raise RuntimeError("Source inputs changed; rebuild the monthly panel before updating IC")
    old_monthly = pd.read_csv(out/"monthly_diagnostics.csv", parse_dates=["signal_date"])
    old_summary = pd.read_csv(out/"summary.csv")
    backup = out/"before_cosine"
    backup.mkdir(exist_ok=True)
    for name in ["monthly_diagnostics.csv", "summary.csv", "manifest.json"]:
        if not (backup/name).exists():
            shutil.copy2(out/name, backup/name)
    columns = ["signal_date", "eligible", "return_observed", "forward_return", *analysis.SIGNALS]
    panel = pd.read_parquet(analysis.CACHE/"monthly_panel.parquet", columns=columns)
    eligible = panel[panel.eligible]
    keys = ["sample", "signal_date", "signal"]
    records = []
    for label, rows in old_monthly.groupby("sample", sort=False):
        data = eligible
        if label == "common_core":
            data = data[np.isfinite(data[analysis.CORE]).all(axis=1)]
        elif label.startswith("pair_"):
            signal = label.removeprefix("pair_")
            data = data[np.isfinite(data[[signal, "gp_assets_ttm"]]).all(axis=1)]
        else:
            assert label == "own"
        signals = rows.signal.unique()
        wanted = set(zip(rows.signal_date, rows.signal))
        for month, group in data.groupby("signal_date", sort=True):
            for signal in signals:
                if (month, signal) not in wanted:
                    continue
                sample = group[np.isfinite(group[signal])]
                records.append({"sample": label, "signal_date": month, "signal": signal,
                    "check_n": len(sample), "check_n_observed": int(sample.return_observed.sum()),
                    **analysis.cosine_metrics(sample, signal)})
        analysis.log(f"Cosine metrics: {label}")
    added = pd.DataFrame(records)
    columns = [c for c in old_monthly if c not in ["ic_spearman", "ic_cosine_raw", "ic_cosine_rank_centered"]]
    monthly = old_monthly[columns].merge(added, on=keys, how="left", validate="one_to_one")
    assert len(monthly) == len(old_monthly)
    assert (monthly.n == monthly.check_n).all()
    assert (monthly.n_observed == monthly.check_n_observed).all()
    monthly = monthly.drop(columns=["check_n", "check_n_observed"])
    monthly["ic_spearman"] = monthly.ic
    pd.testing.assert_frame_equal(monthly[columns], old_monthly[columns], check_exact=True)
    for name in ["ic_cosine_raw", "ic_cosine_rank_centered"]:
        assert monthly[name].dropna().between(-1, 1).all()
    summary = analysis.summarize(monthly)
    original_columns = [c for c in old_summary if c in summary]
    pd.testing.assert_frame_equal(summary[original_columns], old_summary[original_columns],
                                  check_exact=False, atol=1e-12, rtol=1e-12)
    monthly.to_csv(out/"monthly_diagnostics.csv", index=False)
    summary.to_csv(out/"summary.csv", index=False)
    summary[["sample", *analysis.ic_columns_for_export()]].to_csv(out/"ic_comparison.csv", index=False)
    manifest["ic_update"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest(),
        "source_fingerprints_unchanged": True,
        "original_monthly_metrics_unchanged": True,
        "original_summary_metrics_unchanged_within": 1e-12,
        "definitions": {
            "ic_spearman": "Spearman(signal, forward_return); alias ic preserved",
            "ic_cosine_raw": "dot(signal, raw_forward_return)/(norm(signal)*norm(raw_forward_return)); no centering, ranking or risk adjustment",
            "ic_cosine_rank_centered": "cosine(rank(signal)-mean(rank(signal)), raw_forward_return); ranks centered on formation set before missing outcomes",
            "sample": "same finite signal and observed outcome pairs as Spearman; at least 100 observed; zero norm yields missing",
            "aggregation": "equal-weight monthly mean; HAC t statistic with three lags",
            "attribution": "No claim of an exact Paleologo specification or risk whitening"}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    calendar = pd.read_csv(out/"formation_calendar.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    regressions = pd.read_csv(out/"regression_summary.csv")
    analysis.make_report(panel, monthly, summary, regressions, calendar, manifest)
    analysis.log(summary[(summary["sample"]=="own") & (summary.period=="full")][[
        "signal", "mean_ic_spearman", "mean_ic_cosine_raw", "mean_ic_cosine_rank_centered"]].to_string(index=False))


if __name__ == "__main__":
    main()
