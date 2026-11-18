"""Izvoz knjiga (NAV, transakcije) u format istrazivackih testova aplikacije ZSE Analitika.

Jedna neprekinuta simulacija od 2011-05 do kraja kalendara aplikacije; NAV se svodi na
100.000 EUR na bazni datum aplikacije, a transakcije prije prvog dana kalendara se ne izvoze.
Tri razine naknade (0,25 / 0,50 / 1,00 % po strani) uz nepromijenjeno proklizavanje.

  python -P export_app.py <zse-research> <dataset-version-dir> <run-id>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

STRATEGIES = ["kvaliteta", "kvaliteta+vrednovanje", "altman", "kralicek"]
KEYS = {"kvaliteta": "tintor-kvaliteta", "kvaliteta+vrednovanje": "tintor-kvaliteta-vrednovanje",
        "altman": "tintor-altman-z", "kralicek": "tintor-kralicek-df"}
START = "2011-05-01"
CAPITAL = 100_000.0
STALE_DAYS = 30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(zse: Path, base: Path, run_id: str) -> None:
    public = json.loads((base / "public.json").read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(public["dates"])
    base_date = pd.Timestamp(public["baseDate"])
    out = zse / "data" / "derived" / "tintor_claude" / run_id
    out.mkdir(parents=True, exist_ok=False)
    mkt = bt.Market()
    assert mkt.cal[(mkt.cal >= dates[0]) & (mkt.cal <= dates[-1])].equals(dates), "kalendar aplikacije se razlikuje"
    prices = pd.read_parquet(bt.DATA / "prices.parquet", columns=["date", "isin", "trade_date"])
    last = prices[prices.date == dates[-1]].set_index("isin").trade_date
    sig = mkt.sig[mkt.sig.signal_date <= dates[-1]]
    n = bt.CFG["portfolio"]["max_positions"]
    slip = bt.CFG["execution"]["slippage"]
    files = {}
    for v in STRATEGIES:
        targets, _ = bt.build_targets(sig, v, n)
        for cost in (25, 50, 100):
            res = bt.run(mkt, targets, START, dates[-1].strftime("%Y-%m-%d"), commission=cost / 1e4)
            nav = res["nav"]
            k = CAPITAL / nav.nav.loc[base_date]
            d = nav.loc[dates]
            daily = pd.DataFrame({"date": dates, "nav_eur": (d.nav * k).to_numpy(), "cash_eur": (d.cash * k).to_numpy(),
                                  "n_positions": d.n_pos.to_numpy()})
            f = res["fills"]
            f = f[f.exec_date >= dates[0]].copy()
            sign = np.where(f.side.eq("B"), 1.0, -1.0)
            gross_notional = f.shares_adj * f.vwap * k
            trades = pd.DataFrame({
                "date": f.exec_date.to_numpy(), "isin": f["isin"].to_numpy(), "signal_date": f.signal_date.to_numpy(),
                "units": (f.shares_adj * sign).to_numpy(), "notional_eur": gross_notional.to_numpy(),
                "commission_eur": (f.commission * k).to_numpy(), "impact_eur": (gross_notional * slip).to_numpy()})
            trades["cost_eur"] = trades.commission_eur + trades.impact_eur
            sk = res["skipped"]
            sk = sk[sk.signal_date >= dates[0]] if len(sk) else sk
            ordered = trades.notional_eur.sum() + (sk.unfilled_value.sum() * k if len(sk) else 0.0)
            end_px = mkt.close.loc[dates[-1]]
            stale = sum(q * end_px.get(i, 0.0) for i, q in res["positions"].items()
                        if (dates[-1] - pd.Timestamp(last.get(i, dates[-1]))).days > STALE_DAYS) * k
            years = (dates[-1] - base_date).days / 365.25
            result = {
                "annual_turnover": float(min(trades.notional_eur[trades.units > 0].sum(),
                                             trades.notional_eur[trades.units < 0].sum()) / daily.nav_eur.mean() / years),
                "mean_exposure": float(1 - (daily.cash_eur / daily.nav_eur).mean()),
                "terminal_provisional_eur": float(stale),
                "terminal_provisional_share": float(stale / daily.nav_eur.iloc[-1]),
                "unfilled_share": float(sk.unfilled_value.sum() * k / ordered) if len(sk) and ordered else 0.0,
                "fills": int(len(trades)), "cancelled_orders": int(len(sk)),
            }
            checks = {
                "calendar_matches_app": bool(pd.DatetimeIndex(daily.date).equals(dates)),
                "nav_finite_positive": bool(np.isfinite(daily.nav_eur).all() and (daily.nav_eur > 0).all()),
                "trades_inside_calendar": bool(trades.date.isin(dates).all()),
                "costs_reconcile": bool(np.allclose(trades.commission_eur + trades.impact_eur, trades.cost_eur)),
                "signals_precede_execution": bool((trades.signal_date < trades.date).all()),
            }
            stem = f"{KEYS[v]}_{cost}bp"
            daily.to_parquet(out / f"{stem}_daily.parquet", index=False)
            trades.to_parquet(out / f"{stem}_trades.parquet", index=False)
            (out / f"{stem}_complete.json").write_text(json.dumps({"checks": checks, "result": result}, indent=1),
                                                       encoding="utf-8")
            for sfx in ("_daily.parquet", "_trades.parquet", "_complete.json"):
                files[stem + sfx] = sha(out / (stem + sfx))
            print(stem, "CAGR %.4f" % ((daily.nav_eur.iloc[-1] / CAPITAL) ** (1 / years) - 1), result["annual_turnover"].__round__(3),
                  all(checks.values()))
    here = Path(__file__).resolve().parent
    manifest = {
        "run": run_id, "source_project": "quality/scripts/tintor_claude", "base_dataset": base.name,
        "simulation_start": START, "base_date": public["baseDate"], "test_start": public["dates"][0],
        "test_end": public["dates"][-1], "config": bt.CFG, "strategies": KEYS,
        "code_sha256": {p.name: sha(p) for p in sorted(here.glob("*.py"))},
        "input_sha256": {p.name: sha(p) for p in [bt.DATA / "signals.parquet", bt.DATA / "prices.parquet",
                                                  bt.DATA / "dividends.parquet"]},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3])
