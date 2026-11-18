"""Tintorovi pokazatelji iz dnevne tablice characteristics projekta na datume signala.

Skup dionica i datumi signala su isti kao u us.py (data/tintor_claude/us/panel.parquet, eligible).
Izlaz: data/tintor_claude/us/book_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "tintor_claude" / "us"
BOOK = Path("H:/strategies/characteristics_tintor_book_ttm_daily_us_20261117")
MAX_AGE = 200   # dana od kraja tromjesecja, kao u us.py


def main() -> None:
    p = pd.read_parquet(CACHE / "panel.parquet")
    p = p[p.eligible][["signal_date", "security_id", "ticker", "adt", "mcap", "FUND_EODHD_GIC_SECTOR"]]
    meta = pd.read_csv(BOOK / "tintor_book_ttm_predictor_metadata.csv")
    feats = meta[meta.role.isin(["predictor", "diagnostic"])].feature.tolist()
    qa = ["FUND_AVAILABLE_DATE", "FUND_FISCAL_DATE", "FUND_AVAILABILITY_ESTIMATED", "tintor_ttm_window_valid",
          "tintor_fiscal_age_days", "tintor_gics_industry"]
    dates = sorted(p.signal_date.dt.date.unique())
    x = (pl.scan_parquet(BOOK / "tintor_book_ttm_characteristics.parquet")
         .filter(pl.col("trade_date").is_in(dates))
         .select(["security_id", "trade_date", *qa, *feats]).collect().to_pandas())
    x["signal_date"] = pd.to_datetime(x.trade_date)
    out = p.merge(x.drop(columns="trade_date"), on=["signal_date", "security_id"], how="left")
    out["in_book"] = out.FUND_FISCAL_DATE.notna()
    stale = ~(out.tintor_fiscal_age_days <= MAX_AGE) | ~(out.tintor_ttm_window_valid == True)  # noqa: E712
    out["book_ok"] = out.in_book & ~stale
    out.loc[~out.book_ok, feats] = float("nan")
    out.to_parquet(CACHE / "book_features.parquet")
    c = out.groupby("signal_date").agg(podobnih=("security_id", "size"), u_tablici=("in_book", "sum"), svjezih=("book_ok", "sum"))
    print(c.iloc[::8].to_string())
    print(c.sum().to_string())
    print("avail > signal:", int((pd.to_datetime(out.FUND_AVAILABLE_DATE) > out.signal_date).sum()))


if __name__ == "__main__":
    main()
