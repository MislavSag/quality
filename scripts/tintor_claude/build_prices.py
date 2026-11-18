"""Dnevni panel dionica iz ZSE cjenika (svi uvrsteni papiri po danu, i bez prometa).

Izlaz: data/tintor_claude/prices.parquet
  date, isin, symbol, root, security_type, segment, model, listed (uvijek True),
  close, vwap, turnover (EUR; HRK / 7,5345 do 2022.), volume, num_trades,
  adj (kumulativni faktor podjela nakon datuma), close_adj, vwap_adj, volume_adj
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
CFG = json.loads((ROOT / "config" / "tintor_claude.json").read_text(encoding="utf-8"))
HRK_PER_EUR = 7.5345


def main() -> None:
    d = pd.read_parquet(DATA / "price_list_raw.parquet")
    e = d[(d.security_class == "EQTY") & d.model.isin(["CT", "CTLL", "AUCT"])].copy()
    for c in ["close_price", "vwap_price", "volume", "turnover"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["date"] = pd.to_datetime(e.date)
    fx = np.where(e.price_currency.eq("HRK"), HRK_PER_EUR, 1.0)
    e["close"] = e.close_price / fx
    e["vwap"] = e.vwap_price / fx
    e["turnover"] = e.turnover / np.where(e.turnover_currency.eq("HRK"), HRK_PER_EUR, 1.0)
    e["num_trades"] = e.num_trades.fillna(0).astype(int)
    e["root"] = e.symbol.str.split("-").str[0]
    e = e.sort_values(["isin", "date"]).drop_duplicates(["isin", "date"])
    # korporativne akcije: faktor f na datum D znaci 1 stara dionica = f novih od D
    e["adj"] = 1.0
    for s in CFG["corporate_actions"]:
        m = (e["isin"] == s["isin"]) & (e.date < pd.Timestamp(s["date"]))
        e.loc[m, "adj"] *= s["factor"]
    e["close_adj"] = e.close / e.adj
    e["vwap_adj"] = e.vwap / e.adj
    e["volume_adj"] = e.volume * e.adj
    keep = ["date", "isin", "symbol", "root", "security_type", "segment", "model", "close", "vwap", "turnover",
            "volume", "num_trades", "adj", "close_adj", "vwap_adj", "volume_adj", "trade_date"]
    e[keep].to_parquet(DATA / "prices.parquet", index=False)
    print("prices", len(e), e.date.min().date(), e.date.max().date(), "isins", e["isin"].nunique())


if __name__ == "__main__":
    main()
