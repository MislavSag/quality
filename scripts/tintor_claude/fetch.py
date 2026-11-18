"""Preuzimanje javnih ZSE podataka za Tintor/ZSE backtest.

Podnaredbe:
  prices   dnevni cjenik (price-list) po datumu, rest.zse.hr, od 2010-01-01
  master   popis vrijednosnica (uvrstenja/delistiranja) i povijest indeksa
  meta     nadopuna EHO metapodataka financijskih izvjestaja iznad zadnjeg ID-a
  reports  XLS/XLSX datoteke financijskih izvjestaja s eho.zse.hr
  combine  spajanje cjenika i postojecih EHO metapodataka u parquet (prije meta/reports)

Sve je idempotentno: postojece datoteke se preskacu.
"""
from __future__ import annotations

import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"
REST = "https://rest.zse.hr/web/Bvt9fe2peQ7pwpyYqODM"
EHO = "https://eho.zse.hr"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-client"}
LEGACY_META = Path("C:/Users/Mislav/projects_r/zsefundamentals/data/files")


_local = threading.local()


def session() -> requests.Session:
    # trajna veza po dretvi: uspostava nove veze iz Pythona traje ~6 s
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def get(url: str, tries: int = 4, timeout: int = 60) -> requests.Response | None:
    for i in range(tries):
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def prices(end: str) -> None:
    out = DATA / "raw" / "price_list"
    out.mkdir(parents=True, exist_ok=True)
    d, last = date(2010, 1, 1), date.fromisoformat(end)
    days = []
    while d <= last:
        if d.weekday() < 5 and not (out / f"{d}.json.gz").exists():
            days.append(d)
        d += timedelta(days=1)

    def one(day: date) -> None:
        r = get(f"{REST}/price-list/XZAG/{day}/json")
        if r is None or r.status_code != 200:
            print("FAIL", day, flush=True)
            return
        j = r.json()
        slim = {"date": j.get("date"), "securities": j.get("securities", [])}
        with gzip.open(out / f"{day}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False)
        time.sleep(0.15)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, days))
    print("prices done", len(days))


def master(end: str) -> None:
    out = DATA / "raw"
    out.mkdir(parents=True, exist_ok=True)
    r = get(f"{REST}/securities/XZAG/json")
    (out / "securities_master.json").write_text(r.text, encoding="utf-8")
    for name, isin in {"CROBEX": "HRZB00ICBEX6", "CROBEXtr": "HRZB00ICBTR6"}.items():
        rows = []
        for y in range(2010, int(end[:4]) + 1):
            r = get(f"{REST}/index-history/XZAG/{isin}/{y}-01-01/{min(end, f'{y}-12-31')}/json")
            if r is not None and r.status_code == 200:
                rows += r.json().get("history", [])
            else:
                print("index fail", name, y, r.text[:100] if r is not None else None)
        pd.DataFrame(rows).to_csv(out / f"index_{name}.csv", index=False)
        print(name, len(rows))


def parse_meta_page(html: str) -> dict | None:
    from io import StringIO
    import re

    try:
        tabs = pd.read_html(StringIO(html))
    except ValueError:
        return None
    if not tabs:
        return None
    t = tabs[0]
    rec = {str(k).strip(): (None if pd.isna(v) else str(v).strip()) for k, v in zip(t.iloc[:, 0], t.iloc[:, 1])}
    m = re.findall(r'href="([^"]*fileadmin[^"]*)"', html)
    rec["link"] = m[0] if m else None
    return rec


def meta(max_miss: int = 2500) -> None:
    out = DATA / "raw" / "eho_meta_new"
    out.mkdir(parents=True, exist_ok=True)
    start = max([int(p.stem) for p in LEGACY_META.glob("*.csv")] + [int(p.stem) for p in out.glob("*.json")]) + 1
    miss, i = 0, start
    while miss < max_miss:
        f = out / f"{i}.json"
        if f.exists():
            miss, i = 0, i + 1
            continue
        r = get(f"{EHO}/financijski-izvjestaji/view/{i}", tries=2, timeout=30)
        rec = parse_meta_page(r.text) if (r is not None and r.status_code == 200) else None
        if rec and rec.get("link"):
            rec["eho_id"] = str(i)
            f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            miss = 0
        else:
            miss += 1
        i += 1
        time.sleep(0.1)
    print("meta done; last id tried", i)


def all_meta() -> pd.DataFrame:
    import unicodedata, re

    def norm(c: str) -> str:
        c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z0-9]+", "_", c).strip("_")

    old = pd.read_parquet(DATA / "eho_meta_raw.parquet")
    new_dir = DATA / "raw" / "eho_meta_new"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in new_dir.glob("*.json")] if new_dir.exists() else []
    if recs:
        new = pd.DataFrame(recs)
        new.columns = [norm(c) for c in new.columns]
        ren = {c: o for c in new.columns for o in old.columns if c != o and (o.startswith(c) or c.startswith(o))}
        new = new.rename(columns=ren)
        old = pd.concat([old, new[[c for c in new.columns if c in old.columns]]], ignore_index=True)
    return old


def reports() -> None:
    out = DATA / "raw" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    m = all_meta()
    m = m[m.tip_dokumenta.isin(["XLS", "XLSX"]) & m.link.notna()]
    # engleska verzija se preuzima samo ako za isti izvjestaj nema hrvatske
    key = ["izdavatelj", "za_godinu", "razdoblje", "status_revizije", "status_konsolidacije"]
    hr = m[m.na_jeziku == "Hrvatski"]
    en = m[m.na_jeziku != "Hrvatski"].merge(hr[key].drop_duplicates(), on=key, how="left", indicator=True)
    m = pd.concat([hr, en[en._merge == "left_only"].drop(columns="_merge")], ignore_index=True)
    todo = [(r.eho_id, r.link) for r in m.itertuples() if not (out / f"{r.eho_id}{Path(r.link).suffix.lower()}").exists()]
    if "rev" in sys.argv:  # drugi proces moze ici od kraja popisa
        todo = todo[::-1]
    print("reports to download", len(todo), flush=True)

    def one(a) -> None:
        eid, link = a
        if (out / f"{eid}{Path(link).suffix.lower()}").exists():
            return
        r = get(EHO + link)
        if r is None or r.status_code != 200 or len(r.content) < 2000:
            print("FAIL", eid, link, flush=True)
            return
        (out / f"{eid}{Path(link).suffix.lower()}").write_bytes(r.content)
        time.sleep(0.1)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    print("reports done")


def combine() -> None:
    """Spoji preuzete cjenike i postojece EHO metapodatke u parquet datoteke."""
    rows = []
    for f in sorted((DATA / "raw" / "price_list").glob("*.json.gz")):
        rows += json.load(gzip.open(f, "rt", encoding="utf-8"))["securities"]
    pd.DataFrame(rows).to_parquet(DATA / "price_list_raw.parquet")
    old = pd.concat([pd.read_csv(f, dtype=str).assign(eho_id=f.stem) for f in LEGACY_META.glob("*.csv")],
                    ignore_index=True)
    old.to_parquet(DATA / "eho_meta_raw.parquet")
    print("price rows", len(rows), "legacy meta", len(old))


if __name__ == "__main__":
    cmd = sys.argv[1]
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-11-16"
    {"prices": lambda: prices(end), "master": lambda: master(end), "meta": meta, "reports": reports,
     "combine": combine}[cmd]()
