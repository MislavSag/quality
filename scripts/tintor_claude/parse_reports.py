"""Parsiranje TFI obrazaca (XLS/XLSX s eho.zse.hr) u dugi oblik AOP redaka.

Izlaz: data/tintor_claude/aop_rows.parquet
  eho_id, sheet, kind (BS/IS/CF_I/CF_D/OTHER), aop, label, v0..v3 (stupci desno od AOP-a)
i data/tintor_claude/report_headers.parquet (razdoblje iz obrasca, obrazac, pogreske).

Pokretati s `python -P` (datoteke su preuzete s interneta).
"""
from __future__ import annotations

import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tintor_claude"


def norm(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def sheet_kind(name: str) -> str:
    n = norm(name).replace(" ", "").replace("_", "").replace("-", "")
    if n.startswith(("bil", "balance", "bs", "izvjestajofinancijskompolozaju", "ifp")):
        return "BS"
    if n.startswith(("rdg", "rdig", "p&l", "pl", "racundobiti", "isd", "income")):
        return "IS"
    if n.startswith(("nti", "cfi", "ntind", "novcanitokind")):
        return "CF_I"
    if n.startswith(("ntd", "cfd", "ntdir", "novcanitokdir")):
        return "CF_D"
    if n.startswith(("nt", "cf", "novcanitok", "novcanitijek", "cashflow")):
        return "CF_I"
    return "OTHER"


def as_aop(v) -> int | None:
    if isinstance(v, (int, np.integer)) and 0 < v < 1000:
        return int(v)
    if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer() and 0 < v < 1000:
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"\d{1,3}", s) and int(s) > 0:
            return int(s)
    return None


def to_num(v) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("\xa0", "").replace(" ", "")
        if re.fullmatch(r"-?\(?[\d.,]+\)?", s) and any(ch.isdigit() for ch in s):
            neg = s.startswith("(") or s.startswith("-")
            s = s.strip("()-")
            if "," in s and "." in s:
                s = s.replace(".", "").replace(",", ".")
            elif "," in s:
                s = s.replace(",", ".")
            try:
                return -float(s) if neg else float(s)
            except ValueError:
                return np.nan
    return np.nan


def parse_sheet(d: pd.DataFrame) -> list[tuple]:
    """Pronadi AOP stupac (najdulji rastuci niz cijelih brojeva) i vrati retke."""
    a = d.values
    best_c, best_n = None, 0
    for c in range(1, min(a.shape[1], 40)):
        col = [as_aop(v) for v in a[:, c]]
        seq = [x for x in col if x is not None]
        if len(seq) < 8:
            continue
        inc = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1] + 1)
        if inc > best_n:
            best_c, best_n = c, inc
    if best_c is None or best_n < 6:
        return []
    out = []
    for r in range(a.shape[0]):
        aop = as_aop(a[r, best_c])
        if aop is None:
            continue
        labs = [v for v in a[r, :best_c] if isinstance(v, str) and len(v.strip()) > 2]
        if not labs:
            continue
        label = max(labs, key=len).strip()
        vals = []
        for c in range(best_c + 1, a.shape[1]):
            col_has = any(not (isinstance(v, float) and np.isnan(v)) and v is not None for v in a[:, c])
            if col_has:
                vals.append(to_num(a[r, c]))
        vals = (vals + [np.nan] * 4)[:4]
        out.append((aop, label, *vals))
    return out


def header_info(x: pd.ExcelFile) -> dict:
    """Razdoblje i oznake iz lista opcih podataka (najbolji pokusaj)."""
    info = {}
    for s in x.sheet_names:
        n = norm(s)
        if "opc" in n or "general" in n or "podaci" in n:
            try:
                d = x.parse(s, header=None, nrows=70)
            except Exception:
                continue
            dates = []
            texts = []
            for v in d.values.ravel():
                if isinstance(v, pd.Timestamp) or hasattr(v, "year") and hasattr(v, "month") and not isinstance(v, (int, float)):
                    try:
                        dates.append(pd.Timestamp(v))
                    except Exception:
                        pass
                elif isinstance(v, str):
                    texts.append(v)
                    for m in re.findall(r"\b(\d{1,2})\.\s?(\d{1,2})\.\s?(20\d{2})\b", v):
                        try:
                            dates.append(pd.Timestamp(int(m[2]), int(m[1]), int(m[0])))
                        except ValueError:
                            pass
            dates = [t for t in dates if 2005 <= t.year <= 2030]
            if len(dates) >= 2:
                info["period_start"], info["period_end"] = dates[0], dates[1]
            joined = norm(" ".join(texts))
            m = re.search(r"\b(tfi-pod|tfi-ki|tfi-osig|gfi-pod|tfi-zif|tfi-izd)", joined)
            # napomena: obrazac se pouzdanije prepoznaje po stavkama (vidi fundamentals.py)
            info["form_hint"] = m.group(1) if m else None
            break
    return info


def parse_file(path: str) -> tuple[list[tuple], dict]:
    eid = Path(path).stem
    hdr = {"eho_id": eid, "error": None}
    rows = []
    try:
        x = pd.ExcelFile(path)
        hdr["sheets"] = "|".join(x.sheet_names)
        hdr.update(header_info(x))
        for s in x.sheet_names:
            kind = sheet_kind(s)
            if kind == "OTHER":
                continue
            d = x.parse(s, header=None)
            for t in parse_sheet(d):
                rows.append((eid, s, kind, *t))
    except Exception as e:  # ostecena ili nestandardna datoteka
        hdr["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return rows, hdr


def main() -> None:
    files = sorted(str(p) for p in (DATA / "raw" / "reports").glob("*.xls*"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        files = files[:limit]
    rows, hdrs = [], []
    with ProcessPoolExecutor(10) as ex:
        for r, h in ex.map(parse_file, files, chunksize=20):
            rows += r
            hdrs.append(h)
    df = pd.DataFrame(rows, columns=["eho_id", "sheet", "kind", "aop", "label", "v0", "v1", "v2", "v3"])
    df.to_parquet(DATA / "aop_rows.parquet", index=False)
    pd.DataFrame(hdrs).to_parquet(DATA / "report_headers.parquet", index=False)
    h = pd.DataFrame(hdrs)
    print("files", len(files), "rows", len(df), "errors", h.error.notna().sum(), "no rows", len(set(h.eho_id) - set(df.eho_id)))


if __name__ == "__main__":
    main()
