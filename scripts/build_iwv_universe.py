"""Read repaired ETF snapshots as a dated, exact-SID formation membership mask."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import build_quality_dimensions as dimensions

ROOT = dimensions.ROOT
OUT = ROOT / 'results/combined_quality'
CFG = json.loads((ROOT / 'config/combined_quality.json').read_text(encoding='utf-8'))['universe']
SOURCE = Path(CFG['holdings_path'])
FLAGS = OUT / 'iwv_monthly_membership.parquet'
SELECTED = OUT / 'iwv_selected_holdings.parquet'
CALENDAR = OUT / 'iwv_snapshot_calendar.csv'
COVERAGE = OUT / 'iwv_universe_coverage.csv'
RECEIPT = OUT / 'iwv_universe_manifest.json'


def contract():
    return dict(config=CFG, source_sha256=dimensions.sha(SOURCE), code_sha256=dimensions.sha(__file__),
        monthly_sha256=dimensions.sha(dimensions.MONTHLY))


def load_membership():
    expected = contract()
    if RECEIPT.exists():
        prior = json.loads(RECEIPT.read_text())
        if prior['contract'] == expected and all((OUT / name).exists() and dimensions.sha(OUT / name) == digest
                for name, digest in prior['output_sha256'].items()):
            print('Reusing verified dated IWV membership mask', flush=True)
            return pd.read_parquet(FLAGS)
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_parquet(dimensions.MONTHLY, columns=dimensions.KEYS + ['eligible'])
    monthly['signal_date'] = monthly.signal_date.astype('datetime64[ns]')
    dates = monthly[['signal_date']].drop_duplicates().sort_values('signal_date')
    # Find actual publication/target dates first, then read only snapshots used.
    scan = pl.scan_parquet(SOURCE).filter(pl.col('etf') == CFG['etf'])
    snapshots = scan.select(['target_snapshot_date', 'source_snapshot_date', 'is_backfilled']).unique().collect().to_pandas()
    snapshots = snapshots.sort_values('target_snapshot_date')
    assert not snapshots.target_snapshot_date.duplicated().any(), 'Ambiguous source snapshot lineage'
    assert (snapshots.source_snapshot_date <= snapshots.target_snapshot_date).all()
    calendar = pd.merge_asof(dates, snapshots, left_on='signal_date', right_on='target_snapshot_date', direction='backward')
    known = calendar.target_snapshot_date.notna()
    assert (calendar.loc[known, 'target_snapshot_date'] <= calendar.loc[known, 'signal_date']).all()
    calendar['source_age_days'] = (calendar.signal_date - calendar.source_snapshot_date).dt.days
    used = calendar.loc[known, 'target_snapshot_date'].tolist()
    columns = ['etf', 'target_snapshot_date', 'source_snapshot_date', 'is_backfilled', 'sid', 'symbol',
               'identity_status', 'identity_usable', 'weight']
    holdings = pd.read_parquet(SOURCE, columns=columns,
        filters=[('etf', '==', CFG['etf']), ('target_snapshot_date', 'in', used)])
    assert not holdings.duplicated(['target_snapshot_date', 'sid']).any()
    assert holdings.identity_usable.notna().all()
    flags = monthly.merge(calendar[['signal_date', 'target_snapshot_date']], on='signal_date', validate='many_to_one')
    members = holdings[['target_snapshot_date', 'sid', 'identity_usable']].rename(columns={'sid': 'security_id'})
    flags = flags.merge(members, on=['target_snapshot_date', 'security_id'], how='left', validate='many_to_one')
    flags['iwv_member'] = flags.identity_usable.eq(True)
    flags['iwv_eligible'] = flags.eligible & flags.iwv_member
    flags['unresolved_exact_sid'] = flags.identity_usable.eq(False)
    flags = flags.drop(columns='identity_usable')
    records = []
    grouped = {date: g for date, g in holdings.groupby('target_snapshot_date')}
    for row in calendar.itertuples():
        group = flags[flags.signal_date.eq(row.signal_date)]
        snapshot = grouped.get(row.target_snapshot_date)
        records.append(dict(signal_date=row.signal_date, target_snapshot_date=row.target_snapshot_date,
            source_snapshot_date=row.source_snapshot_date, source_age_days=row.source_age_days,
            snapshot_available=pd.notna(row.target_snapshot_date), is_backfilled=bool(row.is_backfilled) if pd.notna(row.target_snapshot_date) else False,
            holdings_n=0 if snapshot is None else len(snapshot),
            holdings_identity_usable_n=0 if snapshot is None else int(snapshot.identity_usable.sum()),
            holdings_unresolved_n=0 if snapshot is None else int((~snapshot.identity_usable).sum()),
            usable_zero_weight_n=0 if snapshot is None else int((snapshot.identity_usable & snapshot.weight.eq(0)).sum()),
            panel_matched_members_n=int(group.iwv_member.sum()),
            usable_members_missing_panel_n=0 if snapshot is None else int(snapshot.identity_usable.sum()) - int(group.iwv_member.sum()),
            broad_eligible_n=int(group.eligible.sum()), iwv_eligible_n=int(group.iwv_eligible.sum()),
            unresolved_exact_sid_in_panel_n=int(group.unresolved_exact_sid.sum())))
    flags.to_parquet(FLAGS, index=False)
    holdings.to_parquet(SELECTED, index=False)
    calendar.to_csv(CALENDAR, index=False)
    coverage = pd.DataFrame(records)
    coverage.to_csv(COVERAGE, index=False)
    receipt = dict(contract=expected, first_snapshot=str(snapshots.target_snapshot_date.min().date()),
        first_available_formation=str(calendar.loc[known, 'signal_date'].min().date()),
        source_snapshots=len(snapshots), selected_snapshots=len(grouped),
        available_formation_months=int(known.sum()), unavailable_formation_months=int((~known).sum()),
        max_selected_source_age_days=int(calendar.source_age_days.max()),
        output_sha256={p.name: dimensions.sha(p) for p in [FLAGS, SELECTED, CALENDAR, COVERAGE]})
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Built IWV mask:', int(known.sum()), 'available months; first formation', receipt['first_available_formation'], flush=True)
    return flags


if __name__ == '__main__':
    load_membership()
