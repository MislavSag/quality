"""Check project migration without rerunning or relabeling historical returns."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    archived=ROOT/'data/reference/quality_rename_20261112'
    before=json.loads((archived/'numeric_outputs_before.json').read_text())
    changed=[name for name,digest in before.items() if not (ROOT/name).exists() or sha(ROOT/name)!=digest]
    assert not changed,changed
    import analyze_quality as canonical
    import analyze_profitability as legacy
    import build_quality_profitability_bundle as bundle
    import build_profitability_bundle as legacy_bundle
    assert canonical.SIGNALS==legacy.SIGNALS and canonical.cosine_similarity is legacy.cosine_similarity
    assert bundle.build_scores is legacy_bundle.build_scores
    assert json.loads((ROOT/'config/quality_profitability_bundle.json').read_text())['components']==json.loads((ROOT/'config/profitability_bundle.json').read_text())['components']
    assert json.loads((ROOT/'config/monthly_long_short.json').read_text())['availability_seed']=='profitability-monthly-v1'
    scripts=list((ROOT/'scripts').glob('*.py'))
    for p in scripts:
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    # Import runtime modules only; some historical validation scripts execute
    # their checks at module scope and are intentionally not imported here.
    for name in ['add_cosine_ic','backtest_monthly_long_short','compare_public_rmw','analyze_regimes',
                 'analyze_factor_portfolio','analyze_joint_fmp','build_joint_fmp_sides',
                 'build_interactive_report','build_joint_fmp_report','build_quality_literature_report']:
        importlib.import_module(name)
    pages=[ROOT/'results/factor_portfolio/report.html',*list((ROOT/'results/joint_fmp').glob('*.html')),ROOT/'results/quality_literature/report.html']
    for p in pages:
        text=p.read_text(encoding='utf-8')
        title=re.search(r'<title>(.*?)</title>',text,re.S).group(1)
        assert 'Quality' in title,(p,title)
        assert '\ufffd' not in text
        for target in re.findall(r'<a[^>]+href="([^"]+)"',text):
            if target.startswith(('http:','https:','#','data:')):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
    for p in [ROOT/'results/factor_portfolio/report.html',*list((ROOT/'results/joint_fmp').glob('*.html'))]:
        t=p.read_text(encoding='utf-8')
        assert 'qualityReportSnapshot' in t and 'profitabilityReportSnapshot' in t
        assert 'link.download=`quality_' in t
    registry=json.loads((ROOT/'data/reference/quality_literature.json').read_text(encoding='utf-8'))
    assert len(registry['sources'])==16
    assert len({s['id'] for s in registry['sources']})==16
    assert all(s['url'].startswith('https://') and s['review_basis'] for s in registry['sources'])
    project=json.loads((ROOT/'config/project.json').read_text(encoding='utf-8'))
    alias=Path(project['canonical_path'])
    assert alias.resolve()==ROOT
    assert Path(project['physical_storage']).resolve()==ROOT
    assert not alias.is_junction()
    assert (alias/'quality.code-workspace').exists()
    assert sha(alias/'scripts/analyze_quality.py')==sha(ROOT/'scripts/analyze_quality.py')
    metadata=Path(project['git_metadata_storage'])
    assert metadata==ROOT/'.git' and metadata.is_dir() and not metadata.is_junction()
    for removed in project.get('removed_legacy_paths',[]):
        assert not os.path.lexists(removed),removed
    for pending in project.get('pending_empty_directories',[]):
        pending=Path(pending)
        assert pending.is_dir() and not pending.is_junction() and not any(pending.iterdir()),pending
    for old in project['legacy_paths']:
        old=Path(old)
        assert sha(old/'scripts/analyze_quality.py')==sha(ROOT/'scripts/analyze_quality.py')
        assert (old/'data').resolve()==(ROOT/'data').resolve()
        assert (old/'results/joint_fmp').resolve()==(ROOT/'results/joint_fmp').resolve()
    assert bundle.OUT.resolve()==(ROOT/'results/profitability_bundle').resolve()
    receipt=dict(status='passed',unchanged_numeric_files=len(before),canonical_python_files=len(scripts),
        branded_html_pages=len(pages),literature_sources=16,compatible_legacy_entry_points=True,
        quality_path=str(alias),physical_storage=str(ROOT),
        path_mode=project['path_mode'],legacy_paths=project['legacy_paths'],
        removed_legacy_paths=project.get('removed_legacy_paths',[]),git_metadata_storage=str(metadata),
        pending_empty_directories=project.get('pending_empty_directories',[]),
        source_archive_sha256=sha(archived/'original_project_sources.zip'),
        checks=['all historical CSV and Parquet byte hashes unchanged','old and new entry points share implementation',
                'signal definitions and seeded short availability unchanged','runtime imports and Python syntax',
                'Quality report titles, download filenames and legacy browser API alias',
                'local report links resolve','literature records have primary URLs and declared review scope'])
    (ROOT/'results/project_rename_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    changed_sources={str(p.relative_to(ROOT)):sha(p) for p in [*scripts,*list((ROOT/'config').glob('*.json')),ROOT/'README.md',ROOT/'quality.code-workspace',*list((ROOT/'scripts/report_assets').glob('*'))] if p.is_file()}
    manifest=dict(date='2026-11-12',project=json.loads((ROOT/'config/project.json').read_text()),
        scope='project identity and navigation; existing scientific experiments retain profitability scope',
        historical_sources=str(archived/'original_project_sources.zip'),current_source_sha256=changed_sources,
        original_numeric_sha256=before)
    (ROOT/'results/project_rename_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
