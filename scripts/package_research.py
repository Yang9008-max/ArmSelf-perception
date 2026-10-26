"""Package the local research snapshot without changing earlier release manifests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile

try:
    from .artifacts import ROOT, sha256
except ImportError:
    from artifacts import ROOT, sha256


def group_name(relative):
    return relative.parts[0] if len(relative.parts) > 1 else 'overview'


def excluded_reason(relative):
    if any(p in ('__pycache__', '.mpl-cache') for p in relative.parts) or relative.suffix == '.pyc':
        return 'machine cache'
    if 'cache' in relative.parts and relative.suffix == '.npz' and relative.name != 'background-reference.npz':
        return 'regenerable optical-flow cache; frozen background and metadata retained'
    return None


def package(destination, tag, source, support_sources=()):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', tag):
        raise ValueError('Use a plain release tag')
    source, destination = source.resolve(), destination.resolve()
    if not source.is_relative_to((ROOT/'outputs').resolve()) or not source.is_dir():
        raise ValueError('Source must be an existing directory inside outputs')
    if destination.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Keep archives outside outputs')
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'artifacts'/f'{tag}.json'
    if manifest_path.exists() or any(destination.iterdir()):
        raise FileExistsError('Choose a fresh tag and empty archive directory')
    groups, excluded = {}, []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        reason = excluded_reason(relative)
        if reason:
            excluded.append(dict(path=path.relative_to(ROOT).as_posix(), reason=reason))
            continue
        entry = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                     sha256=sha256(path))
        groups.setdefault(group_name(relative), []).append(entry)
    existing = {e['path'] for entries in groups.values() for e in entries}
    for support in support_sources:
        support = support.resolve()
        if not support.is_relative_to((ROOT/'outputs').resolve()) or not support.is_dir():
            raise ValueError('Supporting sources must be directories inside outputs')
        for path in sorted(support.rglob('*')):
            if not path.is_file() or path.relative_to(ROOT).as_posix() in existing:
                continue
            relative = path.relative_to(ROOT)
            reason = excluded_reason(relative)
            if reason:
                excluded.append(dict(path=relative.as_posix(), reason=reason))
                continue
            entry = dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha256(path))
            groups.setdefault('supporting-data', []).append(entry)
            existing.add(entry['path'])
    if not groups:
        raise ValueError('No research files found')
    runtime = [e for entries in groups.values() for e in entries
               if Path(e['path']).suffix not in ('.npz', '.npy')]
    manifest = dict(format='arm-self-perception-artifacts-v1', tag=tag,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=source.relative_to(ROOT).as_posix(),
        prerequisites_manifest='artifacts/manifest.json',
        default_bundles=sorted(groups), excluded=excluded,
        exclusion_reason='Python/font caches and regenerable optical-flow arrays omitted; frozen background references, raw data, failed trials and test logs retained',
        bundles={})
    descriptions = {name: f'Complete frozen snapshot for {name}, including raw arrays and failures'
                    for name in groups}
    descriptions['runtime'] = 'Weights, saved reports, replay media, scores and metadata; NPZ/NPY arrays omitted; viewing bundle, not a complete recomputation environment'
    for name, entries in [('runtime', runtime), *sorted(groups.items())]:
        archive = destination/f'ArmSelf-perception-{tag}-{name}.zip'
        print(f'Packing {name}: {len(entries)} files, {sum(e["bytes"] for e in entries)/1024**2:.1f} MiB', flush=True)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for entry in entries:
                path = ROOT/entry['path']
                output.write(path, entry['path'])
        if archive.stat().st_size >= 2_000_000_000:
            raise ValueError(f'Split this bundle further before upload: {name}')
        manifest['bundles'][name] = dict(asset=archive.name, bytes=archive.stat().st_size,
            sha256=sha256(archive),
            url=f'https://github.com/Yang9008-max/ArmSelf-perception/releases/download/{tag}/{archive.name}',
            description=descriptions[name], files=entries)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (destination/manifest_path.name).write_bytes(manifest_path.read_bytes())
    sums = [f'{b["sha256"]}  {b["asset"]}' for b in manifest['bundles'].values()]
    sums.append(f'{sha256(manifest_path)}  {manifest_path.name}')
    (destination/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n', encoding='utf-8')
    print(f'Complete: {len(groups)} full bundles plus runtime; {manifest_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--tag', default='research-2026-10-26')
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/local-check-20261024')
    parser.add_argument('--support-sources', nargs='*', type=Path, default=[],
                        help='Additional output directories required by the snapshot')
    package(**vars(parser.parse_args()))
