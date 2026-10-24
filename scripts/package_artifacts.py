"""Maintainer: freeze output bundles and a per-file SHA-256 manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifacts import ROOT, sha256


def package(destination, tag):
    destination.mkdir(parents=True, exist_ok=True)
    if destination.resolve().is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Keep release archives outside outputs to avoid including archives in themselves')
    included, excluded = [], []
    for path in sorted((ROOT/'outputs').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        is_cache = any(p.startswith('cache') or p.endswith('-cache') for p in path.relative_to(ROOT/'outputs').parts)
        if (is_cache and path.suffix == '.npz' and path.name != 'background-reference.npz') or path.suffix == '.log':
            excluded.append(dict(path=relative, bytes=path.stat().st_size))
        else:
            included.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256(path)))
    live = 'outputs/perception/v6-eye/model/model.pt'
    current = [e for e in included if e['path'].startswith('outputs/embodiment/') or e['path'] == live]
    runtime = [e for e in current if not {'observations', 'evaluation'} & set(Path(e['path']).parts)]
    legacy = [e for e in included if e not in current]
    manifest = dict(format='arm-self-perception-artifacts-v1', tag=tag,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    exclusion_reason='Regenerable per-sample optical-flow caches and process logs; frozen background references and cache metadata retained; no raw scientific observations excluded',
                    excluded=excluded, bundles={})
    descriptions = dict(runtime='Frozen models, reports, decisions and metadata; raw RGB/evaluation arrays omitted',
                        current='Complete E1/E2/E3 raw observations, evaluation, models, traces and reports plus live V6 checkpoint',
                        legacy='Historical V1-V6 raw observations, models, reports and scene snapshots; regenerable caches omitted')
    for name, entries in (('runtime', runtime), ('current', current), ('legacy', legacy)):
        asset = f'ArmSelf-perception-{tag}-{name}.zip'
        archive = destination/asset
        print(f'Packing {name}: {len(entries)} files, {sum(e["bytes"] for e in entries)/1024**2:.1f} MiB', flush=True)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for entry in entries:
                output.write(ROOT/entry['path'], entry['path'])
        manifest['bundles'][name] = dict(asset=asset, bytes=archive.stat().st_size, sha256=sha256(archive),
            url=f'https://github.com/Yang9008-max/ArmSelf-perception/releases/download/{tag}/{asset}',
            description=descriptions[name], files=entries)
    target = ROOT/'artifacts/manifest.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (destination/'manifest.json').write_bytes(target.read_bytes())
    lines = [f'{b["sha256"]}  {b["asset"]}' for b in manifest['bundles'].values()]
    lines.append(f'{sha256(target)}  manifest.json')
    (destination/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Manifest: {target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--tag', default='handoff-2026-10-24')
    package(**vars(parser.parse_args()))
