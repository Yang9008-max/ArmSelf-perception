"""Download, restore and verify the versioned research artifacts (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_target(root, name):
    parts = PurePosixPath(name).parts
    if not parts or parts[0] != 'outputs' or any(p in ('..', '.') or ':' in p or '\\' in p for p in parts):
        raise ValueError(f'Unexpected artifact path: {name}')
    target = root.joinpath(*parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact path escapes the checkout: {name}')
    return target


def restore(archive, bundle, root):
    if sha256(archive) != bundle['sha256']:
        raise ValueError(f'Archive SHA-256 mismatch: {archive}')
    members = {entry['path']: entry for entry in bundle['files']}
    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(members) or set(source.namelist()) != set(members):
            raise ValueError('ZIP members differ from the manifest')
        # Check every existing destination before writing any member.
        for name, entry in members.items():
            target = checked_target(root, name)
            if source.getinfo(name).file_size != entry['bytes']:
                raise ValueError(f'ZIP member size differs: {name}')
            if target.exists() and (not target.is_file() or sha256(target) != entry['sha256']):
                raise FileExistsError(f'Preserving different existing file: {target}. Restore into a clean clone.')
        for name, entry in members.items():
            target = checked_target(root, name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.restore-part')
            with source.open(name) as incoming, temporary.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if sha256(temporary) != entry['sha256']:
                temporary.unlink()
                raise ValueError(f'Extracted file SHA-256 mismatch: {name}')
            temporary.replace(target)
    print(f'Restored {len(members)} verified files from {bundle["asset"]}', flush=True)


def download_url(bundle, release_cache=None):
    """Use the public asset API, avoiding an extra github.com redirect hop."""
    parsed = urlsplit(bundle['url'])
    parts = parsed.path.strip('/').split('/')
    if parsed.hostname != 'github.com' or len(parts) != 6 or parts[2:4] != ['releases', 'download']:
        return bundle['url']
    owner, repo, _, _, tag, _ = parts
    endpoint = f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{quote(tag, safe="")}'
    try:
        if release_cache is not None and endpoint in release_cache:
            release = release_cache[endpoint]
        else:
            request = Request(endpoint, headers={'User-Agent': 'ArmSelf-perception-artifact-restore',
                                                 'Accept': 'application/vnd.github+json'})
            with urlopen(request, timeout=20) as response:
                release = json.load(response)
            if release_cache is not None:
                release_cache[endpoint] = release
        asset = next(entry for entry in release['assets'] if entry['name'] == bundle['asset'])
        if asset['size'] != bundle['bytes'] or asset['state'] != 'uploaded':
            raise ValueError(f'Release asset is incomplete: {bundle["asset"]}')
        print(f'Downloading {bundle["asset"]} through the public GitHub asset API', flush=True)
        return asset['url']
    except OSError as error:
        print(f'GitHub API lookup unavailable; trying the direct Release URL: {error}', flush=True)
        return bundle['url']


def download(bundle, cache, release_cache=None):
    archive = cache/bundle['asset']
    if archive.is_file() and sha256(archive) == bundle['sha256']:
        return archive
    partial = archive.with_suffix(archive.suffix+'.part')
    if partial.is_file() and partial.stat().st_size >= bundle['bytes']:
        if sha256(partial) == bundle['sha256']:
            partial.replace(archive)
            return archive
        partial.unlink()
    url = download_url(bundle, release_cache)
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'ArmSelf-perception-artifact-restore',
                       'Accept': 'application/octet-stream'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                append = offset > 0 and response.status == 206
                with partial.open('ab' if append else 'wb') as stream:
                    # Save each available block; read(n) can lose an incomplete
                    # large block when a slow connection times out mid-read.
                    while chunk := response.read1(1024*1024):
                        stream.write(chunk)
            if sha256(partial) != bundle['sha256']:
                partial.unlink()
                raise ValueError(f'Download SHA-256 mismatch: {bundle["asset"]}')
            partial.replace(archive)
            return archive
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise
            print(f'Download retry {attempt+1}: {error}', flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fetch', 'restore', 'verify', 'list'))
    parser.add_argument('--bundle', default='runtime',
                        help='Bundle name from the selected manifest, or all for its complete dataset')
    parser.add_argument('--archive-dir', type=Path, default=ROOT/'.artifacts',
                        help='Download cache, or directory containing offline Release ZIPs')
    parser.add_argument('--root', type=Path, default=ROOT, help='Destination checkout')
    parser.add_argument('--manifest', type=Path, default=ROOT/'artifacts/manifest.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    names = (tuple(manifest.get('default_bundles', ('current', 'legacy'))) if args.bundle == 'all'
             else tuple(manifest.get('bundle_groups', {}).get(args.bundle, (args.bundle,))))
    unknown = set(names) - set(manifest['bundles'])
    if unknown:
        parser.error(f'Unknown bundles: {sorted(unknown)}; available: {sorted(manifest["bundles"])} and all')
    release_cache = {}
    for name in names:
        bundle = manifest['bundles'][name]
        if args.mode == 'list':
            print(f'{name}: {bundle["bytes"]/1024**2:.1f} MiB, {len(bundle["files"])} files; {bundle["description"]}')
        elif args.mode == 'verify':
            bad = [entry['path'] for entry in bundle['files']
                   if not (p := checked_target(args.root, entry['path'])).is_file()
                   or p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']]
            if bad:
                raise SystemExit(f'{len(bad)} missing/changed files; first entries: {bad[:10]}')
            print(f'{name}: all {len(bundle["files"])} installed files match SHA-256')
        else:
            args.archive_dir.mkdir(parents=True, exist_ok=True)
            archive = download(bundle, args.archive_dir, release_cache) if args.mode == 'fetch' else args.archive_dir/bundle['asset']
            restore(archive, bundle, args.root)


if __name__ == '__main__':
    main()
