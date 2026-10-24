"""Resolve recorded research paths after moving the repository to another host."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def recorded_path(value, root=None):
    """Prefer this checkout's outputs over the original machine's absolute path.

    Historical JSON and checkpoints stay byte-identical for provenance checks.
    This only changes how their directory references are interpreted by readers.
    """
    root = Path(root) if root is not None else PROJECT_ROOT
    parts = str(value).replace('\\', '/').split('/')
    candidates = []
    if 'outputs' in parts:
        relative = parts[parts.index('outputs'):]
        if '..' not in relative:
            candidates.append(root.joinpath(*relative))
    path = Path(value)
    if not path.is_absolute() and not (parts and parts[0].endswith(':')):
        candidates.append(root / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'Recorded artifact is missing: {value}. Restore the experiment bundles '
        'with python scripts/artifacts.py fetch --bundle all from the repository root.')


def portable_path(value):
    """Use repository-relative paths for newly written metadata when possible."""
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
