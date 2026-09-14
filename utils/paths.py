"""Environment-configurable roots for tools outside the training resolver."""
from __future__ import annotations
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULTS={'data':'data','metadata':'data/metadata','cache':'.cache','results':'outputs','output':'runs',
          'preparation':'.cache/preparation','reference':'reference_assets','checkpoint':'checkpoints'}


def runtime_path(kind: str, relative: str = '') -> Path:
    if kind not in DEFAULTS:
        raise ValueError(f'Unknown path root: {kind}')
    default=runtime_path('data')/'metadata' if kind=='metadata' else ROOT/DEFAULTS[kind]
    if kind == 'reference':
        from dataset.layout import reference_assets_root
        default = reference_assets_root()
    root=Path(os.environ.get(f'TOPORIG_{kind.upper()}_ROOT',str(default))).expanduser()
    if not root.is_absolute():root=ROOT/root
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a relative path within the selected root')
    return root/path
