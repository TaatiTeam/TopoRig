#!/usr/bin/env python3
"""Check training prerequisites and prepare caches in dependency order."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.config import load_config
from utils.training_config import check_training_assets


def dependency_errors(config: dict) -> list[str]:
    errors=[]
    try:check_training_assets(config)
    except (FileNotFoundError, ValueError) as exc:errors.append(str(exc))
    modules=['torch','numpy','h5py','bpy']
    if config.get('phase')==2:modules+=['mediapipe','cv2','torchvision','timm','einops']
    for name in modules:
        if importlib.util.find_spec(name) is None:errors.append(f'Missing Python dependency: {name}')
    return errors


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/train.yaml')
    parser.add_argument('--step',choices=['check','mesh','image-meshes','eyelids','waft','all'],default='check')
    args=parser.parse_args();config=load_config(args.config)
    errors=dependency_errors(config)
    if errors:
        print('\n'.join(errors),file=sys.stderr);raise SystemExit(2)
    if args.step=='check':print('Training prerequisites found.');return
    def run(script,*flags):
        subprocess.run([sys.executable,str(ROOT/script),'--config',str(args.config.resolve()),*flags],cwd=ROOT,check=True)
    steps=['mesh']+(['image-meshes','eyelids','waft'] if config['phase']==2 else []) if args.step=='all' else [args.step]
    for step in steps:
        if step=='mesh':
            for split in ('train','val'):
                settings=config['data'][split]
                sources=(['primary'] if settings.get('mesh_args') else [])+[name for name,value in (settings.get('extra_mesh_args') or {}).items() if value and value.get('enabled',True)]
                for source in sources:run('scripts/cache_mesh_dataset.py','--split',split,'--mesh-source',source)
        elif config['phase']!=2:parser.error(f'{step} requires a Stage 2 config')
        elif step=='image-meshes':run('scripts/cache_image_meshes.py','--split','train')
        elif step=='eyelids':run('scripts/cache_validated_welded_eyelid_landmarks.py','--split','train')
        elif step=='waft':run('train.py','--cache-only','--single-process')


if __name__=='__main__':main()
