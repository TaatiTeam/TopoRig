#!/usr/bin/env python3
"""Resolve a portable recipe and launch training with optional resource overrides."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.config import load_config, save_config


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/train.yaml')
    parser.add_argument('--run-name')
    parser.add_argument('--gpus',type=int,default=int(os.environ.get('TOPORIG_GPUS','1')))
    parser.add_argument('--print-config',action='store_true')
    args,forwarded=parser.parse_known_args()
    config=load_config(args.config)
    if args.gpus<1: parser.error('--gpus must be positive')
    config.setdefault('distributed',{}).update(enabled=args.gpus>1,nproc_per_node=args.gpus)
    if args.run_name:
        if Path(args.run_name).name!=args.run_name or args.run_name in {'.','..'}:
            parser.error('--run-name must be a single directory name')
        config['project']['run_name']=args.run_name
        config.setdefault('wandb',{})['run_name']=args.run_name
    if args.print_config:
        import yaml
        print(yaml.safe_dump(config,sort_keys=False));return
    target=Path(config['project']['output_dir'])/config['project']['run_name']/'launch.yaml'
    if target.exists():
        # Resume/initialization options go to train.py; don't overwrite provenance
        # for a second launch targeting the same output directory.
        import tempfile
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='launch-',suffix='.yaml',dir=target.parent)
        os.close(fd);target=Path(name)
    save_config(config,target)
    subprocess.run([sys.executable,str(ROOT/'train.py'),'--config',str(target),*forwarded],cwd=ROOT,check=True)


if __name__=='__main__':main()
