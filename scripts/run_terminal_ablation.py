"""Six fixed terminal-refinement combinations and paired sensitivity sweeps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
KEYS=[f'{g}_{g}_patch0_DS4_sigma{s}' for g in ['WDC','PaviaU'] for s in ['0.5','2']]+[
    'CAVE_beers_patch0_DS4_sigma0.5','CAVE_beers_patch0_DS16_sigma0.5']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--sensitivity',action='store_true')
    a=p.parse_args();os.chdir(ROOT)
    commands=[[sys.executable,'run_sr_subspace_study.py','--key',k,'--device',a.device] for k in KEYS]
    if a.sensitivity:commands=[[sys.executable,'run_sr_subspace_sensitivity.py','--device',a.device]]
    for cmd in commands:print(' '.join(cmd))
    if a.execute:
        out=ROOT/'outputs/subspace';out.mkdir(parents=True,exist_ok=True)
        (out/'plan.json').write_text(json.dumps(dict(scenes=KEYS),indent=2)+'\n')
        for cmd in commands:subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
