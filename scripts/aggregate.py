"""Aggregate complete main-table runs; missing cells are reported, never imputed."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sr_protocol import jobs, METHODS, CASES
from scripts.run_experiments import matrix


def aggregate_sr(root):
    rows=[]; missing=[]
    for method in METHODS:
        for group in ['WDC','PaviaU','CAVE']:
            for scale,sigma in CASES:
                selected=[j for j in jobs() if (j['group'],j['DS'],j['sigma'])==(group,scale,sigma)]
                records=[]
                for j in selected:
                    path=root/method/j['key']/'result.json'
                    if not path.exists(): missing.append(str(path));continue
                    r=json.loads(path.read_text())
                    if not r.get('complete'): missing.append(str(path));continue
                    if method!='DiffTR':
                        reference=json.loads((root/'DiffTR'/j['key']/'result.json').read_text())
                        for k in ['gt_sha256','response_sha256','LR_HSI_sha256','HR_MSI_sha256']:
                            if r[k]!=reference[k]: raise ValueError('Observation mismatch: '+str(path))
                        r=r['measured']['optimization']
                        if not r['available']: missing.append(str(path)+' (no eligible output)');continue
                    records.append(r)
                if len(records)==len(selected):
                    rows.append(dict(method=method,group=group,scale=scale,sigma=sigma,n=len(records),
                        **{m:mean(r[m] for r in records) for m in ['psnr','ssim','nrmse']}))
    return rows,missing


def aggregate_denoising(root):
    rows=[];missing=[];records={}
    for key,cmd in matrix('denoising',root):
        path=root/key/'summary.json'
        if not path.exists():missing.append(str(path));continue
        r=json.loads(path.read_text())
        if '--guidance' in cmd:
            strength=cmd[cmd.index('--guidance')+1]
            r=r['reconstructions']['prox_'+strength]
        group=key.split('_')[0];case=int(key.rsplit('case',1)[1])
        records.setdefault((group,case),[]).append(r)
    for (group,case),runs in sorted(records.items()):
        if len(runs)!=(2 if group=='PaviaU' else 4):continue
        rows.append(dict(group=group,case=case,n=len(runs),
            **{m:mean(r[m] for r in runs) for m in ['psnr','ssim','nrmse']}))
    return rows,missing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',choices=['sr','denoising'])
    p.add_argument('--root',type=Path);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();root=(a.root or ROOT/'outputs'/a.suite).resolve()
    rows,missing=(aggregate_sr if a.suite=='sr' else aggregate_denoising)(root)
    if missing and not a.allow_partial:
        raise SystemExit(f'{len(missing)} runs missing/ineligible; first: {missing[0]}. Use --allow-partial to export only complete cells.')
    root.mkdir(parents=True,exist_ok=True)
    (root/'aggregation.json').write_text(json.dumps(dict(rows=rows,missing=missing),indent=2)+'\n')
    if rows:
        with (root/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'{len(rows)} complete cells; {len(missing)} missing/ineligible runs: {root}')

if __name__=='__main__':main()
