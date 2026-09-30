"""Print or execute independent experiment commands from the release root."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sr_protocol import jobs as sr_jobs, METHODS, prepare_manifest


def matrix(suite, output):
    commands = []
    def add(key, *args): commands.append((key, [sys.executable, *map(str, args)]))
    if suite == 'tabular':
        for task in ['alog', 'acc', 'air', 'click']:
            for rank in [3, 5, 8, 10]:
                add(f'{task}_r{rank}', f'exp_diff_{task}.py', '--rank', rank, '--dev', 0)
    elif suite in ['completion-entry', 'completion-tube']:
        for kind, names in [('RGB', ['Plane','Peppers','House','Sailboat']),
                ('MSI', ['beers','clay','flowers','balloons']),
                ('y4m', ['foreman','news','tempete','paris'])]:
            for name in names:
                for rate in [.05,.10,.15,.20,.25]:
                    add(f'{kind}_{name}_sr{rate}', 'exp_diff_imgC.py', '--type', kind,
                        '--name', name, '--SR', rate, '--dev', 0,
                        *(['--tube'] if suite.endswith('tube') else []))
    elif suite == 'denoising':
        for group, names, patches in [('WDC',['WDC'],range(4)),('PaviaU',['PaviaU'],range(2)),
                                      ('CAVE',['balloons','beads','flowers','fruits'],[0])]:
            for name in names:
                for patch in patches:
                    for case in range(1,6):
                        key = f'{group}_{name}_patch{patch}_case{case}'
                        rank = 16 if group=='WDC' and case==3 else 8
                        if (group=='WDC' and case in [1,2,3]) or (group=='PaviaU' and case==2):
                            guidance = 1 if group=='WDC' and case==1 else 4
                            add(key, 'run_guided_denoising.py', '--scene',group,'--case',case,
                                '--variant','likelihood','--crop',0,'--patch',patch,'--steps',10000,
                                '--batch-size',8192,'--rank',rank,'--guidance',guidance,
                                '--noise-protocol','legacy','--device','cuda:0','--output',output/key)
                        else:
                            add(key,'exp_diff_imgD.py','--group',group,'--name',name,'--patch',patch,
                                '--case',case,'--dev',0,'--rank',rank,'--steps',40000,'--eval-every',1000,
                                '--batch-size',8192,'--sparse-mode','on','--sparse-threshold',.04,
                                '--smooth-weight',.01,'--noise-protocol','legacy',
                                '--results-root',output/'runs','--summary-path',output/key/'summary.json')
    elif suite == 'scoretr':
        for group, patches in [('WDC',range(4)),('PaviaU',range(2))]:
            for patch in patches:
                for case in range(1,6):
                    add(f'{group}_patch{patch}_case{case}','scoretr_study/train.py',
                        '--group',group,'--patch',patch,'--case',case,'--steps',40000)
    return commands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',choices=['tabular','completion-entry','completion-tube','denoising','sr','scoretr'])
    p.add_argument('--execute',action='store_true',help='Default only prints commands')
    p.add_argument('--gpus',nargs='+',default=['0'])
    p.add_argument('--output',type=Path)
    p.add_argument('--key',help='Run just one exact job key')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    a=p.parse_args(); os.chdir(ROOT)
    if len(set(a.gpus))!=len(a.gpus): p.error('GPU IDs must be unique')
    out=(a.output or Path('outputs')/a.suite).resolve()
    if a.suite=='sr':
        selected=[j for j in sr_jobs() if a.key is None or j['key']==a.key]
        commands=[]
        for j in selected:
            for method in sorted(a.methods, key=lambda m: m != 'DiffTR'):
                commands.append((method+'_'+j['key'],[sys.executable,'run_sr_time_matched.py',
                    '--output-dir',str(out),'--job-file',str(out/'plan.json'),
                    '--reference-dir',str(out/'reference'),'--method',method,'--key',j['key'],
                    '--edip-stage2-fraction',str(2/9)]))
    else:
        commands=matrix(a.suite,out)
        commands=[c for c in commands if a.key is None or c[0]==a.key]
    if not commands: p.error('No matching jobs')
    for key,cmd in commands: print(key+': '+shlex.join(cmd))
    if not a.execute: return
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'launcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.suite=='sr':
        # A subset run retains the complete plan, so subsequent jobs can share it.
        plan=dict(jobs=sr_jobs(),methods=METHODS,refinement='soft1000')
        path=out/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('Plan mismatch')
        path.write_text(json.dumps(plan,indent=2)+'\n')
        for j in selected: prepare_manifest(j,out)
        # Each scene and its baselines execute sequentially on the same GPU.
        queues=[[c for c in commands if any(c[0].endswith(j['key']) for j in selected[i::len(a.gpus)])]
                for i in range(len(a.gpus))]
    else: queues=[commands[i::len(a.gpus)] for i in range(len(a.gpus))]
    def worker(gpu,queue):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for key,cmd in queue:
            marker=out/'completed'/f'{key}.json'
            if marker.exists():
                if json.loads(marker.read_text())['command']!=cmd: raise ValueError('Completed command mismatch')
                continue
            logpath=out/'logs'/f'{key}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
            # BCD writes its summary after training; make the requested parent now.
            if '--summary-path' in cmd: Path(cmd[cmd.index('--summary-path')+1]).parent.mkdir(parents=True,exist_ok=True)
            print('START',key,'GPU',gpu,flush=True)
            with logpath.open('a') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps(dict(command=cmd,gpu=gpu),indent=2)+'\n')
            print('DONE',key,flush=True)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,g,q) for g,q in zip(a.gpus,queues)]
        for f in futures: f.result()

if __name__=='__main__': main()
