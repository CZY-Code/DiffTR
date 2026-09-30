"""List missing task inputs before starting training."""
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['tabular','completion','denoising','sr','all'])
    a=p.parse_args();paths=[]
    if a.task in ['tabular','all']:
        paths += ['data/beijing.npy','data/ctr_50k.npy','data/acc/ibm-large-tensor.txt','data/acc/ibm.mat']
        paths += [f'data/alog/{s}-fold-{i}.txt' for s in ['train','test'] for i in range(1,6)]
    if a.task in ['completion','all']:
        paths += [f'data/misc/{n}.tiff' for n in ['Plane','Peppers','House','Sailboat']]
        paths += [f'data/Videos/{n}_cif.y4m' for n in ['foreman','news','tempete','paris']]
    if a.task in ['completion','sr','denoising','all']:
        names={'completion':['beers','clay','flowers','balloons'], 'sr':['beers','clay','flowers','balloons'],
               'denoising':['balloons','beads','flowers','fruits'],
               'all':['beers','clay','flowers','balloons','beads','fruits']}[a.task]
        paths += [f'data/MSIs/{n}' for n in names]
    if a.task in ['sr','denoising','all']:paths+=['data/PaviaU.mat','data/dc.tif']
    if a.task in ['sr','all']:paths+=['response_coefficient.mat','IKONOS_Spectral_Responses.mat','data/responses/TenFuse/SRF_matrix.mat']
    missing=[]
    for path in paths:
        if not (ROOT/path).exists():missing.append(path);print('MISSING',path)
    print(f'{len(paths)-len(missing)}/{len(paths)} paths found (layout check only).')
    raise SystemExit(bool(missing))

if __name__=='__main__':main()
