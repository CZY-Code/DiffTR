"""Render paper SR scenes: shared-scale RGB, spectral angle, and per-band PSNR."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'outputs/sr')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/sr/figures');a=p.parse_args()
    import os,json
    os.chdir(ROOT)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from baseline_sr.protocol import prepare
    from scripts.sr_protocol import jobs
    from ablation_study.common import metrics,write_json,sha
    methods=['CS2DIPs','GTNN','GNLTN','EDIP-Net','DiffTR']
    scenes=[('WDC','WDC',[100,60,30]),('PaviaU','PaviaU',[60,30,10]),('CAVE','beers',[30,16,5])]
    a.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(6,6,figsize=(14,12),layout='constrained')
    bands_fig,bands_axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    audit=[];cmap=plt.get_cmap('magma').copy();cmap.set_bad('gray')
    for row,(group,name,bands) in enumerate(scenes):
        j=next(j for j in jobs() if j['group']==group and j['name']==name and j['patch']==0 and j['DS']==16 and j['sigma']==.5)
        gt,obs,hashes=prepare(j,'cpu',a.root/'reference/observations'/f'{j["key"]}.json');del obs
        lo,hi=np.percentile(gt[:,:,bands],[1,99],axis=(0,1))
        for col,method in enumerate(methods+['GT']):
            if method=='GT':rec=gt
            else:
                folder=a.root/method/j['key'];metadata=json.loads((folder/'result.json').read_text())
                if any(metadata[k]!=v for k,v in hashes.items()):raise ValueError('Observation mismatch')
                path=folder/('reconstruction_seed3407.npy' if method=='DiffTR' else 'reconstruction_optimization.npy')
                rec=np.load(path)
                if rec.shape!=gt.shape or not np.isfinite(rec).all():raise ValueError(path)
                rec=rec.clip(0,1);scores=metrics(gt,rec)
                expected=(next(s for s in metadata['samples'] if s['seed']==3407) if method=='DiffTR' else metadata['measured']['optimization'])
                if any(abs(scores[k]-expected[k])>1e-5 for k in scores):raise ValueError('Stored metric mismatch: '+str(path))
                audit.append(dict(key=j['key'],method=method,sha256=sha(path),**scores))
                mse=np.mean((gt.astype(float)-rec.astype(float))**2,axis=(0,1))
                bands_axes[row].plot(np.arange(1,len(mse)+1),-10*np.log10(np.maximum(mse,1e-30)),label=method)
            rgb=np.clip((rec[:,:,bands]-lo)/np.maximum(hi-lo,1e-12),0,1)
            x,y=gt.astype(float),rec.astype(float)
            den=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1);valid=den>1e-12
            sam=np.full(gt.shape[:2],np.nan)
            sam[valid]=np.degrees(np.arccos(np.clip(np.sum(x*y,axis=-1)[valid]/den[valid],-1,1)))
            axes[2*row,col].imshow(rgb);im=axes[2*row+1,col].imshow(sam,cmap=cmap,vmin=0,vmax=15)
            for ax in [axes[2*row,col],axes[2*row+1,col]]:ax.set_xticks([]);ax.set_yticks([])
            if row==0:axes[0,col].set_title(method)
        axes[2*row,0].set_ylabel(group+' RGB');axes[2*row+1,0].set_ylabel(group+' SAM')
        bands_axes[row].set(title=group,xlabel='Band',ylabel='PSNR (dB)');bands_axes[row].grid(alpha=.25)
    bands_axes[-1].legend();fig.colorbar(im,ax=axes.ravel().tolist(),label='Spectral angle (degrees)',shrink=.6)
    for name,canvas in [('sr_visual_sam',fig),('bandwise_psnr',bands_fig)]:
        for suffix in ['pdf','png']:canvas.savefig(a.output/f'{name}.{suffix}',dpi=300)
        plt.close(canvas)
    write_json(a.output/'provenance.json',dict(reconstructions=audit,source_sha256=sha(Path(__file__)),
        display='GT 1st/99th percentile per scene/channel; SAM 0..15 degrees, undefined gray',
        protocol='DS16 sigma0.5; DiffTR seed3407; baseline primary optimization budget'))

if __name__=='__main__':main()
