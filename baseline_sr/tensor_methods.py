"""Ports of author GTNN/GNLTN ADMM updates, replacing FFT data solve by exact CG.

Float32 CUDA, deterministic seed; MATLAB imresize is replaced by PyTorch bicubic.
GTNN/GNLTN source is pinned under vendor/. No HR truth is accessible here.
"""
from collections import defaultdict
import torch
import torch.nn.functional as F
from baseline_sr.protocol import cg

def shrink(y, rho, learned=False):
    """Batch of tensors [B,n1,n2,n3]; FFT + optional learned mode-3 unitary."""
    original_device = y.device
    # CUDA's batched Jacobi SVD only handles small frontal matrices efficiently.
    # LAPACK is substantially faster for these rectangular matrices; same formula.
    if y.is_cuda and (learned or max(y.shape[1:3])>32):
        y = y.cpu()
    z = torch.fft.fft(y,dim=-1)
    unitary = None
    if learned and y.shape[-1]>1:
        unfolded = z.permute(0,3,2,1).flatten(2)
        # full U needed when mode3 exceeds the other two modes' product.
        unitary,_,_ = torch.linalg.svd(unfolded,full_matrices=unfolded.shape[1]>unfolded.shape[2])
        z = (unitary.mH @ unfolded).reshape(y.shape[0],y.shape[-1],y.shape[2],y.shape[1]).permute(0,3,2,1)
    front = z.permute(0,3,1,2)
    u,s,v = torch.linalg.svd(front,full_matrices=False)
    eps = torch.finfo(torch.float64).eps  # MATLAB eps
    disc = (s-eps).square()-4*(rho-eps*s)
    values = torch.where(disc>0,((s-eps+disc.clamp_min(0).sqrt())/2).clamp_min(0),0)
    z = ((u*values[...,None,:]) @ v).permute(0,2,3,1)
    if unitary is not None:
        unfolded = z.permute(0,3,2,1).flatten(2)
        z = (unitary @ unfolded).reshape(y.shape[0],y.shape[-1],y.shape[2],y.shape[1]).permute(0,3,2,1)
    return torch.fft.ifft(z,dim=-1).real.to(original_device)

class Patches:
    def __init__(self,h,w,size,device):
        self.h,self.w,self.size = h,w,size
        self.stride = size-4
        gh,gw = (h-size)//self.stride+1,(w-size)//self.stride+1
        self.order = torch.arange(gh*gw,device=device).reshape(gh,gw).T.flatten()
        self.inverse = torch.argsort(self.order)
        self.count = F.fold(torch.ones(1,size*size,gh*gw,device=device),
                            (h,w),size,stride=self.stride)[0].permute(1,2,0)
        if (self.count==0).any():
            raise ValueError('Patch grid leaves uncovered pixels')

    def extract(self,x):
        # MATLAB ordering: within a patch y changes fastest; within grid y first.
        c = x.shape[-1]; s = self.size
        p = F.unfold(x.permute(2,0,1)[None],s,stride=self.stride)[0]
        return p[:,self.order].reshape(c,s,s,-1).permute(2,1,0,3).reshape(s*s,c,-1)

    def join(self,p):
        s = self.size; c = p.shape[1]
        cols = p.reshape(s,s,c,-1).permute(2,1,0,3).reshape(c*s*s,-1)[:,self.inverse]
        x = F.fold(cols[None],(self.h,self.w),s,stride=self.stride)[0].permute(1,2,0)
        return x/self.count

def kmeans(x,k, diagnostics=None):
    """Lloyd updates with random sample initialization, matching fkmeans defaults."""
    centers = x[torch.randperm(len(x),device=x.device)[:k]]
    labels = (x@centers.T-.5*centers.square().sum(-1)).argmax(-1)
    for iteration in range(2000):
        # Nearly identical patches can alternate assignments in float32 because
        # dot-product distances subtract almost equal numbers. Refine the same
        # clustering in double precision, without random restarts or GT tuning.
        if iteration == 1000:
            x = x.double()
        _,labels = torch.unique(labels,sorted=True,return_inverse=True)
        count = torch.bincount(labels)
        centers = torch.zeros(len(count),x.shape[-1],device=x.device,dtype=x.dtype)
        centers.index_add_(0,labels,x)
        centers /= count[:,None]
        new = (x@centers.T-.5*centers.square().sum(-1)).argmax(-1)
        if torch.equal(new,labels):
            if diagnostics is not None:
                diagnostics.update(iterations=iteration+1,float64_refinement=iteration>=1000)
            return labels
        labels = new
    raise RuntimeError('Kmeans failed to converge')

def newton(g,a,z,alpha,beta):
    for _ in range(5):
        t = z.tanh(); dt = 1-t.square()
        num = beta*dt*(t-g)+alpha*(z-a)
        den = -2*beta*t*dt*(t-g)+beta*dt.square()+alpha
        candidate = z-num/den
        if candidate.max()>50 or candidate.min() < -500:
            return z
        change = torch.linalg.vector_norm(candidate-z)/torch.linalg.vector_norm(candidate)
        z = candidate
        if change < 1e-4:
            break
    return z

@torch.no_grad()
def solve(obs,steps,report,method,group,budget=None):
    learned = method=='GNLTN'
    p,mu = 10,.01 if learned else .001
    mus = [.01,.01,.001] if learned else [mu]*3
    eta = [.01,.01,.05] if learned else ([.03,.001,.02] if group=='CAVE' else [.01,.0001,.01])
    clusters = 400 if learned or group!='CAVE' else 300
    # Top singular vectors of C x pixels; orthonormal D removes spectral mixing
    # from the LR normal operator, retaining the exact data quadratic.
    u,_,vh = torch.linalg.svd(obs.low.reshape(-1,obs.c).T,full_matrices=False)
    d = u[:,:p]; rd = obs.r @ d
    dominant = vh[0].reshape(*obs.low.shape[:2],1)
    if dominant.mean()<0:
        dominant = -dominant
    dominant = (dominant-dominant.min())/(dominant.max()-dominant.min()).clamp_min(1e-12)
    patches = Patches(obs.h,obs.w,6 if learned else 8,obs.low.device)
    features = patches.extract(torch.cat([obs.msi,obs.interpolate(dominant)],-1))
    clustering = {}
    labels = kmeans(features.permute(2,1,0).flatten(1),clusters,clustering)
    report(0,dict(clustering=clustering))
    groups = defaultdict(list)
    for k in range(int(labels.max())+1):
        idx = torch.where(labels==k)[0]
        if len(idx): groups[len(idx)].append(idx)
    groups = [torch.stack(value) for value in groups.values()]
    a = obs.interpolate(obs.low) @ d
    if budget is not None:
        budget.metadata.update(network_parameters=None, coefficient_variables=a.numel(),
                               spectral_basis_elements=d.numel(),
                               parameter_note='Non-neural optimizer; coefficients are not neural network weights')
    v = [a.clone() for _ in range(3)]
    g = [torch.zeros_like(a) for _ in range(3)]
    pp = [x.tanh() for x in v]
    mm = [torch.zeros_like(a) for _ in range(3)]
    low_coefficient = obs.low @ d
    hr = lambda x: obs.blur(x @ rd.T)
    ht = lambda y: obs.adjoint(hr,y,p)
    lt = lambda y: obs.adjoint(obs.lr,y,p)
    rhs = lt(low_coefficient)+ht(obs.msi)
    normal = lambda x: lt(obs.lr(x))+ht(hr(x))+3*mu*x
    max_residual = 0.
    for it in range(steps):
        a,info = cg(normal,rhs+sum(mu*x+.5*y for x,y in zip(v,g)),a)
        max_residual = max(max_residual,info['relative_residual'])
        for mode in range(3):
            if learned:
                v[mode] = newton(pp[mode]+mm[mode]/(2*mus[mode]),a-g[mode]/(2*mu),v[mode],2*mu,2*mus[mode])
                x = v[mode].tanh()-mm[mode]/(2*mus[mode])
            else:
                x = a-g[mode]/(2*mu)
            all_p = patches.extract(x)
            output = torch.empty_like(all_p)
            for ids in groups:
                batch = all_p[:,:,ids].permute(2,0,1,3)
                order = [(0,1,2,3),(0,1,3,2),(0,2,3,1)][mode]
                inverse = [(0,1,2,3),(0,1,3,2),(0,3,1,2)][mode]
                batch = shrink(batch.permute(order),eta[mode]/(2*mus[mode]),learned).permute(inverse)
                output[:,:,ids] = batch.permute(1,2,0,3)
            result = patches.join(output)
            if learned:
                pp[mode] = result
                mm[mode] += 2*mus[mode]*(result-v[mode].tanh())
            else:
                v[mode] = result
            g[mode] += 2*mu*(v[mode]-a)
        if not torch.isfinite(a).all():
            raise FloatingPointError('Nonfinite ADMM iterate')
        if budget is not None and budget.observe(a @ d.T, steps=it+1):
            break
        report(it+1,dict(cg=info,lr_mse=float((obs.lr(a)-low_coefficient).square().mean()),
                         msi_mse=float((hr(a)-obs.msi).square().mean())))
    return a @ d.T,dict(steps=it+1,rank=p,clusters=clusters,eta=eta,mu=mu,
                        max_cg_relative_residual=max_residual,selection='fixed final iterate',
                        initialization='PyTorch bicubic; positive-mean dominant singular vector',
                        kmeans='random sample initialization, Lloyd until labels stable',clustering=clustering)
