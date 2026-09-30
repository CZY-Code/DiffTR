"""Official EDIP-Net networks, known observation operators, fixed final iterate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

def module(name):
    path = Path(__file__).parent/'vendor/EDIP-Net/model'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('edip_'+name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def solve(obs, steps, report, budget=None):
    nchw = lambda x: x.permute(2,0,1)[None]
    hwc = lambda x: x[0].permute(1,2,0)
    low, msi = nchw(obs.low), nchw(obs.msi)
    # R L = D Y because BOTH observed modalities include G. No second blur here.
    lm1 = nchw(obs.low @ obs.r.T)
    lm2 = F.avg_pool2d(msi,obs.scale)
    s2 = module('network_s2').def_two_stream_interactive(obs.msi.shape[-1],obs.c,obs.low.device)
    if budget is not None:
        from baseline_sr.time_budget import parameter_count
        budget.metadata['stage2_network_parameters'] = parameter_count(s2)
    s2steps = 4000 if steps == 14000 else min(steps,4000)
    def optimizer(net, half):
        opt = torch.optim.Adam(net.parameters(),lr=.004)
        sch = torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch: 1-max(0,epoch+1-half)/(half+1))
        return opt, sch
    opt, sch = optimizer(s2,2000)
    fraction = budget.metadata.get('edip_stage2_fraction') if budget is not None else None
    phase_start = budget.elapsed() if fraction is not None else None
    phase_end = budget.limits['optimization'] * fraction if fraction is not None else None
    def timed_lr(opt, start, end):
        progress = (budget.elapsed()-start)/max(end-start,1e-6)
        lr = .004*max(0.,min(1.,2*(1-progress)))
        for group in opt.param_groups:
            group['lr'] = lr
    for i in range(s2steps):
        if fraction is not None:
            timed_lr(opt,phase_start,phase_end)
        a,b = s2(lm1,lm2)
        loss = F.l1_loss(a,low)+F.l1_loss(b,low)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if fraction is None:
            sch.step()
        if budget is not None and budget.expired():
            raise RuntimeError('EDIP stage2 exhausted the time budget before HR reconstruction')
        if i%100 == 0 or i+1 == s2steps:
            report(i+1,dict(stage=2,loss=float(loss.detach())))
        if fraction is not None and budget.elapsed() >= phase_end:
            report(i+1,dict(stage=2,loss=float(loss.detach()),stage_complete=True))
            break
    s2steps = i+1
    with torch.no_grad():
        inputs = tuple(x.detach() for x in s2(msi,msi))
    del s2,opt,sch
    net = module('network_s3').double_U_net_skip(*inputs,SimpleNamespace(device=low.device,band=240))
    if budget is not None:
        budget.metadata['stage3_network_parameters'] = parameter_count(net)
    opt,sch = optimizer(net,7000)
    if fraction is not None:
        phase_start = budget.elapsed()
        phase_end = max(budget.limits.values())
        budget.metadata['learning_rate_schedule'] = 'constant for first half of each allocated phase; linear decay to zero for second half'
    for i in range(steps):
        if fraction is not None:
            timed_lr(opt,phase_start,phase_end)
        outputs = net(*inputs)
        loss = sum(F.l1_loss(obs.lr(hwc(x)),obs.low)+F.l1_loss(obs.hr(hwc(x)),obs.msi) for x in outputs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite EDIP loss')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if fraction is None:
            sch.step()
        if budget is not None:
            with torch.no_grad():
                a,b = (hwc(x.detach()) for x in outputs)
                choose = (obs.hr(a)-obs.msi).square().mean(-1) <= (obs.hr(b)-obs.msi).square().mean(-1)
                timed_result = torch.where(choose[...,None],a,b)
            if budget.observe(timed_result, stage2_steps=s2steps, stage3_steps=i+1):
                break
        if i%25 == 0 or i+1 == steps:
            report(i+1,dict(stage=3,loss=float(loss.detach())))
    # Use the last training forward, consistent with the author's stored outputs.
    a,b = (hwc(x.detach()) for x in outputs)
    with torch.no_grad():
        choose = (obs.hr(a)-obs.msi).square().mean(-1) <= (obs.hr(b)-obs.msi).square().mean(-1)
        result = torch.where(choose[...,None],a,b)
    return result,dict(stage2_steps=s2steps,stage3_steps=i+1,width=240,
                       selection='fixed final iterate; pixelwise observed MSI residual',
                       degradation='known operators; blind stage bypassed',learning_rate=.004)
