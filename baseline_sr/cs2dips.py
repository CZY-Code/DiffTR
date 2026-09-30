"""Author CGDIP network, with matched operators and torchvision DCN backend."""
import math
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torchvision.ops import DeformConv2d

class DeformConv2dPack(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kw):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=False, **kw)
        self.conv_offset = nn.Conv2d(in_channels, 2*self.kernel_size[0]*self.kernel_size[1],
                                    self.kernel_size, stride=self.stride, padding=self.padding)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x):
        return super().forward(x, self.conv_offset(x))

def solve(obs, steps, report, budget=None):
    path = Path(__file__).parent/'vendor/CS2DIPs/modules'
    sys.path.insert(0, str(path))
    # Keep vendor source byte-identical. Only satisfy its mmcv DCN import.
    package = types.ModuleType('mmcv')
    ops = types.ModuleType('mmcv.ops')
    ops.DeformConv2dPack = DeformConv2dPack
    package.ops = ops
    sys.modules['mmcv'], sys.modules['mmcv.ops'] = package, ops
    import model
    device = obs.low.device
    h, w, c = obs.h, obs.w, obs.c
    lh, lw = obs.low.shape[:2]
    net = model.CGDIP(num_input_channels1d=lh*lw, num_inputz_channels1d=256,
                     num_input_channels2d=obs.msi.shape[-1], num_inputz_channels2d=10,
                     data_shape=[c,h,w], num_output_channels=35, num_channels=256,
                     num_channels_skip=8).to(device).train()
    g = [nn.Parameter(torch.rand(1,256,math.ceil(c/8), device=device)/10),
         nn.Parameter(torch.rand(1,10,math.ceil(h/32),math.ceil(w/32), device=device)/10)]
    noise = [x.detach().clone() for x in g]
    optimizer = torch.optim.Adam(list(net.parameters())+g, lr=.001)
    if budget is not None:
        from baseline_sr.time_budget import parameter_count
        budget.metadata.update(network_parameters=parameter_count(net),
                               optimized_input_parameters=sum(x.numel() for x in g))
    low = obs.low.reshape(1,lh*lw,c)
    msi = obs.msi.permute(2,0,1)[None]
    loss_fn = nn.HuberLoss(reduction='sum', delta=.1)
    best, best_loss = None, float('inf')
    for i in range(steps):
        z, _, _ = net(low, g[0]+noise[0].normal_()/30, msi, g[1]+noise[1].normal_()/30)
        z = z.permute(1,2,0)
        loss = loss_fn(obs.hr(z),obs.msi)+loss_fn(obs.lr(z),obs.low)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite CS2DIPs loss')
        if value < best_loss:
            best, best_loss = z.detach().clone(), value
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if budget is not None and budget.observe(best, steps=i+1,
                                                 best_observation_loss=best_loss):
            break
        if i%25 == 0 or i+1 == steps:
            report(i+1, dict(loss=value, best_observation_loss=best_loss))
    return best, dict(steps=i+1, selection='minimum observation Huber loss',
                      backend='torchvision deformable convolution', rank=35,
                      width=256, learning_rate=.001)
