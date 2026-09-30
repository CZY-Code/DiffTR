import torch
import math
from torch import nn
import numpy as np
from torch.nn.utils import weight_norm, spectral_norm
from typing import Optional, List
from einops import rearrange, reduce

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([100.0])) / (half_size - 1) #10000
        emb = torch.exp(-emb * torch.arange(half_size))
        self.emb = nn.Parameter(emb, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        emb = x * self.emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb
    
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    def __init__(self, embedding_size=32, scale=1.0):
        super(GaussianFourierProjection, self).__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


def config_activation(act: Optional[str]):
    act_dict = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'selu': nn.SELU(),
        'silu': nn.SiLU(),
        'gelu': nn.GELU(),
    }
    if act is None:
        return None
    else:
        assert act.lower() in act_dict.keys(), "ACT not supported"
        return act_dict[act.lower()]

def setup_mlp(input_dim: int, output_dim: int, h_dim: List[int],
              act: Optional[nn.Module], out_act: Optional[nn.Module],
              dropout: Optional[float] = None,
              bn: bool = False, wn: bool = False, sn: bool = False):
    layer = []
    layer_dim = [input_dim] + h_dim
    
    if dropout is not None:
        dropout_layer = nn.Dropout(dropout)

    for i in range(len(layer_dim) - 1):
        if wn:
            layer.append(weight_norm(nn.Linear(layer_dim[i], layer_dim[i+1])))
        elif sn:
            layer.append(spectral_norm(nn.Linear(layer_dim[i], layer_dim[i+1])))
        else:
            layer.append(nn.Linear(layer_dim[i], layer_dim[i+1]))

        if act is not None:
            layer.append(act)

        if bn:
            layer.append(nn.BatchNorm1d(layer_dim[i+1], track_running_stats=False))

        if dropout is not None:
            layer.append(dropout_layer)

    # output layer
    if wn:
        layer.append(weight_norm(nn.Linear(layer_dim[-1], output_dim)))
    elif sn:
        layer.append(spectral_norm(nn.Linear(layer_dim[-1], output_dim)))
    else:
        layer.append(nn.Linear(layer_dim[-1], output_dim))

    if out_act is not None:
        layer.append(out_act)

    return nn.Sequential(*layer)

class MLP(nn.Module):
    """docstring for MLP."""
    def __init__(self, input_dim: int, output_dim: int, h_dim: List[int],
                 act: Optional[nn.Module], 
                 skip_connection: bool = True, 
                 dropout: Optional[float] = None,
                 bn: bool = False, wn: bool = False, sn: bool = False,):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.act = act
        self.bn = bn
        self.wn = wn
        self.sn = sn
        self.skip_connection = skip_connection
        self.dropout = dropout

        self.layer = setup_mlp(input_dim, output_dim, h_dim,
                                self.act, out_act=None, dropout=dropout, 
                                bn=bn, wn=wn, sn=sn)

        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.layer(x)

        if self.skip_connection:
            out = out + self.skip_layer(x)

        return out