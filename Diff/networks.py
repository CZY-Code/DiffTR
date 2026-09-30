
import torch
import torch.nn as nn
from typing import List

from .modules import *
from .diff_utils import *
import rff

INIT_QZ_SIGMA = 0.1 #0.1

# model
class ScalarUNet(nn.Module):
    def __init__(
        self,
        tensor_shape: List[int],
        rank: int,
        h_dim: List[int],
        act: str = 'elu',
        dropout: float = 0.,
        latent_dim: int = 20,
        x_emb_size: int = 8,
        sigma_func: str = 'exp',
        pooling_method: str = 'sum',
        skip_connection: bool = False,
    ):
        super(ScalarUNet, self).__init__()
        self.tensor_shape = list(tensor_shape)
        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)
        
        self.setup_q_z_() #setting shared latent for low-rankness
        self.setup_energy_()
        # self.val_embedding = GaussianFourierProjection(self.x_emb_size)
        self.time_embedding = GaussianFourierProjection(4)

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])

    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [12, 5] [6, 5]
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            # torch.nn.init.normal_(q.data, -2, INIT_QZ_SIGMA)
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)
    
    def setup_energy_(self):
        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        emb_size = 9
        x_enc = MLP(input_dim=emb_size, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # ouput layer
        if self.pooling_method in ['sum', 'attn']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 2 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(input_dim=in_size, output_dim=1,
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 'output': output_layer})

    def forward(self, x, idx, t, sample=False, return_z=False):
        z = []
        for d in range(self.dim):
            if sample: #diagonal Gaussian and use reparameterization trick
                z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        t = self.time_embedding(t)
        # x = self.val_embedding(x) #size [128]->[128, 8]
        x_exp = torch.cat([x.view(-1, 1), t], -1) #size [128, 1+8]
        x_exp = self.layers['x_enc'](x_exp) #[128, 20]
        z_exp = self.layers['z_enc'](z_ten) #[128, 20]

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        else:
            raise RuntimeError('Wrong pooling method!')
        # TODO
        output = self.layers['output'](hidden).unsqueeze(-1) #size [128, 1, 1]
        output += x

        if return_z:
            return output, z 
        else:
            return output 


class ScalarUNetV2(nn.Module):
    def __init__(
        self,
        tensor_shape: List[int],
        rank: int,
        h_dim: List[int],
        act: str = 'elu',
        dropout: float = 0.,
        latent_dim: int = 20,
        x_emb_size: int = 8,
        sigma_func: str = 'exp',
        pooling_method: str = 'sum',
        skip_connection: bool = False,
    ):
        super(ScalarUNetV2, self).__init__()
        self.tensor_shape = list(tensor_shape)
        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)
        
        self.setup_q_z_() ##setting shared latent for low-rankness

        self.val_embedding = SinusoidalEmbedding(self.x_emb_size)
        self.time_embedding = SinusoidalEmbedding(self.x_emb_size)

        act = config_activation(self.act)
        
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # x encoder
        x_enc = MLP(input_dim=self.x_emb_size, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # t encoder
        t_enc = MLP(input_dim=self.x_emb_size, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # ouput layer
        if self.pooling_method in ['sum', 'attn']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 3 * self.latent_dim 
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(input_dim=in_size, output_dim=1, h_dim=self.h_dim, 
                           act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 't_enc': t_enc, 'output': output_layer})

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])

    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [12, 5] [6, 5]
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            # torch.nn.init.normal_(q.data, -2, INIT_QZ_SIGMA)
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)

    def forward(self, x, idx, t, sample=False, return_z=False):
        z = []
        for d in range(self.dim):
            if sample: #diagonal Gaussian and use reparameterization trick
                z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        time_emb = self.time_embedding(t.view(-1,1))
        x_emb = self.val_embedding(x.view(-1,1)) #size [128, 1, 1]->[128, 8]
        # x_exp = torch.cat([x_emb, time_emb], -1) #size [128, 16]
        # x_exp = x_exp * (scale + 1) + shift #[128, 8]
        t_exp = self.layers['t_enc'](time_emb) #[128, 64]
        x_exp = self.layers['x_enc'](x_emb) #[128, 64]
        z_exp = self.layers['z_enc'](z_ten) #[128, 64]

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp, t_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        else:
            raise RuntimeError('Wrong pooling method!')
        
        output = self.layers['output'](hidden).sum(-1).unsqueeze(-1).unsqueeze(-1) #size [128, 1, 1]
        output += x

        if return_z:
            return output, z 
        else:
            return output 


class ScalarUNetV3(nn.Module): #time continuous
    def __init__(
        self,
        tensor_shape: List[int],
        rank: int,
        h_dim: List[int],
        act: str = 'elu',
        dropout: float = 0.,
        latent_dim: int = 20,
        x_emb_size: int = 8,
        sigma_func: str = 'exp',
        pooling_method: str = 'sum',
        skip_connection: bool = False,
    ):
        super(ScalarUNetV3, self).__init__()
        self.tensor_shape = list(tensor_shape)
        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)
        
        self.setup_q_z_() ##setting shared latent for low-rankness

        self.val_embedding = SinusoidalEmbedding(self.x_emb_size)
        self.time_embedding = SinusoidalEmbedding(self.x_emb_size)
        self.time_embedding_v2 = SinusoidalEmbedding(self.x_emb_size)

        act = config_activation(self.act)
        
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # x encoder
        x_enc = MLP(input_dim=self.x_emb_size, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # t encoder
        t_enc = MLP(input_dim=self.x_emb_size, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)

        t_enc_v2 = MLP(input_dim=self.x_emb_size, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # ouput layer
        output_layer = MLP(input_dim=4 * self.latent_dim, output_dim=1, h_dim=self.h_dim, 
                           act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 't_enc': t_enc, 
                                     't_enc_v2': t_enc_v2, 'output': output_layer})

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])

    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [12, 5] [6, 5]
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            # torch.nn.init.normal_(q.data, -2, INIT_QZ_SIGMA)
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)

    def forward(self, x, idx, idt, t, sample=False, return_z=False):
        z = []
        for d in range(self.dim):
            if sample: #diagonal Gaussian and use reparameterization trick
                z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        time_emb = self.time_embedding(t.view(-1,1))
        x_emb = self.val_embedding(x.view(-1,1)) #size [128, 1, 1]->[128, 8]
        idt_emb = self.time_embedding_v2(idt.view(-1,1))
        # x_exp = torch.cat([x_emb, time_emb], -1) #size [128, 16]
        # x_exp = x_exp * (scale + 1) + shift #[128, 8]
        t_exp = self.layers['t_enc'](time_emb) #[128, 64]
        idt_exp = self.layers['t_enc_v2'](idt_emb) #[128, 64]
        x_exp = self.layers['x_enc'](x_emb) #[128, 64]
        z_exp = self.layers['z_enc'](z_ten) #[128, 64]

        if self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp, idt_exp, t_exp], -1)
        else:
            raise RuntimeError('Wrong pooling method!')

        output = self.layers['output'](hidden).unsqueeze(-1) #size [128, 1, 1]
        output += x

        if return_z:
            return output, z 
        else:
            return output 

class ScalarINR2D(nn.Module):
    def __init__(
        self, out_dim, tensor_shape, rank, h_dim, 
        act, dropout, latent_dim, x_emb_size, t_emb_size, 
        sigma_func, pooling_method, skip_connection, posdim, 
        rff_alpha, rff_sigma):
        super(ScalarINR2D, self).__init__()
        self.out_dim = out_dim
        self.tensor_shape = list(tensor_shape)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('norm_factors', torch.tensor(self.tensor_shape, dtype=torch.float32))

        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        self.t_emb_size = t_emb_size
        self.posdim = posdim
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)

        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        x_enc = MLP(input_dim=self.out_dim*self.x_emb_size, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        # ouput layer
        if self.pooling_method in ['sum', 'attn', 'none']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 3 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        
        output_layer = MLP(input_dim=in_size, output_dim=self.out_dim, 
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        # t encoder
        t_enc = MLP(input_dim=self.t_emb_size, output_dim=self.latent_dim, h_dim=self.h_dim, 
                    act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 't_enc': t_enc, 'output': output_layer})
        
        self.U_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.V_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)

        self.encoding = rff.layers.GaussianEncoding(alpha=rff_alpha, sigma=rff_sigma, input_size=1, encoded_size=self.posdim//2)
        self.time_embedding = SinusoidalEmbedding(self.t_emb_size)
        self.val_embedding = SinusoidalEmbedding(self.x_emb_size)
        # self.setup_q_z_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])
    
    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [dim1, rank] [dim2, rank] 需要优化的参数
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)

    def forward(self, val, ind, t, sample=True):
        z = []
        mu = []
        train_ind = ind.float() / self.norm_factors
        mu.append(self.U_Net_mu(self.encoding(train_ind[:,0].unsqueeze(-1))))
        mu.append(self.V_Net_mu(self.encoding(train_ind[:,1].unsqueeze(-1))))
        for d in range(self.dim):
            if sample:
                # z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))[ind[:, d]]
                # z_d = self.q_z_mu[d][ind[:, d]]
                # z_d = gaussian_repar(mu=mu[d], sigma=sigma[d])
                z_d = mu[d]
            else:
                # z_d = self.q_z_mu[d][ind[:, d]]
                z_d = mu[d]
            z.append(z_d)
            
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        time_emb = self.time_embedding(t.unsqueeze(-1))
        val_emb = self.val_embedding(val.unsqueeze(-1)).view(val.shape[0], -1) #size [B, 1, 3]->[B, 3*8]

        t_exp = self.layers['t_enc'](time_emb)
        x_exp = self.layers['x_enc'](val_emb)
        z_exp = self.layers['z_enc'](z_ten)

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp, t_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        elif self.pooling_method == 'none':
            hidden = x_exp
        else:
            raise RuntimeError('Wrong pooling method!')

        output = self.layers['output'](hidden).unsqueeze(1) #size [128] 保证energy为正
        output += val
        
        return output


class ScalarINR3D(nn.Module):
    def __init__(
        self, out_dim, tensor_shape, rank, h_dim, 
        act, dropout, latent_dim, x_emb_size, t_emb_size, 
        sigma_func, pooling_method, skip_connection, posdim, 
        rff_alpha, rff_sigma):
        super(ScalarINR3D, self).__init__()
        self.out_dim = out_dim
        self.tensor_shape = list(tensor_shape)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('norm_factors', torch.tensor(self.tensor_shape, dtype=torch.float32))

        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        self.t_emb_size = t_emb_size
        self.posdim = posdim
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)

        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        
        # x encoder
        x_enc = MLP(input_dim=self.out_dim*self.x_emb_size, output_dim=self.latent_dim, #
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # t encoder
        t_enc = MLP(input_dim=self.t_emb_size, output_dim=self.latent_dim, 
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # ouput layer
        if self.pooling_method in ['sum', 'attn', 'none']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 3 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(input_dim=in_size, output_dim=self.out_dim, 
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 't_enc': t_enc, 'output': output_layer})
        
        self.U_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.V_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.W_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)

        self.encoding = rff.layers.GaussianEncoding(alpha=rff_alpha, sigma=rff_sigma, input_size=1, encoded_size=self.posdim//2)
        self.time_embedding = SinusoidalEmbedding(self.t_emb_size)
        # self.time_embedding = rff.layers.GaussianEncoding(alpha=1, sigma=2, input_size=1, encoded_size=self.t_emb_size//2)
        self.val_embedding = SinusoidalEmbedding(self.x_emb_size)
        # self.val_embedding = rff.layers.GaussianEncoding(alpha=1, sigma=2, input_size=1, encoded_size=self.x_emb_size//2)

        # self.setup_q_z_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])
    
    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [dim1, rank] [dim2, rank] 需要优化的参数
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)

    def forward(self, val, ind, t, sample=True):
        z = []
        mu = []
        train_ind = ind.float() / self.norm_factors
        mu.append(self.U_Net_mu(self.encoding(train_ind[:,0].unsqueeze(-1))))
        mu.append(self.V_Net_mu(self.encoding(train_ind[:,1].unsqueeze(-1))))
        mu.append(self.W_Net_mu(self.encoding(train_ind[:,2].unsqueeze(-1))))
        for d in range(self.dim):
            if sample:
                # z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))[ind[:, d]]
                # z_d = self.q_z_mu[d][ind[:, d]]
                # z_d = gaussian_repar(mu=mu[d], sigma=sigma[d])
                z_d = mu[d]
            else:
                # z_d = self.q_z_mu[d][ind[:, d]]
                z_d = mu[d]
            z.append(z_d)
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        time_emb = self.time_embedding(t.unsqueeze(-1))
        val_emb = self.val_embedding(val.unsqueeze(-1)).view(val.shape[0], -1) #size [B, 1, 3]->[B, 3*8]

        t_exp = self.layers['t_enc'](time_emb)
        x_exp = self.layers['x_enc'](val_emb)
        z_exp = self.layers['z_enc'](z_ten)

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([x_exp, z_exp, t_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        elif self.pooling_method == 'none':
            hidden = x_exp
        else:
            raise RuntimeError('Wrong pooling method!')

        output = self.layers['output'](hidden).unsqueeze(1) #size [128] 保证energy为正
        output += val
        
        return output


class ScalarINR4D(nn.Module):
    def __init__(
        self, out_dim, tensor_shape, rank, h_dim, 
        act, dropout, latent_dim, x_emb_size, t_emb_size, 
        sigma_func, pooling_method, skip_connection, posdims, 
        rff_alphas, rff_sigmas):
        super(ScalarINR4D, self).__init__()
        self.out_dim = out_dim
        self.tensor_shape = list(tensor_shape)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('norm_factors', torch.tensor(self.tensor_shape, dtype=torch.float32))

        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        self.t_emb_size = t_emb_size
        self.posdims = posdims
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)

        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        
        # x encoder
        x_enc = MLP(input_dim=self.out_dim*self.x_emb_size, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # t encoder
        t_enc = MLP(input_dim=self.t_emb_size, output_dim=self.latent_dim, 
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # ouput layer
        if self.pooling_method in ['sum', 'attn', 'none']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 3 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(input_dim=in_size, output_dim=self.out_dim, 
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 't_enc': t_enc, 'output': output_layer})
        
        self.U_Net_mu = MLP(input_dim=self.posdims[0], output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.V_Net_mu = MLP(input_dim=self.posdims[1], output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.W_Net_mu = MLP(input_dim=self.posdims[2], output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.T_Net_mu = MLP(input_dim=self.posdims[3], output_dim=self.rank, 
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)

        self.U_encoding = rff.layers.GaussianEncoding(alpha=rff_alphas[0], sigma=rff_sigmas[0], input_size=1, encoded_size=self.posdims[0]//2)
        self.V_encoding = rff.layers.GaussianEncoding(alpha=rff_alphas[1], sigma=rff_sigmas[1], input_size=1, encoded_size=self.posdims[1]//2)
        self.W_encoding = rff.layers.GaussianEncoding(alpha=rff_alphas[2], sigma=rff_sigmas[2], input_size=1, encoded_size=self.posdims[2]//2)
        self.T_encoding = rff.layers.GaussianEncoding(alpha=rff_alphas[3], sigma=rff_sigmas[3], input_size=1, encoded_size=self.posdims[3]//2)
        self.time_embedding = SinusoidalEmbedding(self.t_emb_size)
        # self.time_embedding = rff.layers.GaussianEncoding(alpha=1, sigma=2, input_size=1, encoded_size=self.t_emb_size//2)
        self.val_embedding = SinusoidalEmbedding(self.x_emb_size)
        # self.val_embedding = rff.layers.GaussianEncoding(alpha=1, sigma=2, input_size=1, encoded_size=self.x_emb_size//2)

        # self.setup_q_z_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])
    
    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [dim1, rank] [dim2, rank] 需要优化的参数
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)

    def forward(self, val, ind, t, sample=True):
        z = []
        mu = []
        train_ind = ind.float() / self.norm_factors
        mu.append(self.U_Net_mu(self.U_encoding(train_ind[:,0].unsqueeze(-1))))
        mu.append(self.V_Net_mu(self.V_encoding(train_ind[:,1].unsqueeze(-1))))
        mu.append(self.W_Net_mu(self.W_encoding(train_ind[:,2].unsqueeze(-1))))
        mu.append(self.T_Net_mu(self.T_encoding(train_ind[:,3].unsqueeze(-1))))
        for d in range(self.dim):
            if sample:
                # z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))[ind[:, d]]
                # z_d = self.q_z_mu[d][ind[:, d]]
                # z_d = gaussian_repar(mu=mu[d], sigma=sigma[d])
                z_d = mu[d]
            else:
                # z_d = self.q_z_mu[d][ind[:, d]]
                z_d = mu[d]
            z.append(z_d)
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        time_emb = self.time_embedding(t.unsqueeze(-1))
        val_emb = self.val_embedding(val.unsqueeze(-1)).view(val.shape[0], -1) #size [B, 1, 3]->[B, 3*8]

        t_exp = self.layers['t_enc'](time_emb)
        x_exp = self.layers['x_enc'](val_emb)
        z_exp = self.layers['z_enc'](z_ten)

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([x_exp, z_exp, t_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        elif self.pooling_method == 'none':
            hidden = x_exp
        else:
            raise RuntimeError('Wrong pooling method!')

        output = self.layers['output'](hidden).unsqueeze(1) #size [128] 保证energy为正
        output += val
        
        return output


class ScalarINRnD(nn.Module): #先将坐标随机投影到单位圆，再分别过INR
    def __init__(
        self, out_dim, tensor_shape, rank, h_dim, 
        act, dropout, latent_dim, x_emb_size, t_emb_size, 
        sigma_func, pooling_method, skip_connection, posdim, 
        rff_alpha, rff_sigma):
        super(ScalarINRnD, self).__init__()
        self.out_dim = out_dim
        self.tensor_shape = list(tensor_shape)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('norm_factors', torch.tensor(self.tensor_shape, dtype=torch.float32))

        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        self.t_emb_size = t_emb_size
        self.posdim = posdim
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)

        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        
        # x encoder
        x_enc = MLP(input_dim=self.out_dim*self.x_emb_size, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # t encoder
        t_enc = MLP(input_dim=self.t_emb_size, output_dim=self.latent_dim, 
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        
        # ouput layer
        if self.pooling_method in ['sum', 'attn', 'none']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 3 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(input_dim=in_size, output_dim=self.out_dim, 
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 't_enc': t_enc, 'output': output_layer})
        
        self.one_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.two_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)
        self.three_Net_mu = MLP(input_dim=self.posdim, output_dim=self.rank,
                        h_dim=self.h_dim, act=act, dropout=self.dropout,
                        bn=False, wn=False, sn=False, 
                        skip_connection=self.skip_connection)

        self.encoding = rff.layers.GaussianEncoding(alpha=rff_alpha, sigma=rff_sigma, input_size=1, encoded_size=self.posdim//2)
        self.time_embedding = SinusoidalEmbedding(self.t_emb_size)
        # self.time_embedding = rff.layers.GaussianEncoding(alpha=1, sigma=2, input_size=1, encoded_size=self.t_emb_size//2)
        self.val_embedding = SinusoidalEmbedding(self.x_emb_size)
        # self.val_embedding = rff.layers.GaussianEncoding(alpha=1, sigma=2, input_size=1, encoded_size=self.x_emb_size//2)

        # self.setup_q_z_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])
    
    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [dim1, rank] [dim2, rank] 需要优化的参数
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)

    def forward(self, val, ind, t, sample=True):
        z = []
        mu = []
        train_ind = ind.float() / self.norm_factors
        mu.append(self.one_Net_mu(self.encoding(train_ind[:,0].unsqueeze(-1))))
        mu.append(self.two_Net_mu(self.encoding(train_ind[:,1].unsqueeze(-1))))
        mu.append(self.three_Net_mu(self.encoding(train_ind[:,2].unsqueeze(-1))))
        for d in range(self.dim):
            if sample:
                # z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))[ind[:, d]]
                # z_d = self.q_z_mu[d][ind[:, d]]
                # z_d = gaussian_repar(mu=mu[d], sigma=sigma[d])
                z_d = mu[d]
            else:
                # z_d = self.q_z_mu[d][ind[:, d]]
                z_d = mu[d]
            z.append(z_d)
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        time_emb = self.time_embedding(t.unsqueeze(-1))
        val_emb = self.val_embedding(val.unsqueeze(-1)).view(val.shape[0], -1) #size [B, 1, 3]->[B, 3*8]

        t_exp = self.layers['t_enc'](time_emb)
        x_exp = self.layers['x_enc'](val_emb)
        z_exp = self.layers['z_enc'](z_ten)

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([x_exp, z_exp, t_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        elif self.pooling_method == 'none':
            hidden = x_exp
        else:
            raise RuntimeError('Wrong pooling method!')

        output = self.layers['output'](hidden).unsqueeze(1) #size [128] 保证energy为正
        output += val
        
        return output