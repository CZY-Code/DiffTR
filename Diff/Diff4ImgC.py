import os
import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from einops import reduce
from accelerate import Accelerator, DistributedDataParallelKwargs
from ema_pytorch import EMA
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity
import numpy as np
from PIL import Image

from .diff_utils import *
from .networks import ScalarINR2D
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions
def normalize_to_neg_one_to_one(val):
    return val * 2 - 1

def unnormalize_to_zero_to_one(val):
    return (val + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        channels = None,
        self_condition = None,
        channel_first = True
    ):
        super().__init__()
        self.model = model

        self.channels = default(channels, lambda: self.model.channels)
        self.self_condition = default(self_condition, lambda: self.model.self_condition)

        self.channel_first = channel_first
        self.seq_index = -1 if channel_first else -2 #[B,L,C]
        self.seq_length = seq_length
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, ind, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, model_forward_kwargs: dict = dict()):
        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        model_output = self.model(x, ind, t, **model_forward_kwargs)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, ind, t, x_self_cond = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        preds = self.model_predictions(x, ind, t, model_forward_kwargs = model_forward_kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, data_ind, t: int, x_self_cond = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, ind = data_ind, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, model_forward_kwargs = model_forward_kwargs)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, data_ind, shape, return_noise = False, model_forward_kwargs: dict = dict()):
        batch, device = shape[0], self.betas.device
        noise = torch.randn(shape, device=device)
        data_val = noise 
        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            data_val, x_start = self.p_sample(data_val, data_ind, t, self_cond, model_forward_kwargs = model_forward_kwargs)

        data_val = self.unnormalize(data_val)

        if not return_noise:
            return data_val

        return data_val, noise

    @torch.no_grad()
    def ddim_sample(self, data_ind, shape, clip_denoised = True, model_forward_kwargs: dict = dict(), return_noise = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        noise = torch.randn(shape, device = device)
        data_val = noise
        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(data_val, data_ind, time_cond, self_cond, 
                                                             clip_x_start = clip_denoised, 
                                                             model_forward_kwargs = model_forward_kwargs)

            if time_next < 0:
                data_val = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(data_val)
            data_val = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        data_val = self.unnormalize(data_val)

        if not return_noise:
            return data_val

        return data_val, noise

    @torch.no_grad()
    def sample(self, data_ind, batch_size = 16, return_noise = False, model_forward_kwargs: dict = dict()):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop

        shape = (batch_size, channels, seq_length) if self.channel_first else (batch_size, seq_length, channels)
        return sample_fn(data_ind, shape, return_noise = return_noise, model_forward_kwargs = model_forward_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, data_val, data_ind, t, noise = None, model_forward_kwargs: dict = dict(), return_reduced_loss = True):
        b = data_val.shape[0]
        n = data_val.shape[self.seq_index]
        noise = default(noise, lambda: torch.randn_like(data_val))

        # noise sample
        noise_val = self.q_sample(x_start = data_val, t = t, noise = noise)

        # predict and take gradient step
        ScalarINR2D.forward
        model_out = self.model(noise_val, data_ind, t, **model_forward_kwargs)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = data_val
        elif self.objective == 'pred_v':
            v = self.predict_v(data_val, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        # loss = F.l1_loss(model_out, target, reduction = 'none') + F.mse_loss(model_out, target, reduction = 'none')

        if not return_reduced_loss:
            return loss * extract(self.loss_weight, t, loss.shape)

        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, data_val, data_ind, *args, **kwargs):
        b, n, device, seq_length, = data_val.shape[0], data_val.shape[self.seq_index], data_val.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        data_val = self.normalize(data_val)

        return self.p_losses(data_val, data_ind, t, *args, **kwargs)


# trainer class
class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        trainset: Dataset,
        testset: Dataset,
        *,
        gt_tensor=None, 
        ori_mask=None, 
        visible_ratio=None, 
        is_tube=False,
        datatype=None, 
        showChannel=None,
        dataname=None,
        ###############
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        adam_betas = (0.9, 0.99),
        weight_decay = 0.01,
        scheduler = 'cosine',
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        save_and_sample_every = 1000,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        norm_type = 'inf'
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.norm_type = norm_type
        self.train_num_steps = train_num_steps

        # dataset and dataloader
        trainset_dl = DataLoader(trainset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        trainset_dl = self.accelerator.prepare(trainset_dl)
        self.trainset_dl = cycle(trainset_dl)

        self.test_dl = self.accelerator.prepare(DataLoader(testset, batch_size = 8192, shuffle = False, 
                                                           pin_memory = True, num_workers = 4))

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas, weight_decay=weight_decay)
        if scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt, T_max=train_num_steps, eta_min=1e-6)
        else:
            raise RuntimeError('Wrong scheduler!')

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # step counter state
        self.step = 0
        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.scheduler)
        self.dataname = dataname
        self.visible_ratio = visible_ratio
        self.is_tube = is_tube
        self.datatype = datatype
        self.showChannel = showChannel
        self.best_metric = {'psnr': 0.0, 'ssim': 0.0, 'nrmse': 0.0}

        self.gt_tensor = gt_tensor.cpu().numpy()
        if self.is_tube == True:
            if self.datatype == 'RGB':
                self.ob_tensor = (gt_tensor * (1 - ori_mask[:,:,None])).to(self.device)
            elif self.datatype == 'MSI':
                self.ob_tensor = (gt_tensor * (1 - ori_mask[:,:,None])).to(self.device)
            elif self.datatype == 'y4m':
                self.ob_tensor = (gt_tensor * (1 - ori_mask[:,:,:,None])).to(self.device)
        else:
            self.ob_tensor = (gt_tensor * (1 - ori_mask)).to(self.device)
            
        self.count_tensor = (1 - ori_mask).to(self.device)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data_ind, data_val = next(self.trainset_dl)

                    with self.accelerator.autocast():
                        GaussianDiffusion1D.forward
                        loss = self.model(data_val, data_ind)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                self.accelerator.wait_for_everyone()
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, self.norm_type)

                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()

                self.accelerator.wait_for_everyone()

                self.step += 1
                if self.accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            self.test()
                        self.save(milestone)

                pbar.update(1)
        self.accelerator.print('training complete')

    def test(self):
        rec_tensor = self.ob_tensor.clone()
        rec_counts = self.count_tensor.clone()
        
        for data_ind, data_val in tqdm(self.test_dl, disable = not self.accelerator.is_main_process):
            GaussianDiffusion1D.sample
            x_hat = self.ema.ema_model.sample(data_ind, batch_size=data_val.shape[0], model_forward_kwargs = {"sample":False}) #[2048, 1, 3]
            
            if self.is_tube == False:
                if self.datatype == 'RGB':
                    batch_tensor, batch_counts = reconstruct_to_tensor_3D(x_hat, data_ind, original_shape=self.model.model.tensor_shape)
                elif self.datatype in ['y4m_V3', 'MSI']:
                    batch_tensor, batch_counts = reconstruct_to_tensor_3D(x_hat, data_ind, original_shape=self.model.model.tensor_shape)
                elif self.datatype in ['y4m', 'LF']:
                    batch_tensor, batch_counts = reconstruct_to_tensor_4D(x_hat, data_ind, original_shape=self.model.model.tensor_shape)

            elif self.is_tube == True:
                if self.datatype == 'RGB':
                    batch_tensor, batch_counts = reconstruct_to_tensor_2D_tube(x_hat, data_ind, original_shape=self.model.model.tensor_shape, C=3)
                elif self.datatype == 'MSI':
                    batch_tensor, batch_counts = reconstruct_to_tensor_2D_tube(x_hat, data_ind, original_shape=self.model.model.tensor_shape, C=31)
                elif self.datatype == 'y4m': #(288, 352, 10, 3)
                    batch_tensor, batch_counts = reconstruct_to_tensor_4D_tube(x_hat, data_ind, original_shape=self.model.model.tensor_shape, C=3)
            
            else:
                raise RuntimeError('Wrong tube type!')

            rec_tensor += batch_tensor
            rec_counts += batch_counts

        if self.is_tube == True:
            rec_counts = rec_counts.unsqueeze(-1)

        final_rec = torch.where(rec_counts > 0, rec_tensor / rec_counts.clamp(min=1), torch.zeros_like(rec_tensor))
        final_rec = np.clip(final_rec.cpu().detach().numpy(), 0, 1)

        if self.datatype in ['y4m', 'y4m_V2']:
            ssim = 0.0
            psnr = 0.0
            nrmse = 0.0
            for n in range(final_rec.shape[2]):
                psnr += peak_signal_noise_ratio(self.gt_tensor[:, :, n], final_rec[:, :, n], data_range=1.0)
                ssim += structural_similarity(self.gt_tensor[:, :, n], final_rec[:, :, n], data_range=1.0, channel_axis=2)
                nrmse += normalized_root_mse(self.gt_tensor[:, :, n], final_rec[:, :, n])
            ssim /= final_rec.shape[2]
            psnr /= final_rec.shape[2]
            nrmse /= final_rec.shape[2]
        elif self.datatype == 'y4m_V3':
            final_rec = final_rec.reshape(288, 352, 10, 3)
            gt_tensor = self.gt_tensor.reshape(288, 352, 10, 3)
            ssim = 0.0
            psnr = 0.0
            nrmse = 0.0
            for n in range(final_rec.shape[2]):
                psnr += peak_signal_noise_ratio(gt_tensor[:, :, n], final_rec[:, :, n], data_range=1.0)
                ssim += structural_similarity(gt_tensor[:, :, n], final_rec[:, :, n], data_range=1.0, channel_axis=2)
                nrmse += normalized_root_mse(gt_tensor[:, :, n], final_rec[:, :, n])
            ssim /= final_rec.shape[2]
            psnr /= final_rec.shape[2]
            nrmse /= final_rec.shape[2]
        elif self.datatype == 'LF':
            final_rec = final_rec.reshape(128, 128, 3, 30)
            gt_tensor = self.gt_tensor.reshape(128, 128, 3, 30)
            ssim = 0.0
            psnr = 0.0
            nrmse = 0.0
            for n in range(final_rec.shape[-1]):
                psnr += peak_signal_noise_ratio(gt_tensor[..., n], final_rec[..., n], data_range=1.0)
                ssim += structural_similarity(gt_tensor[..., n], final_rec[..., n], data_range=1.0, channel_axis=2)
                nrmse += normalized_root_mse(gt_tensor[..., n], final_rec[..., n])
            ssim /= final_rec.shape[-1]
            psnr /= final_rec.shape[-1]
            nrmse /= final_rec.shape[-1]
        else:
            psnr = peak_signal_noise_ratio(self.gt_tensor, final_rec, data_range=1.0)
            ssim = structural_similarity(self.gt_tensor, final_rec, data_range=1.0, channel_axis=2)
            nrmse = normalized_root_mse(self.gt_tensor, final_rec)

        print(f'dataname: {self.dataname}, SR: {self.visible_ratio}, psnr: {psnr:.4f}, ssim: {ssim:.4f}, nrmse: {nrmse:.4f}')
        
        if self.showChannel is not None:
            output_folder = os.path.join('output/Ours/inpainting', self.datatype)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, self.dataname + f'_SR{self.visible_ratio:.2f}_psnr{psnr:.3f}_inpainting.png')
            
            if self.datatype in ['y4m', 'y4m_V2', 'y4m_V3']:
                final_rec = final_rec.transpose(0, 1, 3, 2)  # (H, W, T, C) -> (H, W, C, T)

            img = Image.fromarray((final_rec* 255).astype(np.uint8)[..., self.showChannel])
            img.save(output_path)

        with open(f'imgC_{self.datatype}_result.txt', 'a+') as file:
            file.write(f'dataname: {self.dataname}, SR: {self.visible_ratio}, psnr: {psnr:.4f}, ssim: {ssim:.4f}, nrmse: {nrmse:.4f}' + '\n')
        
        if psnr > self.best_metric['psnr']:
            self.best_metric['psnr'] = float(psnr)
            self.best_metric['ssim'] = float(ssim)
            self.best_metric['nrmse'] = float(nrmse)
            
            

