import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

netG = nn.Conv2d(3, 3, 1, 1, 1)
netD = nn.Sequential(nn.Conv2d(3, 1, 1, 1, 0), nn.AdaptiveAvgPool2d(1))
lossfn = nn.L1Loss()
optimizerD = Adam()
device ='cuda'

batch_size = 1
x = torch.randn((1, 3, 4, 4))
nz= 100
num_timesteps =4
lazy_reg =15
# 更新D
for p in netD.parameters():
    p.requires_grad = True
netD.zero_grad()
# 不更新G
for p in netG.parameters():
    p.requires_grad = False

# t时刻采样
t = torch.randint(0, num_timesteps,
                  (real_data.size(0),), device=device)
x_t, x_tp1 = q_sample_pairs(coeff, x, t)
x_t.requires_grad = True

# train with real
D_real = netD(x_t, t, x_tp1.detach()).view(-1)
errD_real = F.softplus(-D_real).mean()

errD_real.backward(retain_graph=True)

if lazy_reg is None:
    grad_penalty_call(args, D_real, x_t)
else:
    if global_step % args.lazy_reg == 0:
        grad_penalty_call(args, D_real, x_t)

# train with fake
latent_z = torch.randn(batch_size, nz, device=device)
x_0_predict = netG(x_tp1.detach(), t, latent_z)
x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
errD_fake = F.softplus(output).mean()

errD_fake.backward()

errD = errD_real + errD_fake
# Update D
optimizerD.step()

# update G
for p in netD.parameters():
    p.requires_grad = False
netG.zero_grad()

t = torch.randint(0, num_timesteps,
                  (real_data.size(0),), device=device)

x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

latent_z = torch.randn(batch_size, nz, device=device)

x_0_predict = netG(x_tp1.detach(), t, latent_z)
x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

output = netD(x_pos_sample, t, x_tp1.detach())

if isinstance(output, tuple):
    Gg, Gp = output
    # errG = F.softplus(-Gg).mean() + F.softplus(-Gp.view(-1)).mean()
    errG = F.softplus(-Gp.view(-1)).mean()
else:
    errG = F.softplus(-output)
    errG = errG.mean()

errG.backward()
optimizerG.step()



def grad_penalty_call(args, D_real, x_t):
    # 来自文章 Improved Training of Wasserstein GANs
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()


def sample(x, t):
    return x


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * \
                     (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one
