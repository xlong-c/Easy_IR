import numpy as np
import torch

from utils.tools import add_mask


def extract(input, t, shape):
    """
    获取input在t位置的数据展开为shape

    """
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def var_func_geometric(t, beta_min, beta_max):
    """
    方差数据
    """
    return beta_min * ((beta_max / beta_min) ** t)


def var_func_vp(t, beta_min, beta_max):
    """
    vp方差
    """
    log_mean_coeff = -0.25 * t ** 2 * \
                     (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def get_sigma_schedule(num_timesteps, beta_min, beta_max, use_geometric, device):
    """
    Args:
        num_timesteps (int): number of timesteps
        beta_max (float): maximum of beta
        beta_min (float): minimum of beta
        use_geometric (bool):
        device (str): device
    获取噪声强度相关数值

    """
    eps_small = 1e-3

    t = np.arange(0, num_timesteps + 1, dtype=np.float64)
    t = t / num_timesteps
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if use_geometric:
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
    """
    获取正向扩散各种系数    (训练)
    """

    def __init__(self, num_timesteps, beta_min, beta_max, use_geometric, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(num_timesteps, beta_min, beta_max, use_geometric, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


class Posterior_Coefficients():
    """
    获取反向扩散各种系数    (推理)
    """

    def __init__(self, num_timesteps, beta_min, beta_max, use_geometric, device):
        _, _, self.betas = get_sigma_schedule(num_timesteps, beta_min, beta_max, use_geometric, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32,
                          device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * \
                                  (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))


def get_time_schedule(num_timesteps, device):
    """
    获取步数系数
    """
    n_timestep = num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


class Diffusion():
    # 扩散模型组件,x_start(x_0)是无噪声的,x_t则加入了强度为(t/t_total)的噪声
    def __init__(self, opts):
        self.diff_opts = opts['diffusion']

        num_timesteps = self.diff_opts['num_timesteps']
        use_geometric = self.diff_opts['use_geometric']
        beta_min = self.diff_opts['beta_min']
        beta_max = self.diff_opts['beta_max']

        self.device = torch.device('cuda' if opts['train']['gpu_ids'] is not None else 'cpu')
        self.coeff = Diffusion_Coefficients(num_timesteps, beta_min, beta_max, use_geometric, self.device)  # 扩散参数 正
        self.pos_coeff = Posterior_Coefficients(num_timesteps, beta_min, beta_max, use_geometric, self.device)  # 扩散参数 反
        self.T = get_time_schedule(self.diff_opts['num_timesteps'], self.device)

    def get_T(self, size):
        # 生成时刻T
        return torch.randint(low=0, high=self.diff_opts['num_timesteps'],
                             size=size, device=self.device)

    def get_noise(self, xx, mask=None):
        noise = torch.randn_like(xx, device=xx.device)
        noise = torch.fft.fft2(noise)
        noise = torch.fft.fftshift(noise)
        if mask is not None:
            noise = noise * mask
        noise = torch.fft.ifftshift(noise)
        noise = torch.fft.ifft2(noise)
        noise = torch.real(noise)
        import torchvision.utils as vutils
        vutils.save_image(noise.squeeze(0), 'output.png')
        return noise

    def q_sample(self, x_start, t, mask, *, noise=None):
        """
        Diffuse the data (t == 0 means diffused for t step)
        生成t时刻的数据
        """
        if noise is None:
            noise = self.get_noise(x_start, mask)

        x_t = extract(self.coeff.a_s_cum, t, x_start.shape) * x_start + \
              extract(self.coeff.sigmas_cum, t, x_start.shape) * noise
        return x_t

    def q_sample_pairs(self, x_start, t, mask):
        """
           Generate a pair of disturbed images for training
           :param x_start: x_0
           :param t: time step t
           :return: x_t, x_{t+1}
           生成t t+1 数据
           """
        noise = self.get_noise(x_start, mask=mask)
        x_t = self.q_sample(x_start=x_start, t=t, mask=mask)
        x_t_plus_one = extract(self.coeff.a_s, t + 1, x_start.shape) * x_t + \
                       extract(self.coeff.sigmas, t + 1, x_start.shape) * noise
        return x_t, x_t_plus_one

    def q_posterior(self, x_0, x_t, t):
        """
         搜索逆向扩散t时刻参数
         """
        mean = (
                extract(self.pos_coeff.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(self.pos_coeff.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.pos_coeff.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            self.pos_coeff.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(self, x_0, x_t, t, mask):
        """
       生成T-1的数据
       """
        mean, _, log_var = self.q_posterior(x_0, x_t, t)

        noise = self.get_noise(x_t, mask)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    def sample_posterior(self, x_0, x_t, t, mask):
        """
        生成T-1的数据
        """
        sample_x_pos = self.p_sample(x_0, x_t, t, mask)

        return sample_x_pos

    def sample_from_model(self, generator, n_time, x_init, L, mask):
        """
        扩散样本(干净)生成
        """
        x = x_init
        with torch.no_grad():
            for i in reversed(range(n_time)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                t_time = t
                x_0, rec_loss_ = generator(x, L, t_time)
                x_new = self.sample_posterior(x_0, x, t, mask)
                x = x_new.detach()
        return x
