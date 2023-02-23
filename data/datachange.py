import os
from scipy.io import loadmat
import torch


def apply_mask(x, mask):
    """
    输入为[-1，1]实数图像矩阵
    :param x:
    :param mask:采样方式
    :return:
    """
    x = (x + 1.) / 2.  # 映射至0-1
    fft = torch.fft.fft2(x, norm='ortho')  # 2D傅里叶变换
    fft = torch.fft.fftshift(fft)  # 将四周分散的信号移至中间
    fft = fft * mask  # 采样方法  mask为0、1矩阵
    fft = torch.fft.ifftshift(fft)  #
    x = torch.fft.ifft2(fft, norm='ortho')  # 反傅里叶变换
    x = torch.abs(x)  #
    x = x * 2 - 1
    return x


def loadmask(mask_name, mask_perc, cfg):
    if mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join(cfg.TRAIN.mask_Gaussian2D_path,
                             "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(cfg.TRAIN.mask_Gaussian1D_path,
                             "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(cfg.TRAIN.mask_Poisson2D_path,
                             "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))
    return mask


def get_mask(path, mask_type, mask_level):
    if mask_type == "gaussian2d":
        path = os.path.join(path, 'Gaussian2D')
        tname = 'GaussianDistribution2DMask_'
        mmn = 'maskRS2'
    elif mask_type == "gaussian1d":
        path = os.path.join(path, 'Gaussian1D')
        tname = 'GaussianDistribution1DMask_'
        mmn = 'maskRS1'
    elif mask_type == "poisson2d":
        path = os.path.join(path, 'Poisson2D')
        tname = 'PoissonDistributionMask_'
        mmn = 'population_matrix'
    else:
        raise ValueError("no such mask exists: {}".format(mask_type))
    path = os.path.join(path, f'{tname}{mask_level}.mat')
    mask = loadmat(path)[mmn]
    return mask
