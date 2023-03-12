import os

from utils.transforms.DWT_IDWT.DWT_IDWT_layer import DWT_2D


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from math import exp

from torch.nn import L1Loss


class NMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen, target):
        if not (target.size() == gen.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), gen.size()),
                          stacklevel=2)
        nmse_a = torch.sqrt(torch.sum(torch.pow((gen - target), 2), dim=[1, 2, 3]))
        nmse_b = torch.sqrt(torch.sum(torch.pow(target, 2), dim=[1, 2, 3]))
        return torch.mean(nmse_a / nmse_b)  # tensor [batch]


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        if window is not None:
            window = window.to(img1.device)
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class PSNR(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNR, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(
            1.0 / (((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8)).mean()


class DisDwtLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DisDwtLoss, self).__init__()
        self.ll_lose = nn.L1Loss()
        self.hh_lose = nn.MSELoss()
        self.DWT = DWT_2D('haar')

    def forward(self, pred, target):
        dwt1 = self.DWT(pred)
        dwt2 = self.DWT(target)
        dwt1_ll, dwt1_lh, dwt1_hl, dwt1_hh = torch.chunk(dwt1, 4, dim=1)
        dwt2_ll, dwt2_lh, dwt2_hl, dwt2_hh = torch.chunk(dwt2, 4, dim=1)
        l1s = self.ll_lose(dwt1_ll, dwt2_ll)
        l2s = self.hh_lose(dwt1_hl, dwt2_hl) + self.hh_lose(dwt1_lh, dwt2_lh) + self.hh_lose(dwt1_hh, dwt2_hh)
        return (l1s + l2s).mean()


class DisDwtLoss_2(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DisDwtLoss_2, self).__init__()
        self.ll_lose = nn.L1Loss()
        self.hh_lose = nn.MSELoss()
        self.DWT = DWT_2D('haar')

    def forward(self, pred, target):
        dwt1 = self.DWT(pred)
        dwt2 = self.DWT(target)
        dwt1_ll, dwt1_lh, dwt1_hl, dwt1_hh = torch.chunk(dwt1, 4, dim=1)
        dwt2_ll, dwt2_lh, dwt2_hl, dwt2_hh = torch.chunk(dwt2, 4, dim=1)
        l1s = self.hh_lose(dwt1_ll, dwt2_ll)
        l2s = self.ll_lose(dwt1_hl, dwt2_hl) + self.ll_lose(dwt1_lh, dwt2_lh) + self.ll_lose(dwt1_hh, dwt2_hh)
        return (l1s + l2s).mean()

