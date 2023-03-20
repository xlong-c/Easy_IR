import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


def mk_dirs(path, verbose=True):
    """
    创建目录
    """
    # 如果路径不存在，则创建目录
    if not os.path.exists(path):
        if verbose:
            print("[OK] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        # 如果路径已存在，则打印警告信息
        if verbose:
            print("[!!] %s exists ..." % path)


def use_prefetch_generator(on_prefetch_generator, pin_memory):
    """
    使用预取生成器加速数据加载
    """
    if on_prefetch_generator:
        assert pin_memory, '未开启内存锁页！！'

        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        print('构建多线程迭代')
    else:
        DataLoaderX = DataLoader
    return DataLoaderX



def grad_penalty_call(r1_gamma, D_real, x_t):
    """
    用一种梯度惩罚的方法替换 clip weights
    """
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = r1_gamma / 2 * grad_penalty
    grad_penalty.backward()


def init_network(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    '[ERROR] Initialization method [%s] is not implemented' % init_type
                )
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('LayerNorm') != -1 or classname.find('GroupNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('[INFO] 初始化网络权重')
    net.apply(init_func)


def reverse_matrix(matrix):
    """01矩阵取反
    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.abs(matrix - 1)


def add_mask(matrix, mask):
    """
    将mask应用于矩阵
    """
    mask = torch.tensor(mask, dtype=torch.float32)
    matrix = torch.fft.fft2(matrix)
    matrix = torch.fft.fftshift(matrix)
    matrix = matrix * mask
    matrix = torch.fft.ifftshift(matrix)
    matrix = torch.fft.ifft2(matrix)
    matrix = torch.abs(matrix)
    return matrix
