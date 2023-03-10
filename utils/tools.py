import os

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


def mk_dirs(path, verbose=True):
    if not os.path.exists(path):
        if verbose:
            print("[OK] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!!] %s exists ..." % path)


def use_prefetch_generator(on_prefetch_generator, pin_memory):
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
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = r1_gamma / 2 * grad_penalty
    grad_penalty.backward()
