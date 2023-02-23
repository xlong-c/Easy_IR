import os
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

def use_prefetch_generator(on_prefetch_generator,pin_memory):
    if on_prefetch_generator:
        assert pin_memory, '未开启内存锁页！！'
        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())
        print('构建多线程迭代')
    else:
        DataLoaderX = DataLoader
    return DataLoaderX