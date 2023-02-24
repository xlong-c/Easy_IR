import importlib
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from data.dataset_mri import Brain_data
from prefetch_generator import BackgroundGenerator
from data.to_bad_mothond import mri_mask
from utils.data_aug import Augmentation


def get_model(model_name, model_args=None):
    return nn.Sequential(
        nn.Conv2d(1, 1, 1, 1, 0), nn.ReLU(), nn.Conv2d(1, 1, 1, 1, 0)

    )


def get_loss_(loss):
    if loss == 'L1loss':
        return nn.L1Loss()
    elif loss == 'MSELoss':
        return nn.MSELoss()
    else:
        return importlib.import_module('Loss_Metrics', loss['name'])()


def get_loss(loss_name: list, loss_weight: list, is_Metric=True):
    assert len(loss_name) == len(loss_weight)
    losses = []
    for loss in loss_name:

        if is_Metric:
            losses.append(get_loss_(loss).eval())
            print(f'[OK] 评估函数{loss} 启用')

        else:
            losses.append(get_loss_(loss))
            print(f'[OK] 损失函数{loss} 启用')
    if is_Metric:
        loss_weight = torch.ones(len(loss_weight))
    return {'loss': losses, 'loss_weight': loss_weight, 'lose_name': loss_name}


def get_optimizer(optim_name, network, optim_param):
    optim_params = []
    for k, v in network.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    optimizer = eval(f'optim.{optim_name}')(optim_params, **optim_param)
    return optimizer


def get_schedule(scheduler_name, optimizer, schedule_param):
    scheduler = eval(f'lr_scheduler.{scheduler_name}')(optimizer, **schedule_param)
    return scheduler


def get_datapath(path, mode, train_path, val_path, test_path):
    if mode == 'train':
        data_path = os.path.join(path, train_path)
    elif mode == 'val':
        data_path = os.path.join(path, val_path)
    elif mode == 'test':
        data_path = os.path.join(path, test_path)
    else:
        raise RuntimeError('未知的数据集模式，请拓展接口')
    return data_path


def get_dataset(data_path, mode, to_bad_fn_param):
    to_bad = mri_mask(**to_bad_fn_param)
    transform = [Augmentation]
    dataset = Brain_data(data_path, mode=mode, to_bad_fn=to_bad, transform=transform)
    return dataset
