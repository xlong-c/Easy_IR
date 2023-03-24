import importlib
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data.build_dataset import build_dataset
from data.dataset_mri import Brain_data
from data.to_bad_mothond import mri_mask
from utils.data_aug import Augmentation, get_data_transforms
import utils.Loss_Metric as LossMetric
from utils.tools import init_network


def get_model(model_name, model_dir, model_init='kaiming', model_args=None):
    """
    获取模型网络实例
    """
    network = importlib.import_module('.'.join([model_dir, model_name]))
    net = eval('network.get_net(model_args)')
    init_network(net, model_init)
    return net


def get_loss_(loss):
    if loss == 'L1loss':
        return nn.L1Loss()
    elif loss == 'MSELoss':
        return nn.MSELoss()
    else:
        return eval('LossMetric.{0}()'.format(loss))


def get_loss(loss_name: list, loss_weight, is_Metric=False):
    if not is_Metric:
        assert len(loss_name) == len(loss_weight)
    losses = []
    for loss in loss_name:
        if is_Metric:
            losses.append(get_loss_(loss).eval())
            print(f'[OK] 评估函数{loss} 启用')
            loss_weight = torch.ones(len(loss_name))
        else:
            losses.append(get_loss_(loss))
            print(f'[OK] 损失函数{loss} 启用')

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
    print('  '.join(('[OK] 优化器学习率调整策略启用', '衰减率：{}'.format(schedule_param.get("gamma")))))
    scheduler = eval(f'lr_scheduler.{scheduler_name}')(
        optimizer, **schedule_param)
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


def get_dataset(dataset_name,data_path, mode, to_bad_fn_param):
    to_bad = mri_mask(**to_bad_fn_param)
    transform = [Augmentation]
    dataset = build_dataset(dataset_name,data_path, mode=mode,
                            to_bad_fn=to_bad, transform=transform)
    return dataset
