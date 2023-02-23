import importlib
import torch.nn as nn

def get_model(model_name,model_args=None):
    
    return nn.Sequential(
        nn.Conv2d(1,1,1,1,0),nn.ReLU(),nn.Conv2d(1,1,1,1,0)
        
    )

def get_loss_(loss):
    if loss == 'L1loss':
        return nn.L1Loss()
    elif loss == 'MSELoss':
        return nn.MSELoss()
    else:
        return importlib.import_module('Loss_Metrics', loss['name'])()

def get_loss(loss_name:list,loss_weight:list):
    assert len(loss_name) == len(loss_weight)
    losses = []
    for loss in loss_name:
        losses.append(get_loss_(loss['name']))
        print(f'[OK] 损失函数{loss} 启用')
    return {'loss': losses, 'loss_weight': loss_weight, 'lose_name': loss_name}

def get_optimizer(optim, network, opts):
    optim_params = []
    for k, v in network.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    optimizer = importlib.import_module('torch.optim', optim)
    return optimizer(optim_params, opts)

def get_schedule(scheduler, optimizer, opts):
    sch = importlib.import_module('torch.optim.lr_scheduler', scheduler)

    return sch(optimizer, opts)
