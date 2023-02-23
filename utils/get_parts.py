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
    losses = []
    for loss in loss_name:
        losses.append(get_loss_(loss['name']))
        print(f'[OK] 损失函数{loss} 启用')
    return {'loss': losses, 'loss_weight': loss_weight, 'lose_name': loss_name}