import torch 
import torch.nn as nn
from collections import OrderedDict

from Easy_IR.utils.get_parts import get_model


class MODEL(nn.Module):
    def __init__(self,opts):
        self.opts = opts
        self.train_opts = opts['train']
    
    def load(self):
        self.model = get_model(self.train_opts['G_net']['network'],self.train_opts['G_net']['params'])
        self.losser = get_losses(self.train_opts['G_net'])
    
    def feed_data(self,sample_batch):
        self.L,self.H = sample_batch
    
    def forward_G(self):
        self.P = self.model(self.L)
    
    
    
    