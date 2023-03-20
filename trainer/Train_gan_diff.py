import os
import time
import torch
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

from tqdm import tqdm

from trainer.MODEL_gan_diffusion import DIFFGANMODEL as MODEL
from trainer.Train import Trainer
from trainer.build_model import build_model
from utils.Logger import Auto_Logger
from utils.tools import use_prefetch_generator
from utils.get_parts import get_dataset
from utils.utils_dist import get_dist_info


class Trainer_gan_diff(Trainer):
    def __init__(self, opts):
       super(Trainer_gan_diff, self).__init__(opts)


    def load_logger(self):
        super(Trainer_gan_diff, self).load_logger()
        time_rule = 'tooks: {:.2f}S'
        # G_loss = self.train_opts['G_net']['Loss_fn']['loss']
        # loss_rule = self.logger.get_loss_rule(G_loss)
        loss_rule = 'errG: {:<8.4f}  errD: {:<8.4f}'
        epoch_num = '{:>3}'.format(self.train_opts['num_epoch'])
        train_rule = 'EPOCH: ' + epoch_num + \
                     '/{:>3} step: {:<8} LOSS: {:<8.4f} ' + loss_rule + time_rule + '  lr: {:<8.4f}'
        # epoch idx t_loss losses time lr

        # metric time
        self.logger.define_log_rule(
            'train',
            train_rule
        )

        self.logger.define_writer_rule('train', rule=['errG','errD'])

  

   