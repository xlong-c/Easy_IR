import time
import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

from tqdm import tqdm


from utils.Logger import Auto_Logger
from utils.tools import use_prefetch_generator
from utils.get_parts import get_dataset
from utils.utils_dist import get_dist_info

import model


class Trainer():
    def __init__(self, opts):

        self.train_opts = opts['train']
        self.data_opts = opts['data']
        self.seed = opts['train']['seed']
        self.set_seed()
        self.model = model.MODEL(opts)
        self.logger = Auto_Logger(path=opts['save']['dir'],
                                  log_types=['train', 'test', 'valid'],
                                  On_tensorboard=opts['save']['On_tensorboard'])
        self.data_loader = OrderedDict()
        self.load()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def load_logger(self):
        time_rule = 'tooks: {:.2f}S'
        G_loss = self.train_opts['G_net']['Loss_fn']['loss']
        loss_rule = []
        for loss in G_loss:
            loss_rule.append(loss + '  {:>10.4f}')
        loss_rule = '  '.join(loss_rule)
        epoch_num = '{:0>3}'.format(self.train_opts['num_epoch'])
        train_rule = 'EPOCH: '+epoch_num + \
            '/{:0>3} LOSS: {:>10.4f} ' + loss_rule + time_rule

        

        self.logger.define_log_rule(
            'train',
            train_rule
        )
        self.logger.define_log_rule(
            'test',
            train_rule
        )
        self.logger.define_log_rule(
            'valid',
            train_rule
        )
        
        pass

    def load(self):
        self.rank, self.world_size = get_dist_info()
        self.model.load_param(self.save_opts['resume_lable'])
        self.data_loader['train'] = self.get_dataloader(data_path=self.data_opts['trainset_path'],
                                                        mode='train',
                                                        prefetch_generator=self.data_opts['prefetch_generator'],
                                                        to_bad_fn_param=self.data_opts['to_bad_fn_param'],
                                                        dataLoader_param=self.data_opts['data_loader_param'])
        self.data_loader['test'] = self.get_dataloader(data_path=self.data_opts['testset_path'],
                                                       mode='test',
                                                       prefetch_generator=self.data_opts['prefetch_generator'],
                                                       to_bad_fn_param=self.data_opts['to_bad_fn_param'],
                                                       dataLoader_param=self.data_opts['data_loader_param'])

        self.data_loader['valid'] = self.get_dataloader(data_path=self.data_opts['validset_path'],
                                                        mode='valid',
                                                        prefetch_generator=self.data_opts['prefetch_generator'],
                                                        to_bad_fn_param=self.data_opts['to_bad_fn_param'],
                                                        dataLoader_param=self.data_opts['data_loader_param'])
        self.load_logger()

    def get_dataloader(self, data_path, mode,  prefetch_generator, to_bad_fn_param, dataLoader_param):
        dataset = get_dataset(data_path, mode, to_bad_fn_param)
        DataLoaderX = use_prefetch_generator(prefetch_generator,
                                             self.data_opts['data_loader_param']['pin_memory'])
        dataloader = DataLoaderX(dataset, shuffle=(
            mode == 'train'), **dataLoader_param)

        if self.train_opts['dist'] and mode is 'train':
            self.train_sampler = DistributedSampler(dataset,
                                                    shuffle=self.data_opts['shuffle'],
                                                    drop_last=self.data_opts['shuffle'],
                                                    seed=self.seed)
            loader = dataloader(dataset,
                                batch_size=self.data_opts['batch_size'],
                                shuffle=False,
                                num_workers=self.data_opts['num_workers'],
                                drop_last=self.data_opts['drop_last'],
                                pin_memory=self.data_opts['pin_memory'],
                                sampler=self.train_sampler)
        else:
            loader = dataloader(dataset,
                                batch_size=self.data_opts['batch_size'],
                                shuffle=self.data_opts['shuffle'],
                                num_workers=self.data_opts['num_workers'],
                                drop_last=self.data_opts['drop_last'],
                                pin_memory=self.data_opts['pin_memory'],)
        return loader

    def train(self):
        best = 0
        for epoch in range(self.model.start_epoch, self.train_opts['num_epoch']):
            if self.train_opts['dist']:
                self.train_sampler.set_epoch(epoch)
            pass

    def train_a_epoch(self):

        loop = tqdm(enumerate(self.data_loader['train']),
                    total=len(self.data_loader['train']),
                    ncols=self.train_opts['ncols'])

        self.model.train()
        for idx, train_data in loop:
            step_time = time.time()
            self.model.feed_data(train_data)
            self.model.optim_parameters()
            self.logger.log()
