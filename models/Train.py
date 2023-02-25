import os
import time
import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

from tqdm import tqdm

from models.model import MODEL
from utils.Logger import Auto_Logger
from utils.tools import use_prefetch_generator
from utils.get_parts import get_dataset
from utils.utils_dist import get_dist_info


class Trainer():
    def __init__(self, opts):

        self.train_opts = opts['train']
        self.data_opts = opts['data']
        self.save_opts = opts['save']
        self.seed = opts['train']['seed']
        self.rank, self.world_size = get_dist_info()
        self.set_seed()
        self.model = MODEL(opts)
        self.logger = Auto_Logger(path=opts['save']['dir'],
                                  log_types=['train', 'test', 'test'],
                                  On_tensorboard=opts['save']['On_tensorboard'])
        self.data_loader = OrderedDict()
        self.trainer_log_dict = OrderedDict()
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
        train_rule = 'EPOCH: ' + epoch_num + \
                     '/{:0>3} LOSS: {:>10.4f} ' + loss_rule + time_rule + 'lr: {:>10.4f}'
        # epoch t_loss losses time lr
        metrices = self.train_opts['Metric']
        metric_rule = []
        for metric in metrices:
            metric_rule.append(metric + '  {:>10.4f}')
        metric_rule = '  '.join(metric_rule)

        test_rule = 'EPOCH: ' + epoch_num + \
                    '/{:0>3} VAL_MODE ' + metric_rule + time_rule + 'lr: {:>10.4f}'
        # epoch metric time lr
        test_rule = 'TEST_MODE ' + metric_rule + time_rule
        # metric time
        self.logger.define_log_rule(
            'train',
            train_rule
        )
        self.logger.define_log_rule(
            'test',
            test_rule
        )
        self.logger.define_log_rule(
            'test',
            test_rule
        )

        self.logger.define_log_rule('train', rule=self.train_opts['G_net']['Loss_fn']['loss'])
        self.logger.define_log_rule('test', rule=self.train_opts['Metric'])
        self.logger.define_log_rule('test', rule=self.train_opts['Metric'])

    def load(self):
        self.model.load_param(self.save_opts['resume_lable'])

        train_dataset_path = os.path.join(self.data_opts['data_path'], self.data_opts['trainset_path'])
        valid_dataset_path = os.path.join(self.data_opts['data_path'], self.data_opts['validset_path'])
        test_dataset_path = os.path.join(self.data_opts['data_path'], self.data_opts['testset_path'])

        self.data_loader['train'] = self.get_dataloader(data_path=train_dataset_path,
                                                        mode='train',
                                                        prefetch_generator=self.data_opts['prefetch_generator'],
                                                        to_bad_fn_param=self.data_opts['to_bad_fn_param'],
                                                        dataLoader_param=self.data_opts['data_loader_param'])
        self.data_loader['valid'] = self.get_dataloader(data_path=valid_dataset_path,
                                                        mode='valid',
                                                        prefetch_generator=self.data_opts['prefetch_generator'],
                                                        to_bad_fn_param=self.data_opts['to_bad_fn_param'],
                                                        dataLoader_param=self.data_opts['data_loader_param'])

        self.data_loader['test'] = self.get_dataloader(data_path=test_dataset_path,
                                                       mode='test',
                                                       prefetch_generator=self.data_opts['prefetch_generator'],
                                                       to_bad_fn_param=self.data_opts['to_bad_fn_param'],
                                                       dataLoader_param=self.data_opts['data_loader_param'])
        self.set_len = OrderedDict()
        self.set_len['train'] = len(self.data_loader['train'])
        self.set_len['valid'] = len(self.data_loader['valid'])
        self.set_len['test'] = len(self.data_loader['test'])
        self.load_logger()

    def get_dataloader(self, data_path, mode, prefetch_generator, to_bad_fn_param, dataLoader_param):
        dataset = get_dataset(data_path, mode, to_bad_fn_param)
        DataLoaderX = use_prefetch_generator(prefetch_generator,
                                             self.data_opts['data_loader_param']['pin_memory'])
        dataloader = DataLoaderX(dataset, shuffle=(
                mode == 'train'), **dataLoader_param)

        if self.train_opts['dist'] and mode == 'train':
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
                                pin_memory=self.data_opts['pin_memory'], )
        return loader

    def train(self):
        best = 0

        for epoch in range(self.model.start_epoch, self.train_opts['num_epoch']):
            if self.train_opts['dist']:
                self.train_sampler.set_epoch(epoch)
            last_niter = epoch * self.set_len['train']
            self.train_a_epoch(epoch, last_niter)

    def val_a_epoch(self, epoch):
        loop = tqdm(enumerate(self.data_loader['test']),
                    total=self.set_len['test'],
                    ncols=self.train_opts['ncols'])
        Metric_detail = np.zeros_like(self.train_opts['Metric'])
        step_time = time.time()
        for idx, val_data in loop:
            self.model.feed_data(val_data)
            self.model.test_forward()
            log = self.model.get_log_dict()
            Metric_detail += log['Metric_detail']
            loop.set_description('VALIDING')
            loop.set_postfix(Metric=Metric_detail[0])

        Metric_detail = Metric_detail / self.set_len['test']
        log_msg = [epoch,
                   *Metric_detail,
                   time.time() - step_time,
                   self.model.get_log_dict()['lr']
                   ]  # epoch metric time lr
        self.logger.rule_log('test', log_msg)
        self.logger.rule_writer_log(
            'valid',
            Metric_detail,
            epoch
        )

    def train_a_epoch(self, epoch, last_niter):
        loop = tqdm(enumerate(self.data_loader['train']),
                    total=self.set_len['train'],
                    ncols=self.train_opts['ncols'])
        self.model.train()
        for idx, train_data in loop:
            step_time = time.time()
            self.model.feed_data(train_data)
            self.model.optim_parameters()
            log = self.model.get_log_dict()
            log_msg = [epoch,
                       log['G_loss'],
                       *log['G_loss_detail'],
                       time.time() - step_time,
                       log['lr']
                       ]  # epoch t_loss losses time lr
            self.logger.rule_log('train', log_msg)
            self.logger.rule_writer_log(
                'train',
                log['G_loss_detail'],
                idx + last_niter
            )
            loop.set_description(f'Epoch [{epoch}/{self.train_opts["num_epoch"]}]')
            loop.set_postfix(loss=float(log['G_loss']))

    def test(self):
        loop = tqdm(enumerate(self.data_loader['test']),
                    total=self.set_len['test'],
                    ncols=self.train_opts['ncols'])
        test_time = time.time()
        Metric_detail_avg = []
        for idx, val_data in loop:
            step_time = time.time()
            self.model.feed_data(val_data)
            self.model.test_forward()
            log = self.model.get_log_dict()
            Metric_detail = log['Metric_detail']
            Metric_detail_avg.append(Metric_detail)
            loop.set_description('TESTING')
            loop.set_postfix(Metric=Metric_detail[0])

            log_msg = [
                *Metric_detail,
                time.time() - step_time,
            ]  # metric time

            self.logger.rule_log('test', log_msg)

            self.logger.rule_writer_log(
                'test',
                Metric_detail,
                idx
            )
        self.logger.log('test', 'THE LAST RESULT')

        Metric_detail_avg = sum(Metric_detail_avg) / self.set_len['test']
        log_msg = [
            *Metric_detail_avg,
            time.time() - test_time,
        ]  # metric time
        self.logger.rule_log('test', log_msg)
