import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils.EMA import EMA
from utils.get_parts import get_model, get_loss, get_optimizer, get_schedule
from utils.tools import mk_dirs


class MODEL(nn.Module):

    def __init__(self, opts):
        super(MODEL, self).__init__()
        self.opts = opts
        self.diff_opts = opts['diffusion']
        self.train_opts = opts['train']
        self.G_opts = opts['train']['G_net']
        self.D_opts = opts['train']['D_net']
        self.save_opts = opts['save']
        self.start_epoch = 1
        self.global_step = -1
        self.device = torch.device('cuda' if self.train_opts['gpu_ids'] is not None else 'cpu')
        self.load()

    def model_to_device(self, network):
        network = network.to(self.device)
        if self.train_opts['dist']:
            find_unused_parameters = self.train_opts['find_unused_parameters']
            use_static_graph = self.train_opts['_set_static_graph']
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()],
                                              find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()
        else:
            network = DataParallel(network)
        return network

    def load(self):
        self.log_dict = OrderedDict()
        self.define_save_dir()

        self.netG = get_model(self.G_opts['network'], self.G_opts['net_param'])
        self.model_to_device(self.netG)
        self.lossesG = get_loss(self.G_opts['Loss_fn']['loss'], self.G_opts['Loss_fn']['weight'])

        self.optimizerG = get_optimizer(optim_name=self.G_opts['optimizer']['name'],
                                        network=self.netG,
                                        optim_param=self.G_opts['optimizer']['param'])
        self.schedulerG = get_schedule(scheduler_name=self.G_opts['lr_scheduler']['name'],
                                       optimizer=self.optimizerG,
                                       schedule_param=self.G_opts['lr_scheduler']['param'])

        self.Metric = get_loss(self.train_opts['Metric'], None, True)
        if self.train_opts['E_decay'] > 0:
            self.optimizerG = EMA(self.optimizerG, self.train_opts['E_decay'])

    def define_save_dir(self):
        """
        --save_dir
            --version
                log1
                log2
                tensroboardfile
                --checkpoint
        """
        exp_version = self.train_opts['version']  # 实验版本
        self.save_dir = OrderedDict()
        self.save_dir['save_dir'] = self.save_opts['dir']
        self.save_dir['model_path'] = os.path.join(self.save_dir['save_dir'], exp_version, 'checkpoint')
        self.save_dir['log_path'] = os.path.join(self.save_dir['save_dir'], exp_version)
        mk_dirs(self.save_dir['save_dir'])
        mk_dirs(self.save_dir['model_path'])

    @staticmethod
    def get_bare_model(network):
        """
        Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def save(self, network_label, is_best=False, epoch=0, global_step=-1):
        if is_best:
            network_label = 'best' + network_label

        netG_state = self.get_network_state(self.netG)
        content = {
            "netG_state": netG_state,
            'optimizerG': self.optimizerG.state_dict(),
            'schedulerG': self.schedulerG.state_dict(),
            'save_time': time.time(),
            'epoch': epoch + 1,
            'global_step': global_step
        }
        torch.save(content, os.path.join(self.save_dir['model_path'], network_label + '.pth'))

    def load_param(self, network_label):
        if self.save_opts['pretrained'] is not None:
            pretrain_path = self.save_opts['pretrain_path']
        elif self.save_opts['resume']:
            pretrain_path = os.path.join(self.save_dir['model_path'], network_label + '.pth')
        else:
            print('[!!] 不加载模型')
            return
        content = torch.load(pretrain_path,
                             map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        self.netG.load_state_dict(content['netG_state'])
        self.optimizerG.load_state_dict(content['optimizerG'])
        self.schedulerG.load_state_dict(content['schedulerG'])
        save_time = content['save_time']
        self.start_epoch = content['epoch']
        self.global_step = content['global_step']
        print('[OK] 自{}保存的模型中加载'.format(save_time))

    def save_G_only(self):
        netG_state = self.get_network_state(self.netG)
        torch.save(netG_state, os.path.join(self.save_dir['model_path'], 'onlyG.pth'))

    def get_res(self):
        return self.L, self.P, self.H

    def get_log_dict(self):
        return self.log_dict

    def feed_data(self, sample_batch):
        self.L, self.H = sample_batch
        self.L = self.L.to(self.device)
        self.H = self.H.to(self.device)


    def test_forward(self):
        self.netG.eval()
        self.P = self.netG(self.L)
        _, Metric_detail = self.lossfn(self.Metric, self.P, self.H)
        self.log_dict['Metric_detail'] = Metric_detail

    @staticmethod
    def lossfn(loss, pred, target):
        loss_detail = []
        loss_total = 0
        for idx, ls in enumerate(loss['loss']):
            ll = ls(pred, target) * loss['loss_weight'][idx]
            loss_detail.append(ll.item())
            loss_total += ll
        return loss_total, loss_detail

    def train_forward(self, global_step):
        self.P = self.netG(self.L)

        G_loss, G_loss_detail = self.lossfn(self.lossesG, self.P, self.H)
        G_loss.backward()

        G_optimizer_clipgrad = self.G_opts['optimizer_clipgrad'] if self.G_opts['optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.G_opts['optimizer_clipgrad'],
                                           norm_type=2)
        self.optimizerG.step()

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['G_loss_detail'] = G_loss_detail
        self.log_dict['G_lr'] = self.schedulerG.get_last_lr()[0]
        if not self.train_opts['lr_update_per_step']:
            self.schedulerG.step()  # 按步数来更行学习率

        if self.train_opts['E_decay'] > 0:
            self.update_E(self.train_opts['E_decay'])
