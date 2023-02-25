import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.get_parts import get_model, get_loss, get_optimizer, get_schedule
from utils.tools import mk_dirs


class MODEL(nn.Module):
    def __init__(self, opts):
        super(MODEL, self).__init__()
        self.opts = opts
        self.train_opts = opts['train']
        self.G_opts = opts['train']['G_net']
        self.save_opts = opts['save']
        self.start_epoch = 1
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
        self.G_model = get_model(self.G_opts['network'], self.G_opts['net_param'])
        if self.train_opts['E_decay'] > 0:
            self.E_model = get_model(self.G_opts['network'], self.G_opts['net_param'])
        self.model_to_device(self.G_model)
        self.G_losses = get_loss(self.G_opts['Loss_fn']['loss'], self.G_opts['Loss_fn']['weight'])
        self.Metric = get_loss(self.train_opts['Metric'], None, True)
        self.G_optimizer = get_optimizer(optim_name=self.G_opts['optimizer']['name'],
                                         network=self.G_model,
                                         optim_param=self.G_opts['optimizer']['param'])
        self.G_scheduler = get_schedule(scheduler_name=self.G_opts['lr_scheduler']['name'],
                                        optimizer=self.G_optimizer,
                                        schedule_param=self.G_opts['lr_scheduler']['param'])

        if self.train_opts['E_decay'] > 0:
            self.netE = get_model(self.G_opts['network'], self.G_opts['params']).to(self.device).eval()

    def define_save_dir(self):
        exp_version = self.train_opts['version']
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

    def save(self, network_label, is_best=False, epoch=0):
        if is_best:
            network_label = 'best_' + network_label
        self.save_G(network_label, epoch)

    def save_G(self, network_label, epoch):
        network_label = 'G_net_' + network_label
        self.save_network(self.G_model, network_label)
        self.save_orthers(self.G_optimizer, self.G_scheduler, network_label, epoch)

    def save_network(self, network, network_label='G_net'):
        save_filename = '{}.pth'.format(network_label)
        save_path = os.path.join(self.save_dir['model_path'], save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_orthers(self, optimizer, scheduler, network_label='G_net', epoch=0):
        network_label += '_orthers'
        save_filename = '{}.pth'.format(network_label)
        save_path = os.path.join(self.save_dir['model_path'], save_filename)
        state_dict = {
            "optimizer_state_dict": optimizer.state_dict(),
            "schedule_state_dict": scheduler.state_dict(),
            "save_time": time.asctime(time.localtime(time.time())),
            'epoch': epoch
        }
        torch.save(state_dict, save_path)

    def load_param(self, network_label):

        if self.save_opts['pretrained']:
            g_model_path = self.save_opts['pretrain_path']['G_net_path']
            orther_path = self.save_opts['pretrain_path']['G_orthers_path']
            if self.train_opts['E_decay'] > 0:
                e_model_path = self.save_opts['pretrain_path']['E_net_path']
        elif self.save_opts['resume']:
            g_model_path = os.path.join(self.save_dir['model_path'], f'G_net_{network_label}.pth')
            orther_path = os.path.join(self.save_dir['model_path'], f'G_net_{network_label}_orthers.pth')
            if self.train_opts['E_decay'] > 0:
                e_model_path = os.path.join(self.save_dir['model_path'], network_label + 'E_net.pth')
        else:
            print('[!!] 不加载模型')
            return
        self.load_network(self.G_model, g_model_path, self.save_opts['net_load_strict'])
        if self.train_opts['E_decay'] > 0:
            self.load_network(self.E_model, e_model_path, self.save_opts['net_load_strict'])
        self.load_orthers(orther_path,
                          self.G_optimizer, self.G_scheduler)

    def load_network(self, network, network_path, strict=True, param_key='params'):
        if not os.path.exists( network_path):
            print('[!!] 模型文件不存在，不加载模型')
            return
        network = self.get_bare_model(network)
        state_dict = torch.load(network_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        network.load_state_dict(state_dict, strict=strict)

    def load_orthers(self, orther_path, optimizer, scheduler):
        if not os.path.exists(orther_path):
            print('[!!] 模型文件不存在，不加载模型其他组件参数')
            return
        state_dict = torch.load(orther_path,
                                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(state_dict['schedule_state_dict'])
        self.start_epoch = state_dict["epoch"]
        save_time = state_dict['save_time']
        print('[OK] 自{}保存的模型中加载'.format(save_time))

    def get_res(self):
        return self.L, self.P, self.H

    def get_log_dict(self):
        return self.log_dict

    def feed_data(self, sample_batch):
        self.L, self.H = sample_batch
        self.L = self.L.to(self.device)
        self.H = self.H.to(self.device)

    def forward_G(self):
        self.P = self.G_model(self.L)

    def test_forward(self):
        self.G_model.eval()
        self.P = self.G_model(self.L)
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

    def optim_parameters(self):
        self.G_optimizer.zero_grad()
        self.forward_G()
        G_loss, G_loss_detail = self.lossfn(self.G_losses, self.P, self.H)
        G_loss.backward()

        G_optimizer_clipgrad = self.G_opts['optimizer_clipgrad'] if self.G_opts['optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.G_opts['optimizer_clipgrad'],
                                           norm_type=2)
        self.G_optimizer.step()

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['G_loss_detail'] = G_loss_detail
        self.log_dict['G_lr'] = self.G_scheduler.get_last_lr()[0]
        if not self.train_opts['lr_update_per_step']:
            self.G_scheduler.step()  # 按步数来更行学习率

        if self.train_opts['E_decay'] > 0:
            self.update_E(self.train_opts['E_decay'])
