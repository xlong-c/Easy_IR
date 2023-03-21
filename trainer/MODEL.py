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
        self.opts = opts  # 配置字典
        self.train_opts = opts['train']  # 训练设置
        self.G_opts = opts['train']['G_net']  # G网络设置
        self.D_opts = opts['train']['D_net']  # D网络设置
        self.save_opts = opts['save']  # 保存设置
        self.start_epoch = 1  # 开始的epoch
        self.global_step = 1  # 全局步数
        self.device = torch.device('cuda' if self.train_opts['gpu_ids'] is not None else 'cpu')  # 运行设备
        self.load()

    def model_to_device(self, network):
        """
         向当前运行设备迁移网络
        """
        network = network.to(self.device)
        if self.train_opts['dist']:  # 分布式
            find_unused_parameters = self.train_opts['find_unused_parameters']
            use_static_graph = self.train_opts['_set_static_graph']  # 静态图
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()],
                                              find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()
        elif self.device.type == 'cuda':
            network = DataParallel(network)  # 单机
        return network

    def load(self):
        self.log_dict = OrderedDict()
        self.define_save_dir()
        self.netG = get_model(self.G_opts['network'], self.G_opts['network_dir'], self.G_opts['net_init'], self.G_opts['net_param'])
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

        self.scaler = torch.cuda.amp.GradScaler() if self.train_opts['amp'] else None

    def define_save_dir(self):
        """ 检查或者创建保存目录
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
        self.save_dir['model_path'] = os.path.join(self.save_dir['save_dir'], 'checkpoint',exp_version)
        self.save_dir['log_path'] = os.path.join(self.save_dir['save_dir'],'log', exp_version)
        mk_dirs(self.save_dir['save_dir'])
        mk_dirs(self.save_dir['model_path'])

    @staticmethod
    def get_bare_model(network):
        """
        Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        翻译：获取裸模型，特别是在包装下
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def get_network_state(self, network):
        """
        获取网络权重并迁移至cpu
        转英语：Get network weights and migrate to cpu
        """
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        return state_dict

    def save(self, network_label, is_best=False, epoch=0, global_step=-1):
        """
        保存训练节点
        """
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

    def get_model_pth(self, network_label):
        """
        加载模型节点
        """
        if not self.save_opts['resume']:  # 不加载模型
            print('[!!] 不加载模型')
            return False
        pretrain_path = os.path.join(self.save_dir['model_path'], network_label + '.pth')
        if self.save_opts['pretrain_path'] is not None:
            pretrain_path = self.save_opts['pretrain_path']
        if not os.path.exists(pretrain_path):
            print('[!!] 模型文件不存在,不加载模型')
            return False
        return pretrain_path

    def load_param(self, network_label):
        """
        加载模型极其相关参数
        """
        pretrain_path = self.get_model_pth(network_label)
        if pretrain_path == False:
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
        """
        仅保存G网络参数
        """
        netG_state = self.get_network_state(self.netG)
        torch.save(netG_state, os.path.join(self.save_dir['model_path'], 'onlyG.pth'))

    def get_res(self):
        """
        获取数据结果(数据)
        """
        return self.L, self.P, self.H

    def get_log_dict(self):
        """
        获取日志数据(数值)
        """
        return self.log_dict

    def feed_data(self, sample_batch):
        """
        填充训练数据
        """
        self.L, self.H = sample_batch
        self.L = self.L.to(self.device)
        self.H = self.H.to(self.device)

    def test_forward(self):
        """
        推理
        """
        self.netG.eval()
        self.P = self.netG(self.L)
        _, Metric_detail = self.lossfn(self.Metric, self.P, self.H)
        self.log_dict['Metric_detail'] = Metric_detail

    @staticmethod
    def lossfn(loss, pred, target):
        """
        执行损失函数
        """
        loss_detail = []
        loss_total = 0
        for idx, ls in enumerate(loss['loss']):
            ll = ls(pred, target) * loss['loss_weight'][idx]
            loss_detail.append(ll.item())
            loss_total += ll
        return loss_total, loss_detail

    def global_step_pp(self):
        self.global_step = self.global_step +1
        
    def scheduler_step(self):
        self.schedulerG.step()
    
    def train_forward(self) :
        """
        网络训练
        """
        self.P = self.netG(self.L)

        G_loss, G_loss_detail = self.lossfn(self.lossesG, self.P, self.H)
        if self.scaler is not None:
            self.scaler.scale(G_loss).backward()
            self.scaler.step(self.optimizerG)
            self.scaler.update()
        else:
            G_loss.backward()
            self.optimizerG.step()
        if self.train_opts['lr_update_per_step']:
            self.scheduler_step() # 按步数来更行学习率
        G_optimizer_clipgrad = self.G_opts['optimizer_clipgrad'] if self.G_opts['optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.G_opts['optimizer_clipgrad'],
                                           norm_type=2)

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['G_loss_detail'] = G_loss_detail
        self.log_dict['G_lr'] = self.schedulerG.get_last_lr()[0]
        

        if self.train_opts['E_decay'] > 0:
            self.update_E(self.train_opts['E_decay'])

    def optimizer_step(self):
        """
        优化器更新
        """
        self.optimizerG.step()
