import os
import time
import torch

from trainer.MODEL import MODEL
from utils.get_parts import get_model, get_loss, get_optimizer, get_schedule


class GANMODEL(MODEL): # 继承MODEL类
    def __init__(self, opts):
        super(GANMODEL, self).__init__(opts)

    def load(self):
        super(GANMODEL, self).load() # 调用父类的load方法
        self.netD = get_model(model_name=self.D_opts['network'],
                              model_dir=self.D_opts['network_dir'],
                              model_init=self.D_opts['net_init'],
                              model_args=self.D_opts['net_param'])
        self.model_to_device(self.netD)
        self.lossesD = get_loss(self.D_opts['Loss_fn']['loss'], self.D_opts['Loss_fn']['weight'])
        self.optimizerD = get_optimizer(optim_name=self.D_opts['optimizer']['name'],
                                        network=self.netD,
                                        optim_param=self.D_opts['optimizer']['param'])
        self.schedulerD = get_schedule(scheduler_name=self.D_opts['lr_scheduler']['name'],
                                       optimizer=self.optimizerD,
                                       schedule_param=self.D_opts['lr_scheduler']['param'])

    def save(self, network_label, is_best=False, epoch=0, global_step=-1):
        if is_best:
            network_label = 'best' + network_label

        netG_state = self.get_network_state(self.netG)
        netD_state = self.get_network_state(self.netD)
        content = {
            "netG_state": netG_state,
            "netD_state": netD_state,
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'schedulerG': self.schedulerG.state_dict(),
            'schedulerD': self.schedulerD.state_dict(),
            'save_time': time.time(),
            'epoch': epoch + 1,
            'global_step': global_step
        }
        torch.save(content, os.path.join(self.save_dir['model_path'], network_label + '.pth'))

    def load_param(self, network_label):
        pretrain_path = self.get_model_pth(network_label)
        if pretrain_path == False:
            return
        content = torch.load(pretrain_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        self.netG.load_state_dict(content['netG_state'])
        self.netD.load_state_dict(content['netD_state'])
        self.optimizerG.load_state_dict(content['optimizerG'])
        self.optimizerD.load_state_dict(content['optimizerD'])
        self.schedulerG.load_state_dict(content['schedulerG'])
        self.schedulerD.load_state_dict(content['schedulerD'])
        save_time = content['save_time']
        self.start_epoch = content['epoch']
        self.global_step = content['global_step']
        print('[OK] 自{}保存的模型中加载'.format(save_time))

    def test_forward(self):
        self.netG.eval()
        self.P = self.netG(self.L)
        _, Metric_detail = self.lossfn(self.Metric, self.P, self.H)
        self.log_dict['Metric_detail'] = Metric_detail

    def train_forward(self, global_step):
        self.P = self.netG(self.L)

        self.optimizerG.zero_grad()
        self.forward_G()
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

    def optimizer_step(self):
        super(GANMODEL, self).optimizer_step()
        self.optimizerD.step()
