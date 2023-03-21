import warnings

import torch

from trainer.MODEL_gan import GANMODEL
from utils.Loss_Metric import NMSE
from utils.diffusion import Diffusion
from utils.tools import grad_penalty_call, reverse_matrix
import torch.nn.functional as F


def NMSE(gen, target):
    if not (target.size() == gen.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), gen.size()),
                      stacklevel=2)
    nmse_a = torch.sqrt(torch.sum(torch.pow((gen - target), 2), dim=[1, 2, 3]))
    nmse_b = torch.sqrt(torch.sum(torch.pow(target, 2), dim=[1, 2, 3]))
    return torch.mean(nmse_a / nmse_b)  # tensor [batch]


class DIFFGANMODEL(GANMODEL):

    def __init__(self, opts):
        super(DIFFGANMODEL, self).__init__(opts)
        self.diff_opts = opts['diffusion']

    def load(self):
        super(DIFFGANMODEL, self).load()
        self.diffusion = Diffusion(self.opts)

    def feed_data(self, sample_batch):
        L, H, mask = sample_batch
        super().feed_data((L, H))
        # Q: reverse_matrix的作用？
        # A: mask反转，mask为1的地方不加噪声，mask为0的地方加噪声
        self.mask = reverse_matrix(mask).to(self.device)  # mask 反转

    def train_forward(self):
        # 先让D走
        for p in self.netD.parameters():
            p.requires_grad = True
        self.netD.zero_grad()
        # t 时刻
        t = self.diffusion.get_T((self.H.shape[0],))  # 生成时刻t
        x_t, x_tp1 = self.diffusion.q_sample_pairs(self.H, t, self.mask)  # 生成t t+1 数据
        x_t.requires_grad = True  # 使得x_t可以求导
        # train with real
        D_real = self.netD(x_t, t, x_tp1.detach())

        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()

        if self.diff_opts['lazy_reg_step'] is None:
            grad_penalty_call(self.diff_opts['r1_gamma'], D_real, x_t)
        else:
            if self.global_step % self.diff_opts['lazy_reg_step'] == 0:
                grad_penalty_call(self.diff_opts['r1_gamma'], D_real, x_t)

        # train with fake
        x_0_predict, x_pre = self.netG(x_tp1.detach(), self.L, t)
        x_pos_sample = self.diffusion.sample_posterior(x_0_predict, x_tp1, t, mask=self.mask)

        output = self.netD(x_pos_sample, t, x_tp1.detach()).view(-1)
        errD_fake = F.softplus(output).mean()

        errD_fake.backward()
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        # update G
        for p in self.netD.parameters():
            p.requires_grad = False

        for p in self.netG.parameters():
            p.requires_grad = True
        self.netG.zero_grad()

        t = self.diffusion.get_T((self.H.shape[0],))
        x_t, x_tp1 = self.diffusion.q_sample_pairs(self.H, t, self.mask)

        x_0_predict, x_pre = self.netG(x_tp1.detach(), self.L, t)
        x_pos_sample = self.diffusion.sample_posterior(x_0_predict, x_tp1, t, self.mask)

        output = self.netD(x_pos_sample, t, x_tp1.detach()).view(-1)
        errG = F.softplus(-output).mean()
        # reconstructior loss
        rec_loss_ = torch.zeros((1,))
        rec_loss = torch.zeros((1,))
        if self.diff_opts['rec_loss']:
            rec_loss = NMSE(x_0_predict, self.H)
            rec_loss_ = NMSE(x_pre, self.H)

            errG = errG + rec_loss + rec_loss_

        errG.backward()
        self.optimizerG.step()
        self.log_dict['loss_detail'] = [errG.item(), errD.item(), rec_loss_.item(), rec_loss.item()]
        self.log_dict['loss_total'] = errG.item() + errD.item()
        self.log_dict['G_lr'] = self.schedulerG.get_last_lr()[0]
        if self.train_opts['lr_update_per_step']:
            self.scheduler_step()

    def test_forward(self):
        self.netG.eval()
        x_tp1 = self.L + self.diffusion.get_noise(self.L, self.mask)  # torch.randn_like(self.L).to(self.L.device)
        with torch.no_grad():
            self.P = self.diffusion.sample_from_model(generator=self.netG,
                                                      n_time=self.diff_opts['num_timesteps'],
                                                      x_init=x_tp1,
                                                      L=self.L,
                                                      mask=self.mask
                                                      )
        self.netG.train()
        _, Metric_detail = self.lossfn(self.Metric, self.P, self.H)
        self.log_dict['Metric_detail'] = Metric_detail
