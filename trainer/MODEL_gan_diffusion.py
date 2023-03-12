import torch

from trainer.MODEL_gan import GANMODEL
from utils.diffusion import Diffusion
from utils.tools import grad_penalty_call
import torch.nn.functional as F


class DIFFGANMODEL(GANMODEL):

    def __init__(self, opts):
        super(DIFFGANMODEL, self).__init__(opts)

    def load(self):
        super(DIFFGANMODEL, self).load()
        self.diffusion = Diffusion(self.opts)

    def train_forward(self, global_step):
        # 先让D走
        for p in self.netD.parameters():
            p.requires_grad = True
        self.netD.zero_grad()
        # t 时刻
        # t = torch.randint(0, self.opts['diff']['num_timesteps'],
        #                   (self.L.shape[0],), device=self.device)
        t = self.diffusion.get_T((self.H.shape[0],))
        x_t, x_tp1 = self.diffusion.q_sample_pairs(self.H, t)
        x_t.requires_grad = True

        # train with real
        D_real = self.netD(x_t, t, x_tp1.detach())

        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()

        if self.diff_opts['lazy_reg_step'] is None:
            grad_penalty_call(self.diff_opts['r1_gamma'], D_real, x_t)
        else:
            if global_step % self.diff_opts['lazy_reg_step'] == 0:
                grad_penalty_call(self.diff_opts['r1_gamma'], D_real, x_t)

        # train with fake
        latent_z = torch.randn(self.H.shape[0], self.diff_opts['nz'], device=self.device)
        x_0_predict = self.netG(x_tp1.detach(), self.L, t, latent_z)
        x_pos_sample = self.diffusion.sample_posterior(x_0_predict, x_tp1, t)

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
        x_t, x_tp1 = self.diffusion.q_sample_pairs(self.H, t)

        latent_z = torch.randn(self.H.shape[0], self.diff_opts['nz'], device=self.device)

        x_0_predict = self.netG(x_tp1.detach(), self.L, t, latent_z)
        x_pos_sample = self.diffusion.sample_posterior(x_0_predict, x_tp1, t)

        output = self.netD(x_pos_sample, t, x_tp1.detach()).view(-1)
        errG = F.softplus(-output).mean()
        # reconstructior loss
        if self.diff_opts['rec_loss']:
            rec_loss = F.l1_loss(x_0_predict, self.H)
            errG = errG + rec_loss

        errG.backward()
        self.optimizerG.step()
        self.log_dict['loss_detail'] = [errG.item(), errD.item()]
        self.log_dict['loss_total'] = errG.item() + errD.item()
        self.log_dict['G_lr'] = self.schedulerG.get_last_lr()[0]

    def test_forward(self):
        self.netG.eval()
        x_tp1 = torch.randn_like(self.L).to(self.L.device)
        with torch.no_grad():
            self.P = self.diffusion.sample_from_model(generator=self.netG,
                                                      n_time=self.diff_opts['num_timesteps'],
                                                      x_init=x_tp1,
                                                      L=self.L)
        self.netG.train()
        _, Metric_detail = self.lossfn(self.Metric, self.P, self.H)
        self.log_dict['Metric_detail'] = Metric_detail
