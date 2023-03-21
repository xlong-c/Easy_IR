from trainer.Train import Trainer


class Trainer_gan_diff(Trainer):
    def __init__(self, opts):
        super(Trainer_gan_diff, self).__init__(opts)

    def load_logger(self):
        super(Trainer_gan_diff, self).load_logger()
        # 重构训练日志规则
        time_rule = 'tooks: {:.2f}S'
        loss_rule = 'errG: {:<8.4f}  errD: {:<8.4f}  sure_loss: {:<8.4f} sure_loss2: {:<8.4f}'
        epoch_num = '{:>3}'.format(self.train_opts['num_epoch'])
        train_rule = 'EPOCH: ' + epoch_num + \
                     '/{:>3} step: {:<8} LOSS: {:<8.4f} ' + loss_rule + time_rule + '  lr: {:<8.4f}'

        self.logger.define_log_rule(
            'train',
            train_rule
        )

        self.logger.define_writer_rule('train', rule=['errG', 'errD', 'sure_loss', 'sure_loss2'])
