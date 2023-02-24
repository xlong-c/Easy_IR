from utils.get_option import get_options
from models.model import MODEL
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == "__main__":
    # opts = get_options('options/cfg.yml')
    # model = MODEL(opts)
    # model.save('aka',is_best=True)
    # model.load_param('best_aka')
    G_loss = ['PSNR', 'NMSE']
    time_rule = 'tooks: {:.2f}S'
    loss_rule = []
    for loss in G_loss:
        loss_rule.append(loss + '  {:>10.4f}')
    loss_rule = '  '.join(loss_rule)
    epoch_num = '{:0>3}'.format(40)
    train_rule = 'EPOCH: '+epoch_num + \
        '/{:0>3} LOSS: {:>10.4f} ' + loss_rule + time_rule
    print(train_rule)
