import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from trainer.Train_gan_diff import Trainer
from utils.get_option import get_options, save_options

if __name__ == "__main__":
    opts = get_options('options/diff_cfg.yml')
    tainer = Trainer(opts)
    save_options(opts)
    tainer.train_stage()
    tainer.test_stage()
