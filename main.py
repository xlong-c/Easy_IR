import os
from trainer.build_trainer import build_trainer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.get_option import get_options, save_options

if __name__ == "__main__":
    opts = get_options('options/diff_cfg.yml')
    tainer = build_trainer(opts)
    save_options(opts)
    tainer.train_stage()
    tainer.test_stage()
