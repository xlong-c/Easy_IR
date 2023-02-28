import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from models.Train import Trainer
from utils.get_option import get_options, save_options
from models.MODEL import MODEL

if __name__ == "__main__":
    opts = get_options('options/cfg.yml')
    tainer = Trainer(opts)
    save_options(opts)
    tainer.train_stage()
    tainer.test_stage()
