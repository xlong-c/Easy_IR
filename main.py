import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from models.Train import Trainer
from utils.get_option import get_options
from models.model import MODEL

if __name__ == "__main__":
    opts = get_options('options/cfg.yml')
    tainer = Trainer(opts)
    tainer.train_stage()
