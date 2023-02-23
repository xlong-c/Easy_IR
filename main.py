from models.model import MODEL
from utils.get_option import get_options


if __name__ == "__main__":
    opts = get_options('options/cfg.yml')
    model = MODEL(opts)
    model.save('aka',is_best=True)
    model.load_param('best_aka')
    
    