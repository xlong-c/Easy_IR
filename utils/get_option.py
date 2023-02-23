import yaml
import os

def get_options(path):
    f = open(path, "r", encoding='utf-8')
    opts = yaml.load(f, Loader=yaml.FullLoader)
    print('[OK] 已经加载配置文件')
    return opts


def save_options(opts):
    print('[OK] 配置文件已保存')
    with open(os.path.join(opts['save_dir'], 'cfg.yaml'), 'w') as ff:
        yaml.dump(opts, ff)
        ff.close()

