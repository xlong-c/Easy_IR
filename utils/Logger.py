import logging
from collections import OrderedDict
import os
from torch.utils.tensorboard import SummaryWriter

'''
示例:
myLogger = Auto_Logger('res', ['train', 'test', 'valid'], True)
a = OrderedDict()
a['PSNR'] = 44.85858
myLogger.log_tmp('test', a)
myLogger.flush('test', verbose= True)
myLogger.define_log_rule('test', 'PSNR: {:<10.4f}')
b = [46.66666]
myLogger.rule_log('test', b, verbose=True)
myLogger.set_writer_rule('test', ['PSNR', 'NMSE'])
myLogger.rule_writer_log('test', [46.333, 0.99598], 18)

输出:
['PSNR: 44.8586   ']
PSNR: 46.6667 
  
test.log中:
PSNR: 44.8586   
PSNR: 46.6667   

日志目录结构:
res 
    -test.log
    -train.log
    -test.log
    -event
'''
class Auto_Logger():
    def __init__(self, path, log_types: list, On_tensorboard=False):
        """日志生成

        Args:
            path (_type_): 日志路径
            log_types (list): 日志有多少个分类，数组里面是字符串
            On_tensorboard (bool, optional): 是否打开tensorboard. Defaults to False.
        """
        self.save_path = path
        self.log_types = log_types
        self.loggers = OrderedDict()
        self.log_streams = OrderedDict()
        self.tmp_log = OrderedDict()
        self.rule_tmp_log = OrderedDict()
        if On_tensorboard:
            self.load_writer()
        self.load()

    def load_writer(self):
        self.writer = SummaryWriter(log_dir=self.save_path)
        self.writer_rule = OrderedDict()
        for log_type in self.log_types:
            self.writer_rule[log_type] = []

    def load(self):
        '''
        加载日志
        '''
        for log_type in self.log_types:
            log_path = os.path.join(self.save_path, '{}.log'.format(log_type))
            self.loggers[log_type] = logging.getLogger(log_type)
            self.loggers[log_type].addHandler(logging.FileHandler(log_path))
            self.loggers[log_type].setLevel(logging.DEBUG)
            self.log_streams[log_type] = logging.StreamHandler()
            self.log_streams[log_type] .setLevel(logging.ERROR)
        for log_type in self.log_types:
            self.tmp_log[log_type] = []
            self.rule_tmp_log[log_type] = ''

    def tostr(self, data, precision: int = 4, width: int = 10):
        """将非字符串转换为字符串

        Args:
            data (_type_): 数据
            precision (int, optional): 小数保留位数. Defaults to 4.
            width (int, optional): 数据转字符占位宽，左对齐. Defaults to 10.

        Returns:
            字符串化数据
        """
        if isinstance(data, str):
            return data
        if isinstance(data, float):
            ff = ''.join(["{:<", str(width), ".", str(precision), "f}"])
            return ff.format(data)
        if isinstance(data, int):
            ff = ''.join(["{:<", str(width), '}'])
            return ff.format(data)

    def log_tmp(self, log_type, log, precision: int = 4, width: int = 10):
        """暂存日志

        Args:
            log_type (str): 日志分类
            log (str): 日志类容
            precision (int, optional): 小数精度. Defaults to 4.
            width (int, optional): 位宽，左对齐. Defaults to 10.
        """
        tmp_log = None
        if isinstance(log, dict):
            t_log = []
            for item in log:
                t_log.append(
                    ': '.join([item, self.tostr(log[item], precision, width)]))
            tmp_log = '  '.join(t_log)
        elif isinstance(log, list):
            tmp_log = '  '.join(log)
        else:
            tmp_log = self.tostr(log, precision, width)
        self.tmp_log[log_type].append(tmp_log)

    def flush(self, log_type, verbose=False):
        """将暂存日志记录到日志文件上

        Args:
            log_type (str): 日志分类
            verbose (bool, optional): 是否打印. Defaults to False.
        """
        if verbose:
            print(self.tmp_log[log_type])
        self.loggers[log_type].debug(' '.join(self.tmp_log[log_type]))
        self.tmp_log[log_type] .clear()

    def define_log_rule(self, log_type, rule):
        """设置日志规则

        Args:
            log_type (str): 日志分类
            rule (str): 日志规则
        """
        self.rule_tmp_log[log_type] = rule

    def rule_log(self, log_type, log, verbose=False):
        """以设置的规则记录日志

        Args:
            log_type (str): 日志分类
            log (list, Iterable): 日志内容，数组或元组
            verbose (bool, optional): 是否打印. Defaults to False.
        """
        tmp_log = self.rule_tmp_log[log_type].format(*log)
        if verbose:
            print(tmp_log)
        self.loggers[log_type].debug(tmp_log)

    def log(self, log_type, log, verbose=False):
        """直接写入日志

        Args:
            log_type (str): 日志分类
            log (str): 日志内容
            verbose (bool, optional): 是否打印. Defaults to False.
        """
        if verbose:
            print(log)
        self.loggers[log_type].debug(log)

    def writer_log(self, log_type: str, log: dict, niter: int):
        """tensorboard记录
        Args:
            log_type (str): 日志分类
            log (dict): 日志内容
            niter (int): 计数器
        """
        for log_key in log:
            self.writer.add_scalar(tag='{}/{}'.format(log_type, log_key),
                                   scalar_value=log[log_key],
                                   global_step=niter
                                   )
        self.writer.flush()

    def set_writer_rule(self, log_type: str, rule: list):
        assert self.writer is not None
        """设置tensorboard规则

        Args:
            log_type (str): 日志分类
            rule (list): 日志规则
        """
        self.writer_rule[log_type] = rule

    def rule_writer_log(self, log_type, log: list, niter: int):
        """tensorboard规则记录

        Args:
            log_type (str): 日志分类
            log (list): 日志内容
            niter (int): 计数器
        """
      
        assert len(log) == len(self.writer_rule[log_type])
        for idx, item in enumerate(self.writer_rule[log_type]):
            self.writer.add_scalar(tag='{}/{}'.format(log_type, item),
                                   scalar_value=log[idx],
                                   global_step=niter
                                   )

    def close(self):
        """关闭
        """
        self.writer.close()


# myLogger = Auto_Logger('res', ['train', 'test', 'valid'], True)
# a = OrderedDict()
# a['PSNR'] = 44.85858
# myLogger.log_tmp('test', a)
# myLogger.flush('test', verbose= True)
# myLogger.define_log_rule('test', 'PSNR: {:<10.4f}')
# b = [46.66666]
# myLogger.rule_log('test', b, verbose=True)
# myLogger.set_writer_rule('test', ['PSNR', 'NMSE'])
# myLogger.rule_writer_log('test', [46.333, 0.99598], 18)
# a = {"PSNR":44.31588}
