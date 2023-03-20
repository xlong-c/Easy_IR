"""
数据处理的python文件
"""
import numpy
from data.datachange import apply_mask
import joblib
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Brain_data(Dataset):
    """
    读取大脑数据集  读取的格式应为pickle数据格式  此外，对于训练数据有一个增强
    """

    def __init__(self, path, mode: str, to_bad_fn, transform=None):
        """
        大脑数据集的读取类的初始化函数， 之后对于使用pytorch读取自身设定的数据集，模仿本方法即可
        :param path 待读取的数据的路径
        :param is_train: 是否为训练数据
        :param mask: 采样方式
        :param transform: 训练图像增强时请传入transform
        """
        ### 数据集的数据信息在-1到1之间
        with open(path, 'rb') as f:
            self.data = joblib.load(f)
            # self.data = pickle.load(f)  # num 256 256 1
        self.mode = mode
        self.transform = transform
        self.to_bad_fn = to_bad_fn

        print(mode + ' dataset was loaded successfully! the length is {}'.format(self.__len__()))
        print(f'数据最值   最大值: {self.data.max()}, 最小值: {self.data.min()}')

    def __len__(self):  # len(var)
        return self.data.shape[0]

    def __getitem__(self, item):  # var[index]
        """
        使得数据可以通过索引返回， 训练图像需增强
        :param item: 索引
        :return:
        """
        H = self.data[item]
        if self.mode == 'train':
            for f in self.transform:
                H = f()(H)
        H = ToTensor()(H).reshape(1, 256, 256)
        L = self.to_bad_fn.tobad(H)
        return L, H


if __name__ == '__main__':
    path = r'D:\dataset\MICCAI13_SegChallenge\validation.pickle'
    with open(path, 'rb') as f:
        data = joblib.load(f)
    print(data.max(), data.min())
    print(data.shape)
