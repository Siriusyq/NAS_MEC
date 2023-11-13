# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
# 进行基本的验证数据集随机分组
import numpy as np

from utils import *


class DataProvider:
    VALID_SEED = 0  # random seed for the validation set
# 声明了静态方法 name，从而可以实现不实例化使用 DataProvider.name()，当然也可以实例化再调用： a = DataProvider();a.name()
    @staticmethod
    def name():
        """ Return name of the dataset """
        raise NotImplementedError

    @property
    def data_shape(self):
        """ Return shape as python list of one data entry """
        raise NotImplementedError

    @property
    def n_classes(self):
        """ Return `int` of num classes """
        raise NotImplementedError

    @property
    def save_path(self):
        """ local path to save the data """
        raise NotImplementedError

    @property
    def data_url(self):
        """ link to download the data """
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_labels, valid_size, n_classes):
        train_size = len(train_labels)
        # assert train_size > valid_size, '{} {}'.format(train_size, valid_size)# assert 检查条件是否符合，符合则执行后面的返回消息（可选），否则返回AssertionError
        g = torch.Generator() # a pseudorandom number generator for sampling
        g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist() # Returns a random permutation of integers from 0 to train size - 1.
        train_indexes, valid_indexes = [], []
        per_class_remain = get_split_list(valid_size, n_classes)
        for idx in rand_indexes:
            label = train_labels[idx]
            if isinstance(label, float):
                label = int(label)
            elif isinstance(label, np.ndarray):
                label = np.argmax(label)
            else:
                assert isinstance(label, int)
            if per_class_remain[label] > 0:
                valid_indexes.append(idx)
                per_class_remain[label] -= 1
            else:
                train_indexes.append(idx)
        return train_indexes, valid_indexes
