# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import yaml
import os
import sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir='~/.torch/proxyless_nas', overwrite=False):
    target_dir = url.split('//')[-1]
    target_dir = os.path.dirname(target_dir)
    model_dir = os.path.expanduser(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


class LatencyEstimator(object):
    # def __init__(self, url='https://file.lzhu.me/projects/proxylessNAS/LatencyTools/mobile_trim.yaml'):
    def __init__(self):
        # FIXME: may not be using latest latency estimation model with overwrite=False
        # fname = download_url(url, overwrite=False)
        with open('Raspbian.yaml', 'r') as fp:
            self.lut0 = yaml.load(fp)
        with open('Nvidia.yaml', 'r') as fp:
            self.lut1 = yaml.load(fp)
        with open('V100.yaml', 'r') as fp:
            self.lut2 = yaml.load(fp)
    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, expand=None, kernel=None, stride=None, idskip=None, device=2):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 2.
                2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input), 'output:%s' % self.repr_shape(output), ]

        if ltype in ('expanded_conv',):
            assert None not in (expand, kernel, stride, idskip)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 'idskip:%d' % idskip]
        key = '-'.join(infos)
        if device == 0:
            return self.lut0[key]['mean']/4000
        if device == 1:
            return self.lut1[key]['mean']/1000
        if device == 2:
            return self.lut2[key]['mean']/1000

# if __name__ == '__main__':
#     est = LatencyEstimator()
#     # s = est.predict('expanded_conv', _input=(112, 112, 16), output=(56, 56, 24), expand=3, kernel=5, stride=2, idskip=0)
#     s = est.predict('global_avg_pooling', _input=(112, 112, 16), output=(56, 56, 24), expand=3, kernel=5, stride=2, idskip=0)
#     print(s)
