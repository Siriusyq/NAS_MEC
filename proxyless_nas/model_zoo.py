from functools import partial
import json
import pickle as pkl
import torch

from .utils import download_url
from .nas_modules import ProxylessNASNets
from .cifar_modules import PyramidTreeNet


def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path =  '/home/tianyuqing/Projects/Origin-NAS/searched_proxyless_nas/net.config'
    net_config_json = json.load(open(net_config_path, 'r'))
    
    if net_config_json['name'] == ProxylessNASNets.__name__:
        net = ProxylessNASNets.build_from_config(net_config_json)
    else:
        net = PyramidTreeNet.build_from_config(net_config_json)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = '/home/tianyuqing/Projects/Origin-NAS-3/searched_proxyless_nas/condition3.pth.tar'
        init = torch.load(init_path, map_location='cpu')

        # print(init['state_dict'].keys())

        # print(net.state_dict().keys())
        Common = []
        only_in_supernet=[]
        # for i in init['state_dict'].keys():
        #     if i in net.state_dict().keys():
        #         Common.append(i)
        #     else:
        #         only_in_supernet.append(i)
        # print('Common',Common)
        # print('only_in_super',only_in_supernet)
        # for i in net.state_dict().keys():
        print([i for i in net.state_dict().keys() if i not in init['state_dict'].keys() and i.startswith('blocks.1.')])
        print([i for i in init['state_dict'].keys() if i not in net.state_dict().keys() and i.startswith('blocks.1.')])

        



        # exit()
        net.load_state_dict(init['state_dict'])
        
        net.load_state_dict(init)
    return net


# proxyless_cpu = partial(
#     proxyless_base,
#     net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_cpu.config",
#     net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_cpu.pth"
# )

# proxyless_gpu = partial(
#     proxyless_base,
#     net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.config",
#     net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.pth")

# proxyless_mobile = partial(
#     proxyless_base,
#     net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile.config",
#     net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile.pth")

# proxyless_mobile_14 = partial(
#     proxyless_base,
#     net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile_14.config",
#     net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile_14.pth")

# proxyless_cifar = partial(
#     proxyless_base,
#     net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_cifar.config",
#     net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_cifar.pth"
# )

proxyless_condition1_100_140 = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/condition3.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/condition3.pth.tar"
)
