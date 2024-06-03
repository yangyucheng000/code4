#Reference: https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/sample_code.html
from basicsr.utils.options import parse_options
from basicsr.archs import build_network
from os import path as osp
import mindspore as ms
# 通过MindSpore的Cell，打印Cell里所有参数名和shape，返回字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params
# root_path = '/home/zzh/RIDCP_dehazing_MindSpore/options/RIDCP.yml'
# from basicsr.models.dehaze_vq_model import VQDehazeModel
ms.set_context(device_target="GPU" , pynative_synchronize=True)
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
opt, _ = parse_options(root_path, is_train=True)
# net = VQDehazeModel(opt)
net_g = build_network(opt['network_g'])
# net_g = build_network(opt)

ms_param = mindspore_params(net_g)
print("="*20)
