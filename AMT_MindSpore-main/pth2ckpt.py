# Reference: https://github.com/inata1024/UHDFour/blob/main/pth2ckpt.py
import torch
import mindspore as ms
import os
# from networks.AMT_S_MindSpore import Model
from networks.AMT_G_MIndSpore import Model
# from networks.AMT_L_MindSpore import Model


def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    _pt_params = {} # 去除前面的modules.
    for name in par_dict:
        parameter = par_dict[name]
        if isinstance(parameter, torch.Tensor):
            print(name, parameter.numpy().shape)
            _pt_params[name] = parameter.numpy()
        else:
            print(f"Skipping non-tensor object: {name} of type {type(parameter)}")
    return _pt_params


def pytorch_params(pth_file):
    checkpoint = torch.load(pth_file, map_location='cpu')
    # 检查是否存在state_dict条目
    if 'state_dict' in checkpoint:
        par_dict = checkpoint['state_dict']
    else:
        # 如果没有state_dict条目，假设整个文件就是参数字典
        par_dict = checkpoint

    _pt_params = {}  # 去除前面的modules.并转换为numpy数组
    for name, parameter in par_dict.items():
        if isinstance(parameter, torch.Tensor):
            print(name, parameter.numpy().shape)
            _pt_params[name] = parameter.numpy()
        else:
            print(f"Skipping non-tensor object: {name} of type {type(parameter)}")
    return _pt_params

# pytorch_params('/root/autodl-tmp/AMT_MindSpore/pretrained/amt-s.pth')

'''
feat_encoder.conv1.weight (32, 3, 7, 7)
feat_encoder.conv1.bias (32,)
feat_encoder.layer1.0.conv1.weight (8, 32, 1, 1)
feat_encoder.layer1.0.conv1.bias (8,)
feat_encoder.layer1.0.conv2.weight (8, 8, 3, 3)
feat_encoder.layer1.0.conv2.bias (8,)
feat_encoder.layer1.0.conv3.weight (32, 8, 1, 1)
feat_encoder.layer1.0.conv3.bias (32,)
feat_encoder.layer1.1.conv1.weight (8, 32, 1, 1)
feat_encoder.layer1.1.conv1.bias (8,)
feat_encoder.layer1.1.conv2.weight (8, 8, 3, 3)
feat_encoder.layer1.1.conv2.bias (8,)
feat_encoder.layer1.1.conv3.weight (32, 8, 1, 1)
feat_encoder.layer1.1.conv3.bias (32,)
feat_encoder.layer2.0.conv1.weight (16, 32, 1, 1)
feat_encoder.layer2.0.conv1.bias (16,)
feat_encoder.layer2.0.conv2.weight (16, 16, 3, 3)
feat_encoder.layer2.0.conv2.bias (16,)
feat_encoder.layer2.0.conv3.weight (64, 16, 1, 1)
feat_encoder.layer2.0.conv3.bias (64,)
feat_encoder.layer2.0.downsample.0.weight (64, 32, 1, 1)
feat_encoder.layer2.0.downsample.0.bias (64,)
feat_encoder.layer2.1.conv1.weight (16, 64, 1, 1)
feat_encoder.layer2.1.conv1.bias (16,)
feat_encoder.layer2.1.conv2.weight (16, 16, 3, 3)
feat_encoder.layer2.1.conv2.bias (16,)
feat_encoder.layer2.1.conv3.weight (64, 16, 1, 1)
feat_encoder.layer2.1.conv3.bias (64,)
feat_encoder.layer3.0.conv1.weight (24, 64, 1, 1)
feat_encoder.layer3.0.conv1.bias (24,)
feat_encoder.layer3.0.conv2.weight (24, 24, 3, 3)
feat_encoder.layer3.0.conv2.bias (24,)
feat_encoder.layer3.0.conv3.weight (96, 24, 1, 1)
feat_encoder.layer3.0.conv3.bias (96,)
feat_encoder.layer3.0.downsample.0.weight (96, 64, 1, 1)
feat_encoder.layer3.0.downsample.0.bias (96,)
feat_encoder.layer3.1.conv1.weight (24, 96, 1, 1)
feat_encoder.layer3.1.conv1.bias (24,)
feat_encoder.layer3.1.conv2.weight (24, 24, 3, 3)
feat_encoder.layer3.1.conv2.bias (24,)
feat_encoder.layer3.1.conv3.weight (96, 24, 1, 1)
feat_encoder.layer3.1.conv3.bias (96,)
feat_encoder.conv2.weight (84, 96, 1, 1)
feat_encoder.conv2.bias (84,)
encoder.pyramid1.0.0.weight (20, 3, 3, 3)
encoder.pyramid1.0.0.bias (20,)
encoder.pyramid1.0.1.weight (20,)
encoder.pyramid1.1.0.weight (20, 20, 3, 3)
encoder.pyramid1.1.0.bias (20,)
encoder.pyramid1.1.1.weight (20,)
encoder.pyramid2.0.0.weight (32, 20, 3, 3)
encoder.pyramid2.0.0.bias (32,)
encoder.pyramid2.0.1.weight (32,)
encoder.pyramid2.1.0.weight (32, 32, 3, 3)
encoder.pyramid2.1.0.bias (32,)
encoder.pyramid2.1.1.weight (32,)
encoder.pyramid3.0.0.weight (44, 32, 3, 3)
encoder.pyramid3.0.0.bias (44,)
encoder.pyramid3.0.1.weight (44,)
encoder.pyramid3.1.0.weight (44, 44, 3, 3)
encoder.pyramid3.1.0.bias (44,)
encoder.pyramid3.1.1.weight (44,)
encoder.pyramid4.0.0.weight (56, 44, 3, 3)
encoder.pyramid4.0.0.bias (56,)
encoder.pyramid4.0.1.weight (56,)
encoder.pyramid4.1.0.weight (56, 56, 3, 3)
encoder.pyramid4.1.0.bias (56,)
encoder.pyramid4.1.1.weight (56,)
decoder4.convblock.0.0.weight (112, 113, 3, 3)
decoder4.convblock.0.0.bias (112,)
decoder4.convblock.0.1.weight (112,)
decoder4.convblock.1.conv1.0.weight (112, 112, 3, 3)
decoder4.convblock.1.conv1.0.bias (112,)
decoder4.convblock.1.conv1.1.weight (112,)
decoder4.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder4.convblock.1.conv2.0.bias (20,)
decoder4.convblock.1.conv2.1.weight (20,)
decoder4.convblock.1.conv3.0.weight (112, 112, 3, 3)
decoder4.convblock.1.conv3.0.bias (112,)
decoder4.convblock.1.conv3.1.weight (112,)
decoder4.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder4.convblock.1.conv4.0.bias (20,)
decoder4.convblock.1.conv4.1.weight (20,)
decoder4.convblock.1.conv5.weight (112, 112, 3, 3)
decoder4.convblock.1.conv5.bias (112,)
decoder4.convblock.1.prelu.weight (112,)
decoder4.convblock.2.weight (112, 48, 4, 4)
decoder4.convblock.2.bias (48,)
decoder3.convblock.0.0.weight (132, 136, 3, 3)
decoder3.convblock.0.0.bias (132,)
decoder3.convblock.0.1.weight (132,)
decoder3.convblock.1.conv1.0.weight (132, 132, 3, 3)
decoder3.convblock.1.conv1.0.bias (132,)
decoder3.convblock.1.conv1.1.weight (132,)
decoder3.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder3.convblock.1.conv2.0.bias (20,)
decoder3.convblock.1.conv2.1.weight (20,)
decoder3.convblock.1.conv3.0.weight (132, 132, 3, 3)
decoder3.convblock.1.conv3.0.bias (132,)
decoder3.convblock.1.conv3.1.weight (132,)
decoder3.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder3.convblock.1.conv4.0.bias (20,)
decoder3.convblock.1.conv4.1.weight (20,)
decoder3.convblock.1.conv5.weight (132, 132, 3, 3)
decoder3.convblock.1.conv5.bias (132,)
decoder3.convblock.1.prelu.weight (132,)
decoder3.convblock.2.weight (132, 36, 4, 4)
decoder3.convblock.2.bias (36,)
decoder2.convblock.0.0.weight (96, 100, 3, 3)
decoder2.convblock.0.0.bias (96,)
decoder2.convblock.0.1.weight (96,)
decoder2.convblock.1.conv1.0.weight (96, 96, 3, 3)
decoder2.convblock.1.conv1.0.bias (96,)
decoder2.convblock.1.conv1.1.weight (96,)
decoder2.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder2.convblock.1.conv2.0.bias (20,)
decoder2.convblock.1.conv2.1.weight (20,)
decoder2.convblock.1.conv3.0.weight (96, 96, 3, 3)
decoder2.convblock.1.conv3.0.bias (96,)
decoder2.convblock.1.conv3.1.weight (96,)
decoder2.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder2.convblock.1.conv4.0.bias (20,)
decoder2.convblock.1.conv4.1.weight (20,)
decoder2.convblock.1.conv5.weight (96, 96, 3, 3)
decoder2.convblock.1.conv5.bias (96,)
decoder2.convblock.1.prelu.weight (96,)
decoder2.convblock.2.weight (96, 24, 4, 4)
decoder2.convblock.2.bias (24,)
decoder1.convblock.0.0.weight (60, 64, 3, 3)
decoder1.convblock.0.0.bias (60,)
decoder1.convblock.0.1.weight (60,)
decoder1.convblock.1.conv1.0.weight (60, 60, 3, 3)
decoder1.convblock.1.conv1.0.bias (60,)
decoder1.convblock.1.conv1.1.weight (60,)
decoder1.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder1.convblock.1.conv2.0.bias (20,)
decoder1.convblock.1.conv2.1.weight (20,)
decoder1.convblock.1.conv3.0.weight (60, 60, 3, 3)
decoder1.convblock.1.conv3.0.bias (60,)
decoder1.convblock.1.conv3.1.weight (60,)
decoder1.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder1.convblock.1.conv4.0.bias (20,)
decoder1.convblock.1.conv4.1.weight (20,)
decoder1.convblock.1.conv5.weight (60, 60, 3, 3)
decoder1.convblock.1.conv5.bias (60,)
decoder1.convblock.1.prelu.weight (60,)
decoder1.convblock.2.weight (60, 24, 4, 4)
decoder1.convblock.2.bias (24,)
update4.convc1.weight (64, 392, 1, 1)
update4.convc1.bias (64,)
update4.convf1.weight (40, 4, 7, 7)
update4.convf1.bias (40,)
update4.convf2.weight (20, 40, 3, 3)
update4.convf2.bias (20,)
update4.conv.weight (68, 84, 3, 3)
update4.conv.bias (68,)
update4.gru.0.weight (76, 116, 3, 3)
update4.gru.0.bias (76,)
update4.gru.2.weight (76, 76, 3, 3)
update4.gru.2.bias (76,)
update4.feat_head.0.weight (76, 76, 3, 3)
update4.feat_head.0.bias (76,)
update4.feat_head.2.weight (44, 76, 3, 3)
update4.feat_head.2.bias (44,)
update4.flow_head.0.weight (76, 76, 3, 3)
update4.flow_head.0.bias (76,)
update4.flow_head.2.weight (4, 76, 3, 3)
update4.flow_head.2.bias (4,)
update3.convc1.weight (64, 392, 1, 1)
update3.convc1.bias (64,)
update3.convf1.weight (40, 4, 7, 7)
update3.convf1.bias (40,)
update3.convf2.weight (20, 40, 3, 3)
update3.convf2.bias (20,)
update3.conv.weight (68, 84, 3, 3)
update3.conv.bias (68,)
update3.gru.0.weight (76, 104, 3, 3)
update3.gru.0.bias (76,)
update3.gru.2.weight (76, 76, 3, 3)
update3.gru.2.bias (76,)
update3.feat_head.0.weight (76, 76, 3, 3)
update3.feat_head.0.bias (76,)
update3.feat_head.2.weight (32, 76, 3, 3)
update3.feat_head.2.bias (32,)
update3.flow_head.0.weight (76, 76, 3, 3)
update3.flow_head.0.bias (76,)
update3.flow_head.2.weight (4, 76, 3, 3)
update3.flow_head.2.bias (4,)
update2.convc1.weight (64, 392, 1, 1)
update2.convc1.bias (64,)
update2.convf1.weight (40, 4, 7, 7)
update2.convf1.bias (40,)
update2.convf2.weight (20, 40, 3, 3)
update2.convf2.bias (20,)
update2.conv.weight (68, 84, 3, 3)
update2.conv.bias (68,)
update2.gru.0.weight (76, 92, 3, 3)
update2.gru.0.bias (76,)
update2.gru.2.weight (76, 76, 3, 3)
update2.gru.2.bias (76,)
update2.feat_head.0.weight (76, 76, 3, 3)
update2.feat_head.0.bias (76,)
update2.feat_head.2.weight (20, 76, 3, 3)
update2.feat_head.2.bias (20,)
update2.flow_head.0.weight (76, 76, 3, 3)
update2.flow_head.0.bias (76,)
update2.flow_head.2.weight (4, 76, 3, 3)
update2.flow_head.2.bias (4,)
comb_block.0.weight (18, 9, 3, 3)
comb_block.0.bias (18,)
comb_block.1.weight (18,)
comb_block.2.weight (3, 18, 3, 3)
comb_block.2.bias (3,)
'''

# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params

# mindspore_params(Model())
'''
feat_encoder.norm1.moving_mean (32,)
feat_encoder.norm1.moving_variance (32,)
feat_encoder.norm1.gamma (32,)
feat_encoder.norm1.beta (32,)
feat_encoder.conv1.weight (32, 3, 7, 7)
feat_encoder.conv1.bias (32,)
feat_encoder.layer1.0.conv1.weight (8, 32, 1, 1)
feat_encoder.layer1.0.conv1.bias (8,)
feat_encoder.layer1.0.conv2.weight (8, 8, 3, 3)
feat_encoder.layer1.0.conv2.bias (8,)
feat_encoder.layer1.0.conv3.weight (32, 8, 1, 1)
feat_encoder.layer1.0.conv3.bias (32,)
feat_encoder.layer1.0.norm1.moving_mean (8,)
feat_encoder.layer1.0.norm1.moving_variance (8,)
feat_encoder.layer1.0.norm1.gamma (8,)
feat_encoder.layer1.0.norm1.beta (8,)
feat_encoder.layer1.0.norm2.moving_mean (8,)
feat_encoder.layer1.0.norm2.moving_variance (8,)
feat_encoder.layer1.0.norm2.gamma (8,)
feat_encoder.layer1.0.norm2.beta (8,)
feat_encoder.layer1.0.norm3.moving_mean (32,)
feat_encoder.layer1.0.norm3.moving_variance (32,)
feat_encoder.layer1.0.norm3.gamma (32,)
feat_encoder.layer1.0.norm3.beta (32,)
feat_encoder.layer1.1.conv1.weight (8, 32, 1, 1)
feat_encoder.layer1.1.conv1.bias (8,)
feat_encoder.layer1.1.conv2.weight (8, 8, 3, 3)
feat_encoder.layer1.1.conv2.bias (8,)
feat_encoder.layer1.1.conv3.weight (32, 8, 1, 1)
feat_encoder.layer1.1.conv3.bias (32,)
feat_encoder.layer1.1.norm1.moving_mean (8,)
feat_encoder.layer1.1.norm1.moving_variance (8,)
feat_encoder.layer1.1.norm1.gamma (8,)
feat_encoder.layer1.1.norm1.beta (8,)
feat_encoder.layer1.1.norm2.moving_mean (8,)
feat_encoder.layer1.1.norm2.moving_variance (8,)
feat_encoder.layer1.1.norm2.gamma (8,)
feat_encoder.layer1.1.norm2.beta (8,)
feat_encoder.layer1.1.norm3.moving_mean (32,)
feat_encoder.layer1.1.norm3.moving_variance (32,)
feat_encoder.layer1.1.norm3.gamma (32,)
feat_encoder.layer1.1.norm3.beta (32,)
feat_encoder.layer2.0.conv1.weight (16, 32, 1, 1)
feat_encoder.layer2.0.conv1.bias (16,)
feat_encoder.layer2.0.conv2.weight (16, 16, 3, 3)
feat_encoder.layer2.0.conv2.bias (16,)
feat_encoder.layer2.0.conv3.weight (64, 16, 1, 1)
feat_encoder.layer2.0.conv3.bias (64,)
feat_encoder.layer2.0.norm1.moving_mean (16,)
feat_encoder.layer2.0.norm1.moving_variance (16,)
feat_encoder.layer2.0.norm1.gamma (16,)
feat_encoder.layer2.0.norm1.beta (16,)
feat_encoder.layer2.0.norm2.moving_mean (16,)
feat_encoder.layer2.0.norm2.moving_variance (16,)
feat_encoder.layer2.0.norm2.gamma (16,)
feat_encoder.layer2.0.norm2.beta (16,)
feat_encoder.layer2.0.norm3.moving_mean (64,)
feat_encoder.layer2.0.norm3.moving_variance (64,)
feat_encoder.layer2.0.norm3.gamma (64,)
feat_encoder.layer2.0.norm3.beta (64,)
feat_encoder.layer2.0.norm4.moving_mean (64,)
feat_encoder.layer2.0.norm4.moving_variance (64,)
feat_encoder.layer2.0.norm4.gamma (64,)
feat_encoder.layer2.0.norm4.beta (64,)
feat_encoder.layer2.0.downsample.0.weight (64, 32, 1, 1)
feat_encoder.layer2.0.downsample.0.bias (64,)
feat_encoder.layer2.1.conv1.weight (16, 64, 1, 1)
feat_encoder.layer2.1.conv1.bias (16,)
feat_encoder.layer2.1.conv2.weight (16, 16, 3, 3)
feat_encoder.layer2.1.conv2.bias (16,)
feat_encoder.layer2.1.conv3.weight (64, 16, 1, 1)
feat_encoder.layer2.1.conv3.bias (64,)
feat_encoder.layer2.1.norm1.moving_mean (16,)
feat_encoder.layer2.1.norm1.moving_variance (16,)
feat_encoder.layer2.1.norm1.gamma (16,)
feat_encoder.layer2.1.norm1.beta (16,)
feat_encoder.layer2.1.norm2.moving_mean (16,)
feat_encoder.layer2.1.norm2.moving_variance (16,)
feat_encoder.layer2.1.norm2.gamma (16,)
feat_encoder.layer2.1.norm2.beta (16,)
feat_encoder.layer2.1.norm3.moving_mean (64,)
feat_encoder.layer2.1.norm3.moving_variance (64,)
feat_encoder.layer2.1.norm3.gamma (64,)
feat_encoder.layer2.1.norm3.beta (64,)
feat_encoder.layer3.0.conv1.weight (24, 64, 1, 1)
feat_encoder.layer3.0.conv1.bias (24,)
feat_encoder.layer3.0.conv2.weight (24, 24, 3, 3)
feat_encoder.layer3.0.conv2.bias (24,)
feat_encoder.layer3.0.conv3.weight (96, 24, 1, 1)
feat_encoder.layer3.0.conv3.bias (96,)
feat_encoder.layer3.0.norm1.moving_mean (24,)
feat_encoder.layer3.0.norm1.moving_variance (24,)
feat_encoder.layer3.0.norm1.gamma (24,)
feat_encoder.layer3.0.norm1.beta (24,)
feat_encoder.layer3.0.norm2.moving_mean (24,)
feat_encoder.layer3.0.norm2.moving_variance (24,)
feat_encoder.layer3.0.norm2.gamma (24,)
feat_encoder.layer3.0.norm2.beta (24,)
feat_encoder.layer3.0.norm3.moving_mean (96,)
feat_encoder.layer3.0.norm3.moving_variance (96,)
feat_encoder.layer3.0.norm3.gamma (96,)
feat_encoder.layer3.0.norm3.beta (96,)
feat_encoder.layer3.0.norm4.moving_mean (96,)
feat_encoder.layer3.0.norm4.moving_variance (96,)
feat_encoder.layer3.0.norm4.gamma (96,)
feat_encoder.layer3.0.norm4.beta (96,)
feat_encoder.layer3.0.downsample.0.weight (96, 64, 1, 1)
feat_encoder.layer3.0.downsample.0.bias (96,)
feat_encoder.layer3.1.conv1.weight (24, 96, 1, 1)
feat_encoder.layer3.1.conv1.bias (24,)
feat_encoder.layer3.1.conv2.weight (24, 24, 3, 3)
feat_encoder.layer3.1.conv2.bias (24,)
feat_encoder.layer3.1.conv3.weight (96, 24, 1, 1)
feat_encoder.layer3.1.conv3.bias (96,)
feat_encoder.layer3.1.norm1.moving_mean (24,)
feat_encoder.layer3.1.norm1.moving_variance (24,)
feat_encoder.layer3.1.norm1.gamma (24,)
feat_encoder.layer3.1.norm1.beta (24,)
feat_encoder.layer3.1.norm2.moving_mean (24,)
feat_encoder.layer3.1.norm2.moving_variance (24,)
feat_encoder.layer3.1.norm2.gamma (24,)
feat_encoder.layer3.1.norm2.beta (24,)
feat_encoder.layer3.1.norm3.moving_mean (96,)
feat_encoder.layer3.1.norm3.moving_variance (96,)
feat_encoder.layer3.1.norm3.gamma (96,)
feat_encoder.layer3.1.norm3.beta (96,)
feat_encoder.conv2.weight (84, 96, 1, 1)
feat_encoder.conv2.bias (84,)
encoder.pyramid1.0.0.weight (20, 3, 3, 3)
encoder.pyramid1.0.0.bias (20,)
encoder.pyramid1.0.1.w (20,)
encoder.pyramid1.1.0.weight (20, 20, 3, 3)
encoder.pyramid1.1.0.bias (20,)
encoder.pyramid1.1.1.w (20,)
encoder.pyramid2.0.0.weight (32, 20, 3, 3)
encoder.pyramid2.0.0.bias (32,)
encoder.pyramid2.0.1.w (32,)
encoder.pyramid2.1.0.weight (32, 32, 3, 3)
encoder.pyramid2.1.0.bias (32,)
encoder.pyramid2.1.1.w (32,)
encoder.pyramid3.0.0.weight (44, 32, 3, 3)
encoder.pyramid3.0.0.bias (44,)
encoder.pyramid3.0.1.w (44,)
encoder.pyramid3.1.0.weight (44, 44, 3, 3)
encoder.pyramid3.1.0.bias (44,)
encoder.pyramid3.1.1.w (44,)
encoder.pyramid4.0.0.weight (56, 44, 3, 3)
encoder.pyramid4.0.0.bias (56,)
encoder.pyramid4.0.1.w (56,)
encoder.pyramid4.1.0.weight (56, 56, 3, 3)
encoder.pyramid4.1.0.bias (56,)
encoder.pyramid4.1.1.w (56,)
decoder4.convblock.0.0.weight (112, 113, 3, 3)
decoder4.convblock.0.0.bias (112,)
decoder4.convblock.0.1.w (112,)
decoder4.convblock.1.conv1.0.weight (112, 112, 3, 3)
decoder4.convblock.1.conv1.0.bias (112,)
decoder4.convblock.1.conv1.1.w (112,)
decoder4.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder4.convblock.1.conv2.0.bias (20,)
decoder4.convblock.1.conv2.1.w (20,)
decoder4.convblock.1.conv3.0.weight (112, 112, 3, 3)
decoder4.convblock.1.conv3.0.bias (112,)
decoder4.convblock.1.conv3.1.w (112,)
decoder4.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder4.convblock.1.conv4.0.bias (20,)
decoder4.convblock.1.conv4.1.w (20,)
decoder4.convblock.1.conv5.weight (112, 112, 3, 3)
decoder4.convblock.1.conv5.bias (112,)
decoder4.convblock.1.prelu.w (112,)
decoder4.convblock.2.weight (112, 48, 4, 4)
decoder4.convblock.2.bias (48,)
decoder3.convblock.0.0.weight (132, 136, 3, 3)
decoder3.convblock.0.0.bias (132,)
decoder3.convblock.0.1.w (132,)
decoder3.convblock.1.conv1.0.weight (132, 132, 3, 3)
decoder3.convblock.1.conv1.0.bias (132,)
decoder3.convblock.1.conv1.1.w (132,)
decoder3.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder3.convblock.1.conv2.0.bias (20,)
decoder3.convblock.1.conv2.1.w (20,)
decoder3.convblock.1.conv3.0.weight (132, 132, 3, 3)
decoder3.convblock.1.conv3.0.bias (132,)
decoder3.convblock.1.conv3.1.w (132,)
decoder3.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder3.convblock.1.conv4.0.bias (20,)
decoder3.convblock.1.conv4.1.w (20,)
decoder3.convblock.1.conv5.weight (132, 132, 3, 3)
decoder3.convblock.1.conv5.bias (132,)
decoder3.convblock.1.prelu.w (132,)
decoder3.convblock.2.weight (132, 36, 4, 4)
decoder3.convblock.2.bias (36,)
decoder2.convblock.0.0.weight (96, 100, 3, 3)
decoder2.convblock.0.0.bias (96,)
decoder2.convblock.0.1.w (96,)
decoder2.convblock.1.conv1.0.weight (96, 96, 3, 3)
decoder2.convblock.1.conv1.0.bias (96,)
decoder2.convblock.1.conv1.1.w (96,)
decoder2.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder2.convblock.1.conv2.0.bias (20,)
decoder2.convblock.1.conv2.1.w (20,)
decoder2.convblock.1.conv3.0.weight (96, 96, 3, 3)
decoder2.convblock.1.conv3.0.bias (96,)
decoder2.convblock.1.conv3.1.w (96,)
decoder2.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder2.convblock.1.conv4.0.bias (20,)
decoder2.convblock.1.conv4.1.w (20,)
decoder2.convblock.1.conv5.weight (96, 96, 3, 3)
decoder2.convblock.1.conv5.bias (96,)
decoder2.convblock.1.prelu.w (96,)
decoder2.convblock.2.weight (96, 24, 4, 4)
decoder2.convblock.2.bias (24,)
decoder1.convblock.0.0.weight (60, 64, 3, 3)
decoder1.convblock.0.0.bias (60,)
decoder1.convblock.0.1.w (60,)
decoder1.convblock.1.conv1.0.weight (60, 60, 3, 3)
decoder1.convblock.1.conv1.0.bias (60,)
decoder1.convblock.1.conv1.1.w (60,)
decoder1.convblock.1.conv2.0.weight (20, 20, 3, 3)
decoder1.convblock.1.conv2.0.bias (20,)
decoder1.convblock.1.conv2.1.w (20,)
decoder1.convblock.1.conv3.0.weight (60, 60, 3, 3)
decoder1.convblock.1.conv3.0.bias (60,)
decoder1.convblock.1.conv3.1.w (60,)
decoder1.convblock.1.conv4.0.weight (20, 20, 3, 3)
decoder1.convblock.1.conv4.0.bias (20,)
decoder1.convblock.1.conv4.1.w (20,)
decoder1.convblock.1.conv5.weight (60, 60, 3, 3)
decoder1.convblock.1.conv5.bias (60,)
decoder1.convblock.1.prelu.w (60,)
decoder1.convblock.2.weight (60, 24, 4, 4)
decoder1.convblock.2.bias (24,)
update4.convc1.weight (64, 392, 1, 1)
update4.convc1.bias (64,)
update4.convf1.weight (40, 4, 7, 7)
update4.convf1.bias (40,)
update4.convf2.weight (20, 40, 3, 3)
update4.convf2.bias (20,)
update4.conv.weight (68, 84, 3, 3)
update4.conv.bias (68,)
update4.gru.0.weight (76, 116, 3, 3)
update4.gru.0.bias (76,)
update4.gru.2.weight (76, 76, 3, 3)
update4.gru.2.bias (76,)
update4.feat_head.0.weight (76, 76, 3, 3)
update4.feat_head.0.bias (76,)
update4.feat_head.2.weight (44, 76, 3, 3)
update4.feat_head.2.bias (44,)
update4.flow_head.0.weight (76, 76, 3, 3)
update4.flow_head.0.bias (76,)
update4.flow_head.2.weight (4, 76, 3, 3)
update4.flow_head.2.bias (4,)
update3.convc1.weight (64, 392, 1, 1)
update3.convc1.bias (64,)
update3.convf1.weight (40, 4, 7, 7)
update3.convf1.bias (40,)
update3.convf2.weight (20, 40, 3, 3)
update3.convf2.bias (20,)
update3.conv.weight (68, 84, 3, 3)
update3.conv.bias (68,)
update3.gru.0.weight (76, 104, 3, 3)
update3.gru.0.bias (76,)
update3.gru.2.weight (76, 76, 3, 3)
update3.gru.2.bias (76,)
update3.feat_head.0.weight (76, 76, 3, 3)
update3.feat_head.0.bias (76,)
update3.feat_head.2.weight (32, 76, 3, 3)
update3.feat_head.2.bias (32,)
update3.flow_head.0.weight (76, 76, 3, 3)
update3.flow_head.0.bias (76,)
update3.flow_head.2.weight (4, 76, 3, 3)
update3.flow_head.2.bias (4,)
update2.convc1.weight (64, 392, 1, 1)
update2.convc1.bias (64,)
update2.convf1.weight (40, 4, 7, 7)
update2.convf1.bias (40,)
update2.convf2.weight (20, 40, 3, 3)
update2.convf2.bias (20,)
update2.conv.weight (68, 84, 3, 3)
update2.conv.bias (68,)
update2.gru.0.weight (76, 92, 3, 3)
update2.gru.0.bias (76,)
update2.gru.2.weight (76, 76, 3, 3)
update2.gru.2.bias (76,)
update2.feat_head.0.weight (76, 76, 3, 3)
update2.feat_head.0.bias (76,)
update2.feat_head.2.weight (20, 76, 3, 3)
update2.feat_head.2.bias (20,)
update2.flow_head.0.weight (76, 76, 3, 3)
update2.flow_head.0.bias (76,)
update2.flow_head.2.weight (4, 76, 3, 3)
update2.flow_head.2.bias (4,)
comb_block.0.weight (18, 9, 3, 3)
comb_block.0.bias (18,)
comb_block.1.w (18,)
comb_block.2.weight (3, 18, 3, 3)
comb_block.2.bias (3,)
'''

# Reference: https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/sample_code.html

# def pytorch2mindspore(default_file = '/root/autodl-tmp/AMT_MindSpore/pretrained/amt-s.pth'):
#     # read pth file
#     par_dict = torch.load(default_file)['state_dict']
#     params_list = []
#     for name in par_dict:
#         param_dict = {}
#         print(name)
#         parameter = par_dict[name]
#         param_dict['name'] = name
#         param_dict['data'] = ms.Tensor(parameter.numpy())
#         params_list.append(param_dict)
#     ms.save_checkpoint(params_list,  'amt-s.ckpt')


def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    # bn_ms2pt = {".w": "weight"}

    new_params_list = []
    for ms_param in ms_params.keys():
        if "norm" in ms_param:
            pass   
        else:
            # 如找到参数对应加入到参数列表
            if ms_param[-1] == 'w':
                # print(ms_param)
                ms_param_before = ms_param
                ms_param = ms_param[:-1] + 'weight' 
                # print(ms_param)
                if ms_param in pt_params:
                    ms_value = pt_params[ms_param]
                    new_params_list.append({"name": ms_param_before, "data": ms.Tensor(ms_value)})

            elif ms_param in pt_params:
                ms_value = pt_params[ms_param]
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)

# ckpt_path = "/root/autodl-tmp/AMT_MindSpore/pretrained/amt-s.ckpt"
ckpt_path = "/root/autodl-tmp/AMT_MindSpore/pretrained/amt-g.ckpt"
# ckpt_path = "/root/autodl-tmp/AMT_MindSpore/pretrained/amt-l.ckpt"

# param_convert(mindspore_params(Model()), pytorch_params('/root/autodl-tmp/AMT_MindSpore/pretrained/amt-s.pth'), ckpt_path)
param_convert(mindspore_params(Model()), pytorch_params('/root/autodl-tmp/AMT_MindSpore/pretrained/amt-g.pth'), ckpt_path)
# param_convert(mindspore_params(Model()), pytorch_params('/root/autodl-tmp/AMT_MindSpore/pretrained/amt-l.pth'), ckpt_path)

# mindspore_params(Model())
# # pytorch_params('/root/autodl-tmp/AMT_MindSpore/pretrained/amt-s.pth')

# pytorch2mindspore()