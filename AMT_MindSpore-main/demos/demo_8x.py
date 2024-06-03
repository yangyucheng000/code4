"""
a simple 8x demo (please use the weights `gopro_amt-s.pth`)

    > Interpolating eight intermediate frames

    ```bash
    python demos/demo_8x.py -c [CFG] -p [CKPT_PATH] -x [IMG0] -y [IMG1] -o [OUT_PATH]
    ```

    - Results are in the `[OUT_PATH]` (default is `results/8x`) folder.

"""

import os
import sys
import tqdm
# import torch
import mindspore as ms
from mindspore import ops, Tensor
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
# from torchvision.utils import make_grid
import mindspore.numpy as mnp
sys.path.append('.')
from utils.utils import read, img2tensor
from utils.utils import (
    read, write,
    img2tensor, tensor2img
    )
from utils.build_utils import build_from_cfg
from utils.utils import InputPadder

def make_grid_mindspore(tensor, nrow=8, padding=2, normalize=False, value_range=None, pad_value=0.0):
    """
    在MindSpore中创建图像的网格。

    参数:
    - tensor (Tensor): 形状为(B, C, H, W)的张量，代表一批图像。
    - nrow (int): 每行显示的图像数量。
    - padding (int): 图像间的填充大小。
    - normalize (bool): 是否将图像标准化到[0, 1]。
    - value_range (tuple): 用于标准化的最小值和最大值。
    - pad_value (float): 填充的像素值。

    返回:
    - grid (Tensor): 包含图像网格的张量。
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("输入必须是MindSpore张量")

    B, C, H, W = tensor.shape

    # 计算网格的大小
    rows = np.ceil(B / nrow).astype(int)
    cols = min(B, nrow)

    grid_height = rows * H + (rows - 1) * padding
    grid_width = cols * W + (cols - 1) * padding

    # 转换为整数，确保张量形状参数是整数类型
    grid_height = int(grid_height)
    grid_width = int(grid_width)

    grid = mnp.ones((C, grid_height, grid_width)) * pad_value
    for i in range(B):
        row = i // nrow
        col = i % nrow
        h_start = row * (H + padding)
        w_start = col * (W + padding)
        grid[:, h_start:h_start+H, w_start:w_start+W] = tensor[i]

    if normalize:
        if value_range is not None:
            min_val, max_val = value_range
            grid = (grid - min_val) / (max_val - min_val)
        grid = mnp.clip(grid, 0, 1)

    return grid

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'Demo 8x',
                )
parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml') 
parser.add_argument('-p', '--ckpt', default='pretrained/gopro_amt-s.pth') 
parser.add_argument('-x', '--img0', default='assets/quick_demo/img0.png') 
parser.add_argument('-y', '--img1', default='assets/quick_demo/img1.png') 
parser.add_argument('-o', '--out_path', default='results/8x') 
args = parser.parse_args()

# ----------------------- Initialization ----------------------- 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'GPU'
ms.set_context(device_target=device, pynative_synchronize=True)

cfg_path = args.config
ckpt_path = args.ckpt
img0_path = args.img0
img1_path = args.img1
out_path = args.out_path
if osp.exists(out_path) is False:
    os.makedirs(out_path)

# -----------------------  Load model ----------------------- 
network_cfg = OmegaConf.load(cfg_path).network
network_name = network_cfg.name
print(f'Loading [{network_name}] from [{ckpt_path}]...')
model = build_from_cfg(network_cfg)
# ckpt = torch.load(ckpt_path)
# model.load_state_dict(ckpt['state_dict'])
param_dict = ms.load_checkpoint(ckpt_path)
param_not_load, _ = ms.load_param_into_net(model, param_dict)
print(param_not_load)

# model = model.to(device)
# model.eval()
model.set_train(False)

# -----------------------  Load input frames ----------------------- 
img0 = read(img0_path)
img1 = read(img1_path)
# img0_t = img2tensor(img0).to(device)
# img1_t = img2tensor(img1).to(device)
img0_t = img2tensor(img0)
img1_t = img2tensor(img1)
padder = InputPadder(img0_t.shape, 16)
img0_t, img1_t = padder.pad(img0_t, img1_t)
# embt = torch.arange(1/8, 1, 1/8).float().view(1, 7, 1, 1).to(device)
embt = ops.arange(1/8, 1, 1/8).float().view(1, 7, 1, 1)

# -----------------------  Interpolater ----------------------- 
imgt_preds = []
for i in range(7):
    # with torch.no_grad():
    imgt_pred = model(img0_t, img1_t, embt[:, i: i + 1, ...], eval=True)['imgt_pred']
    imgt_pred = padder.unpad(imgt_pred)
    # imgt_preds.append(imgt_pred.detach())
    imgt_preds.append(imgt_pred)

# concat_img = torch.cat([img0_t, *imgt_preds, img1_t], 0)
concat_img = ops.cat([img0_t, *imgt_preds, img1_t], 0)
concat_img = make_grid_mindspore(concat_img, nrow=3)
concat_img = tensor2img(concat_img)
write(f'{out_path}/grid.png', concat_img)

# -----------------------  Write generate frames to disk ----------------------- 
for i, imgt_pred in enumerate(imgt_preds):
    imgt_pred = tensor2img(imgt_pred)
    write(f'{out_path}/imgt_pred_{i}.png', imgt_pred)