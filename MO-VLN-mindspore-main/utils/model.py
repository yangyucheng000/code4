import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import nn


zeros = ops.Zeros()
ones = ops.Ones()
shape = ops.Shape()


def get_grid(pose, grid_size):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.shape[0]
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = ops.stack([cos_t, -sin_t,
                           zeros(cos_t.shape, ms.float32)], 1)
    theta12 = ops.stack([sin_t, cos_t,
                           zeros(cos_t.shape, ms.float32)], 1)
    theta1 = ops.stack([theta11, theta12], 1)

    theta21 = ops.stack([ones(x.shape, ms.float32),
                           -zeros(x.shape, ms.float32), x], 1)
    theta22 = ops.stack([zeros(x.shape, ms.float32),
                           ones(x.shape, ms.float32), y], 1)
    theta2 = ops.stack([theta21, theta22], 1)

    rot_grid = ops.affine_grid(theta1, grid_size)
    trans_grid = ops.affine_grid(theta2, grid_size)

    return rot_grid, trans_grid


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = nn.MaxPool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

