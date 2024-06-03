from basicsr.utils.registry import ARCH_REGISTRY
# from torch import nn as nn
# from torch.nn import functional as F
# from torch.nn.utils import spectral_norm
import mindspore as ms
from mindspore import nn, ops
from mindspore.common import initializer
from mindspore.nn import LayerNorm
import numpy as np
# class SpectralNorm(nn.Cell):
#     def __init__(self, weight_shape, num_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.weight_shape = weight_shape
#         self.num_iterations = num_iterations
#         self.u = ms.Parameter(ms.Tensor(np.random.normal(size=(weight_shape[0], 1)), dtype=ms.float32))
#         self.weight = ms.Parameter(initializer('Normal', weight_shape), name='weight')

#     def forward(self, x):
#         w_reshaped = self.weight.view(self.weight_shape[0], -1)
#         u_hat = self.u
#         v_hat = None
#         for _ in range(self.num_iterations):
#             v_hat = F.normalize(mnp.matmul(w_reshaped.T, u_hat), axis=0)
#             u_hat = F.normalize(mnp.matmul(w_reshaped, v_hat), axis=0)
        
#         sigma = mnp.matmul(mnp.matmul(u_hat.T, w_reshaped), v_hat)
#         w_norm = self.weight / sigma
#         return F.matmul(x, w_norm)

# class SpectralNorm(nn.Cell):
#     def __init__(self, wrapped_module, n_power_iterations=1, dim=0):
#         super(SpectralNorm, self).__init__()
#         self.wrapped_module = wrapped_module
#         self.n_power_iterations = n_power_iterations
#         self.dim = dim
#         self._init_params()

#     def _init_params(self):
#         w = getattr(self.wrapped_module, 'weight')
#         height = w.shape[self.dim]
#         self.u = ms.Parameter(ms.Tensor(ms.normal_init((height, 1)), dtype=ms.float32), requires_grad=False)
#         self.wrapped_module.weight.set_data(w)

#     def _update_u_v(self):
#         w = getattr(self.wrapped_module, 'weight')
#         for _ in range(self.n_power_iterations):
#             v = ops.L2Normalize()(ops.matmul(w.transpose(), self.u))
#             self.u.set_data(ops.L2Normalize()(ops.matmul(w, v)))
#         sigma = ops.matmul(ops.matmul(self.u.transpose(), w), v).asnumpy()
#         setattr(self.wrapped_module, 'weight', ms.Parameter(w / sigma, requires_grad=True))

#     def construct(self, x):
#         self._update_u_v()
#         return self.wrapped_module(x)

@ARCH_REGISTRY.register()
# class UNetDiscriminatorSN(nn.Module):
class UNetDiscriminatorSN(nn.Cell):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        # norm = spectral_norm
        norm = LayerNorm()

        # # the first convolution
        # self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # # downsample
        # self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        # self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        # self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # # upsample
        # self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        # self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        # self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # # extra convolutions
        # self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        # self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        # self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, padding=1, pad_mode='pad'))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, padding=1, pad_mode='pad'))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, padding=1, pad_mode='pad'))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, padding=1, pad_mode='pad'))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, padding=1, pad_mode='pad'))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, padding=1, pad_mode='pad'))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, padding=1, pad_mode='pad'))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, padding=1, pad_mode='pad'))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, padding=1, pad_mode='pad')

    def forward(self, x):
        # downsample
        # x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        # x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        # x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # # upsample
        # x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        # x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        # if self.skip_connection:
        #     x4 = x4 + x2
        # x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        # x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        # if self.skip_connection:
        #     x5 = x5 + x1
        # x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        # x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        # if self.skip_connection:
        #     x6 = x6 + x0

        # # extra convolutions
        # out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        # out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        # out = self.conv9(out)

        x0 = ops.leaky_relu(self.conv0(x), alpha=0.2)
        x1 = ops.leaky_relu(self.conv1(x0), alpha=0.2)
        x2 = ops.leaky_relu(self.conv2(x1), alpha=0.2)
        x3 = ops.leaky_relu(self.conv3(x2), alpha=0.2)

        # upsample
        x3 = ops.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x4 = ops.leaky_relu(self.conv4(x3), alpha=0.2)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = ops.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x5 = ops.leaky_relu(self.conv5(x4), alpha=0.2)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = ops.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x6 = ops.leaky_relu(self.conv6(x5), alpha=0.2)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = ops.leaky_relu(self.conv7(x6), alpha=0.2)
        out = ops.leaky_relu(self.conv8(out), alpha=0.2)
        out = self.conv9(out)

        return out