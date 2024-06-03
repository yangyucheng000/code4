import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import ops
from mindspore import Tensor
from torchsparse_ms.nn.cuda.devoxelize import SPDevoxelizeForward, SPDevoxelizeBackward

import torch
import numpy as np

__all__ = ['spdevoxelize', 'calc_ti_weights']


def calc_ti_weights(coords: ms.Tensor,
                    idx_query: ms.Tensor,
                    scale: float = 1) -> ms.Tensor:
    p = coords
    # p_torch = torch.tensor(coords.asnumpy()).cuda()
    # idx_query_torch = torch.tensor(idx_query.asnumpy()).cuda()
    F.stop_gradient(p)
    if scale != 1:
        pf = ops.floor(coords / scale) * scale
        # pf_torch = torch.floor(p_torch / scale) * scale
    else:
        pf = ops.floor(coords)
        # pf_torch = torch.floor(p_torch)
    pc = pf + scale
    # pc_torch = pf_torch + scale
    # print(f'unique(pc - pc_torch): '
    #       f'{np.unique(pc.asnumpy() - pc_torch.detach().cpu().numpy())}')

    x = p[:, 0].view(-1, 1)
    y = p[:, 1].view(-1, 1)
    z = p[:, 2].view(-1, 1)
    # x_torch = p_torch[:, 0].view(-1, 1)
    # y_torch = p_torch[:, 1].view(-1, 1)
    # z_torch = p_torch[:, 2].view(-1, 1)

    xf = pf[:, 0].view(-1, 1).astype(ms.float32)
    yf = pf[:, 1].view(-1, 1).astype(ms.float32)
    zf = pf[:, 2].view(-1, 1).astype(ms.float32)
    # xf_torch = pf_torch[:, 0].view(-1, 1).float()
    # yf_torch = pf_torch[:, 1].view(-1, 1).float()
    # zf_torch = pf_torch[:, 2].view(-1, 1).float()

    xc = pc[:, 0].view(-1, 1).astype(ms.float32)
    yc = pc[:, 1].view(-1, 1).astype(ms.float32)
    zc = pc[:, 2].view(-1, 1).astype(ms.float32)
    # xc_torch = pc_torch[:, 0].view(-1, 1).float()
    # yc_torch = pc_torch[:, 1].view(-1, 1).float()
    # zc_torch = pc_torch[:, 2].view(-1, 1).float()

    # print(f'unique(xc - xc_torch): '
    #       f'{np.unique(xc.asnumpy() - xc_torch.detach().cpu().numpy())}')
    # print(f'unique(yc - yc_torch): '
    #       f'{np.unique(yc.asnumpy() - yc_torch.detach().cpu().numpy())}')
    # print(f'unique(zc - zc_torch): '
    #       f'{np.unique(zc.asnumpy() - zc_torch.detach().cpu().numpy())}')

    w0 = (xc - x) * (yc - y) * (zc - z)
    # w0_torch = (xc_torch - x_torch) * (yc_torch - y_torch) * (zc_torch -z_torch)
    w1 = (xc - x) * (yc - y) * (z - zf)
    # w1_torch = (xc_torch - x_torch) * (yc_torch - y_torch) * (z_torch - zf_torch)
    w2 = (xc - x) * (y - yf) * (zc - z)
    # w2_torch = (xc_torch - x_torch) * (y_torch - yf_torch) * (zc_torch - z_torch)
    w3 = (xc - x) * (y - yf) * (z - zf)
    # w3_torch = (xc_torch - x_torch) * (y_torch - yf_torch) * (z_torch - zf_torch)
    w4 = (x - xf) * (yc - y) * (zc - z)
    # w4_torch = (x_torch - xf_torch) * (yc_torch - y_torch) * (zc_torch - z_torch)
    w5 = (x - xf) * (yc - y) * (z - zf)
    # w5_torch = (x_torch - xf_torch) * (yc_torch - y_torch) * (z_torch - zf_torch)
    w6 = (x - xf) * (y - yf) * (zc - z)
    # w6_torch = (x_torch - xf_torch) * (y_torch - yf_torch) * (zc_torch - z_torch)
    w7 = (x - xf) * (y - yf) * (z - zf)
    # w7_torch = (x_torch - xf_torch) * (y_torch - yf_torch) * (z_torch - zf_torch)

    w = ops.concat([w0, w1, w2, w3, w4, w5, w6, w7], axis=1)
    w = w.transpose(1, 0)
    # w_torch = torch.cat([w0_torch, w1_torch, w2_torch, w3_torch, w4_torch, w5_torch, w6_torch, w7_torch], dim=1).transpose(1, 0).contiguous()
    # print(f'w1: {w}, dtype: {w.dtype}')
    # print(f'w1_torch: {w_torch}, dtype: {w_torch.dtype}')
    # print(f'unique(w1 - w1_torch): '
    #       f'{np.unique(w.asnumpy() - w_torch.detach().cpu().numpy())}')
    if scale != 1:
        w /= scale ** 3
        # w_torch /= scale ** 3
    # print(f'w2: {w}, dtype: {w.dtype}')
    # print(f'unique(w2 - w2_torch): '
    #       f'{np.unique(w.asnumpy() - w_torch.detach().cpu().numpy())}')
    w[idx_query == -1] = 0
    # w_torch[idx_query_torch == -1] = 0
    # print(f'w3: {w}, dtype: {w.dtype}')
    # print(f'unique(w3 - w3_torch): '
    #       f'{np.unique(w.asnumpy() - w_torch.detach().cpu().numpy())}')

    w_sum = ops.sum(w, dim=0)
    # w_sum_torch = torch.sum(w_torch, dim=0)
    # print(f'unique(w_sum - w_sum_torch): '
    #       f'{np.unique(w_sum.asnumpy() - w_sum_torch.detach().cpu().numpy())}')

    # if not np.array_equal(np.unique(w_sum.asnumpy() - w_sum_torch.detach().cpu().numpy()),  np.zeros(1)):
    #     print(f'**************************************')
    #     print("------------------------------")
    #     print("save data: ")
    #     np.savez('./ops_sum.npz',
    #              ms=w.asnumpy(), torch=w_torch.detach().cpu().numpy())
    #     print("save successfully")
    #     print("------------------------------")


    w /= w_sum + 1e-8
    # w_torch /= w_sum_torch + 1e-8
    F.stop_gradient(w)

    return w


class DevoxelizeFunction(nn.Cell):
    def __init__(self):
        super(DevoxelizeFunction, self).__init__()
        self.sp_devoxelize_forward = SPDevoxelizeForward()
        self.sp_devoxelize_backward = SPDevoxelizeBackward()

    def construct(self, feats: Tensor, coords: Tensor,
                weights: Tensor) -> Tensor:

        if ms.get_context("device_target") == 'GPU':
            output = self.sp_devoxelize_forward(
                feats, coords.astype(ms.int32), weights)
        else:
            raise NotImplementedError

        return output


    # def bprop(self, feats: Tensor, coords: Tensor,
    #             weights: Tensor, output: Tensor, grad_output: Tensor):
    #
    #     if ms.get_context("device_target") == 'GPU':
    #         grad_feats = self.sp_devoxelize_backward(
    #             grad_output, coords.astype(ms.int32), weights, feats.shape[0])
    #     else:
    #         raise NotImplementedError
    #     return grad_feats, None, None


def spdevoxelize(feats: Tensor, coords: Tensor,
                 weights: Tensor) -> Tensor:
    return DevoxelizeFunction()(feats, coords, weights)
