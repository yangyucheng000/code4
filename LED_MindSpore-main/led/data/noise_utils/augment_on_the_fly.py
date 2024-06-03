# import torch
# from torch import nn
import mindspore as ms
from mindspore import nn
# class BasicTransform(nn.Module):
class BasicTransform(nn.Cell):
    def __init__(self, use_hflip, use_rot, batch_aug) -> None:
        super().__init__()
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.batch_aug = batch_aug

    # def augment(self, *datas):
    #     hflip = self.use_hflip and torch.randint(0, 2, (1,)).item() == 1
    #     vflip = self.use_rot and torch.randint(0, 2, (1,)).item() == 1
    #     rot90 = self.use_rot and torch.randint(0, 2, (1,)).item() == 1
    #     if hflip:
    #         datas = [torch.flip(data, (-1,)) for data in datas]
    #     if vflip:
    #         datas = [torch.flip(data, (-2,)) for data in datas]
    #     if rot90:
    #         datas = [torch.transpose(data, -1, -2) for data in datas]
    #     return datas
    def augment(self, *datas):
        hflip = self.use_hflip and ms.ops.randint(0, 2, (1,)).item() == 1
        vflip = self.use_rot and ms.ops.randint(0, 2, (1,)).item() == 1
        rot90 = self.use_rot and ms.ops.randint(0, 2, (1,)).item() == 1
        if hflip:
            datas = [ms.ops.flip(data, (-1,)) for data in datas]
        if vflip:
            datas = [ms.ops.flip(data, (-2,)) for data in datas]
        if rot90:
            datas = [ms.ops.swapaxes(data, -1, -2) for data in datas]
        return datas


    def forward(self, *datas):
        if self.batch_aug:
            return self.augment(*datas)
        B = datas[0].size(0)
        # chunked_datas = [torch.chunk(data, B, dim=0) for data in datas]
        chunked_datas = [ms.ops.chunk(data, B, axis=0) for data in datas]
        # 第三个参数名同
        aug_datas = [self.augment(*data) for data in zip(*chunked_datas)]
        # aug_datas = [torch.cat(data) for data in zip(*aug_datas)]
        aug_datas = [ms.ops.cat(data) for data in zip(*aug_datas)]
        return aug_datas
