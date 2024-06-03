import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Constant
from mindspore import ops, Tensor
import torchsparse_ms
from torchsparse_ms import nn as spnn
from torchsparse_ms import PointTensor, SparseTensor
import torch
import torch.nn as torch_nn
import torchsparse
from torchsparse import nn as torch_spnn

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point, initial_voxelize_torch, point_to_voxel_torch, voxel_to_point_torch

__all__ = ['SPVCNN_MS_TEST']

def save_ouptut_data(name, output):
    print(f"save {name} data: ")
    np.savez(f'./{name}.npz', output=output.asnumpy())
    print("save successfully")

def compare_output_data(name, output, dtype):
    sample = np.load(f"./{name}.npz")
    print("sample.shape: ", sample["output"].shape, "input.dtype: ", sample["output"].dtype)
    output_ori = ms.Tensor(sample["output"], dtype=dtype)
    print(f"compare {name} data: ")
    print(f"output-output_ori: {ops.unique(output - output_ori)[0]}")


class BasicConvolutionBlockTorch(torch_nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            torch_spnn.BatchNorm(outc),
            torch_spnn.ReLU(True),
        )

    def forward(self, x):
        out1 = self.net[0](x)
        out2 = self.net[1](out1)
        out3 = self.net[2](out2)
        return out1, out2, out3


class BasicDeconvolutionBlockTorch(torch_nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            torch_spnn.BatchNorm(outc),
            torch_spnn.ReLU(True),
        )

    def forward(self, x):
        out1 = self.net[0](x)
        out2 = self.net[1](out1)
        out3 = self.net[2](out2)
        return out1, out2, out3


class ResidualBlockTorch(torch_nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.stride = stride
        self.net = torch_nn.Sequential(
            torch_spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            torch_spnn.BatchNorm(outc),
            torch_spnn.ReLU(True),
            torch_spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            torch_spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = torch_nn.Identity()
        else:
            self.downsample = torch_nn.Sequential(
                torch_spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                torch_spnn.BatchNorm(outc),
            )

        self.relu = torch_spnn.ReLU(True)

    def forward(self, x):
        x0 = self.net[0](x)
        x1 = self.net[1](x0)
        x2 = self.net[2](x1)
        x3 = self.net[3](x2)
        x4 = self.net[4](x3)
        if self.inc == self.outc and self.stride == 1:
            ds = self.downsample(x)
        else:
            ds0 = self.downsample[0](x)
            ds = self.downsample[1](ds0)
        out = self.relu(x4 + ds)
        return out

class BasicConvolutionBlock(nn.Cell):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, weight=None):
        super(BasicConvolutionBlock, self).__init__()
        self.net = nn.SequentialCell(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,
                        weight=weight.net[0].kernel.data),
            spnn.BatchNorm(outc),
            spnn.ReLU(),
        )

    def construct(self, x):
        # out = self.net(x)
        out1 = self.net[0](x)
        out2 = self.net[1](out1)
        out3 = self.net[2](out2)
        return out1, out2, out3

class BasicDeconvolutionBlock(nn.Cell):

    def __init__(self, inc, outc, ks=3, stride=1, weight=None):
        super().__init__()
        self.net = nn.SequentialCell(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True,
                        weight=weight.net[0].kernel.data),
            spnn.BatchNorm(outc),
            spnn.ReLU(),
        )

    def construct(self, x):
        out1 = self.net[0](x)
        out2 = self.net[1](out1)
        out3 = self.net[2](out2)
        return out1, out2, out3


class ResidualBlock(nn.Cell):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, weight=None):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.stride = stride
        self.net = nn.SequentialCell(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,
                        weight=weight.net[0].kernel.data),
            spnn.BatchNorm(outc),
            spnn.ReLU(),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1, weight=weight.net[3].kernel.data),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.SequentialCell(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride, weight=weight.downsample[0].kernel.data),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU()

    def construct(self, x):
        # out = self.relu(self.net(x) + self.downsample(x))
        x0 = self.net[0](x)
        x1 = self.net[1](x0)
        x2 = self.net[2](x1)
        x3 = self.net[3](x2)
        x4 = self.net[4](x3)
        if self.inc == self.outc and self.stride == 1:
            ds = self.downsample(x)
        else:
            ds0 = self.downsample[0](x)
            ds = self.downsample[1](ds0)
        out = self.relu(x4 + ds)
        return out


class SPVCNN_MS_TEST(nn.Cell):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.torch_stem = torch_nn.Sequential(
            torch_spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            torch_spnn.BatchNorm(cs[0]),
            torch_spnn.ReLU(True),
            torch_spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            torch_spnn.BatchNorm(cs[0]),
            torch_spnn.ReLU(True)).cuda()

        self.torch_stage1 = torch_nn.Sequential(
            BasicConvolutionBlockTorch(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlockTorch(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlockTorch(cs[1], cs[1], ks=3, stride=1, dilation=1),
        ).cuda()

        self.torch_stage2 = torch_nn.Sequential(
            BasicConvolutionBlockTorch(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlockTorch(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlockTorch(cs[2], cs[2], ks=3, stride=1, dilation=1),
        ).cuda()

        self.torch_stage3 = torch_nn.Sequential(
            BasicConvolutionBlockTorch(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlockTorch(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlockTorch(cs[3], cs[3], ks=3, stride=1, dilation=1),
        ).cuda()

        self.torch_stage4 = torch_nn.Sequential(
            BasicConvolutionBlockTorch(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlockTorch(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlockTorch(cs[4], cs[4], ks=3, stride=1, dilation=1),
        ).cuda()

        self.torch_up1 = torch_nn.ModuleList([
            BasicDeconvolutionBlockTorch(cs[4], cs[5], ks=2, stride=2),
            torch_nn.Sequential(
                ResidualBlockTorch(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlockTorch(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ]).cuda()

        self.torch_up2 = torch_nn.ModuleList([
            BasicDeconvolutionBlockTorch(cs[5], cs[6], ks=2, stride=2),
            torch_nn.Sequential(
                ResidualBlockTorch(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlockTorch(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ]).cuda()

        self.torch_up3 = torch_nn.ModuleList([
            BasicDeconvolutionBlockTorch(cs[6], cs[7], ks=2, stride=2),
            torch_nn.Sequential(
                ResidualBlockTorch(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlockTorch(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ]).cuda()

        self.torch_up4 = torch_nn.ModuleList([
            BasicDeconvolutionBlockTorch(cs[7], cs[8], ks=2, stride=2),
            torch_nn.Sequential(
                ResidualBlockTorch(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlockTorch(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ]).cuda()

        self.torch_classifier = torch_nn.Sequential(torch_nn.Linear(cs[8], kwargs['num_classes'])).cuda()

        self.torch_point_transforms = torch_nn.ModuleList([
            torch_nn.Sequential(
                torch_nn.Linear(cs[0], cs[4]),
                torch_nn.BatchNorm1d(cs[4]),
                torch_nn.ReLU(True),
            ),
            torch_nn.Sequential(
                torch_nn.Linear(cs[4], cs[6]),
                torch_nn.BatchNorm1d(cs[6]),
                torch_nn.ReLU(True),
            ),
            torch_nn.Sequential(
                torch_nn.Linear(cs[6], cs[8]),
                torch_nn.BatchNorm1d(cs[8]),
                torch_nn.ReLU(True),
            )
        ]).cuda()

        self.weight_initializationTorch()
        self.torch_dropout = torch_nn.Dropout(0.3, True).cuda()

        self.stem = nn.SequentialCell(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1, weight=self.torch_stem[0].kernel.data),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1, weight=self.torch_stem[3].kernel.data),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU())

        self.stage1 = nn.SequentialCell(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, weight=self.torch_stage1[0]),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, weight=self.torch_stage1[1]),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, weight=self.torch_stage1[2]),
        )

        self.stage2 = nn.SequentialCell(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, weight=self.torch_stage2[0]),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, weight=self.torch_stage2[1]),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, weight=self.torch_stage2[2]),
        )

        self.stage3 = nn.SequentialCell(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, weight=self.torch_stage3[0]),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, weight=self.torch_stage3[1]),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, weight=self.torch_stage3[2]),
        )

        self.stage4 = nn.SequentialCell(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, weight=self.torch_stage4[0]),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, weight=self.torch_stage4[1]),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, weight=self.torch_stage4[2]),
        )

        self.up1 = nn.CellList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, weight=self.torch_up1[0]),
            nn.SequentialCell(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1, weight=self.torch_up1[1][0]),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, weight=self.torch_up1[1][1]),
            )
        ])

        self.up2 = nn.CellList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, weight=self.torch_up2[0]),
            nn.SequentialCell(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1, weight=self.torch_up2[1][0]),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, weight=self.torch_up2[1][1]),
            )
        ])

        self.up3 = nn.CellList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, weight=self.torch_up3[0]),
            nn.SequentialCell(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1, weight=self.torch_up3[1][0]),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, weight=self.torch_up3[1][1]),
            )
        ])

        self.up4 = nn.CellList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2,  weight=self.torch_up4[0]),
            nn.SequentialCell(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1, weight=self.torch_up4[1][0]),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, weight=self.torch_up4[1][1]),
            )
        ])

        self.classifier = nn.SequentialCell(nn.Dense(in_channels=cs[8], out_channels=kwargs['num_classes'], weight_init=None, bias_init=None))

        self.point_transforms = nn.CellList([
            nn.SequentialCell(
                nn.Dense(cs[0], cs[4], weight_init=Tensor(self.torch_point_transforms[0][0].weight.detach().cpu().numpy()), bias_init=Tensor(self.torch_point_transforms[0][0].bias.detach().cpu().numpy())),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(),
            ),
            nn.SequentialCell(
                nn.Dense(cs[4], cs[6], weight_init=Tensor(self.torch_point_transforms[1][0].weight.detach().cpu().numpy()), bias_init=Tensor(self.torch_point_transforms[1][0].bias.detach().cpu().numpy())),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(),
            ),
            nn.SequentialCell(
                nn.Dense(cs[6], cs[8], weight_init=Tensor(self.torch_point_transforms[2][0].weight.detach().cpu().numpy()), bias_init=Tensor(self.torch_point_transforms[2][0].bias.detach().cpu().numpy())),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(p=0.3)

    def weight_initialization(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm1d):
                cell.gamma.set_data(initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer("zeros", cell.beta.shape, cell.beta.dtype))

    def weight_initializationTorch(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, torch_nn.BatchNorm1d):
                torch_nn.init.constant_(cell.weight, 1)
                torch_nn.init.constant_(cell.bias, 0)

    def construct(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        z_torch = torchsparse.PointTensor(torch.tensor(x.F.asnumpy()),
                                    torch.tensor(x.C.float().asnumpy())).cuda()
        #
        print(f'------------------init voxelize--------------------')

        x0 = initial_voxelize(z, self.pres, self.vres)
        x0_torch = initial_voxelize_torch(z_torch, self.pres, self.vres)

        print(f'unique(x0.F - x0_torch.F): '
              f'{np.unique(x0.F.asnumpy() - x0_torch.F.detach().cpu().numpy())}')
        print(f'unique(x0.F - x0_torch.F).shape: '
              f'{np.unique(x0.F.asnumpy() - x0_torch.F.detach().cpu().numpy()).shape}')
        print(f'unique(x0.C - x0_torch.C): '
              f'{np.unique(x0.C.asnumpy() - x0_torch.C.detach().cpu().numpy())}')
        print(f'unique(x0.C - x0_torch.C).shape: '
              f'{np.unique(x0.C.asnumpy() - x0_torch.C.detach().cpu().numpy()).shape}')

        print(f'---------------------------------------------------')

        print(f'------------------stem--------------------')
        stem0_x0 = self.stem[0](x0)
        stem1_x0 = self.stem[1](stem0_x0)
        stem2_x0 = self.stem[2](stem1_x0)
        stem3_x0 = self.stem[3](stem2_x0)
        stem4_x0 = self.stem[4](stem3_x0)
        stem5_x0 = self.stem[5](stem4_x0)

        stem0_torch_x0 = self.torch_stem[0](x0_torch)
        stem1_torch_x0 = self.torch_stem[1](stem0_torch_x0)
        stem2_torch_x0 = self.torch_stem[2](stem1_torch_x0)
        stem3_torch_x0 = self.torch_stem[3](stem2_torch_x0)
        stem4_torch_x0 = self.torch_stem[4](stem3_torch_x0)
        stem5_torch_x0 = self.torch_stem[5](stem4_torch_x0)

        print(f'unique(stem0_x0.F - stem0_torch_x0.F): '
              f'{np.unique(stem0_x0.F.asnumpy() - stem0_torch_x0.F.detach().cpu().numpy())}')
        print(f'unique(stem0_x0.F - stem0_torch_x0.F).shape: '
              f'{np.unique(stem0_x0.F.asnumpy() - stem0_torch_x0.F.detach().cpu().numpy()).shape}')
        print(f'unique(stem1_x0.F - stem1_torch_x0.F): '
              f'{np.unique(stem1_x0.F.asnumpy() - stem1_torch_x0.F.detach().cpu().numpy())}')
        print(f'unique(stem1_x0.F - stem1_torch_x0.F).shape: '
              f'{np.unique(stem1_x0.F.asnumpy() - stem1_torch_x0.F.detach().cpu().numpy()).shape}')
        print(f'unique(stem2_x0.F - stem2_torch_x0.F): '
              f'{np.unique(stem2_x0.F.asnumpy() - stem2_torch_x0.F.detach().cpu().numpy())}')
        print(f'unique(stem2_x0.F - stem2_torch_x0.F).shape: '
              f'{np.unique(stem2_x0.F.asnumpy() - stem2_torch_x0.F.detach().cpu().numpy()).shape}')

        print(f'unique(stem3_x0.F - stem3_torch_x0.F): '
              f'{np.unique(stem3_x0.F.asnumpy() - stem3_torch_x0.F.detach().cpu().numpy())}')
        print(f'unique(stem3_x0.F - stem3_torch_x0.F).shape: '
              f'{np.unique(stem3_x0.F.asnumpy() - stem3_torch_x0.F.detach().cpu().numpy()).shape}')

        print(f'unique(stem4_x0.F - stem4_torch_x0.F): '
              f'{np.unique(stem4_x0.F.asnumpy() - stem4_torch_x0.F.detach().cpu().numpy())}')
        print(f'unique(stem4_x0.C - stem4_torch_x0.C): '
              f'{np.unique(stem4_x0.C.asnumpy() - stem4_torch_x0.C.detach().cpu().numpy())}')
        print(f'unique(stem5_x0.F - stem5_torch_x0.F): '
              f'{np.unique(stem5_x0.F.asnumpy() - stem5_torch_x0.F.detach().cpu().numpy())}')
        print(f'unique(stem5_x0.C - stem5_torch_x0.C): '
              f'{np.unique(stem5_x0.C.asnumpy() - stem5_torch_x0.C.detach().cpu().numpy())}')

        print(f'-------------------------------------------')

        print(f'------------------voxel to point--------------------')

        z0, _, _, _ = voxel_to_point(stem5_x0, z, nearest=False)
        z0_torch, _, _, _ = voxel_to_point_torch(stem5_torch_x0, z_torch, nearest=False)

        print(f'unique(z0.F - z0_torch.F): '
              f'{np.unique(z0.F.asnumpy() - z0_torch.F.detach().cpu().numpy())}')
        print(f'unique(z0.F - z0_torch.F).shape: '
              f'{np.unique(z0.F.asnumpy() - z0_torch.F.detach().cpu().numpy()).shape}')

        print(f'----------------------------------------------------')
        #
        print(f'------------------point to voxel--------------------')

        x1 = point_to_voxel(x0, z0)
        x1_torch = point_to_voxel_torch(x0_torch, z0_torch)

        print(f'unique(x1.F - x1_torch.F): '
              f'{np.unique(x1.F.asnumpy() - x1_torch.F.detach().cpu().numpy())}')
        print(f'unique(x1.F - x1_torch.F).shape: '
              f'{np.unique(x1.F.asnumpy() - x1_torch.F.detach().cpu().numpy()).shape}')

        print(f'----------------------------------------------------')

        print(f'------------------stage1-------------------')

        stage0_0_x1, stage0_1_x1, stage0_2_x1 = self.stage1[0](x1)
        stage1_x1 = self.stage1[1](stage0_2_x1)
        stage2_x1 = self.stage1[2](stage1_x1)

        stage0_0_x2, stage0_1_x2, stage0_2_x2 = self.stage2[0](stage2_x1)
        stage1_x2 = self.stage2[1](stage0_2_x2)
        stage2_x2 = self.stage2[2](stage1_x2)

        stage0_0_x3, stage0_1_x3, stage0_2_x3 = self.stage3[0](stage2_x2)
        stage1_x3 = self.stage3[1](stage0_2_x3)
        stage2_x3 = self.stage3[2](stage1_x3)

        stage0_0_x4, stage0_1_x4, stage0_2_x4 = self.stage4[0](stage2_x3)
        stage1_x4 = self.stage4[1](stage0_2_x4)
        stage2_x4 = self.stage4[2](stage1_x4)

        stage0_0_torch_x1, stage0_1_torch_x1, stage0_2_torch_x1 = self.torch_stage1[0](x1_torch)
        stage1_torch_x1 = self.torch_stage1[1](stage0_2_torch_x1)
        stage2_torch_x1 = self.torch_stage1[2](stage1_torch_x1)

        stage0_0_torch_x2, stage0_1_torch_x2, stage0_2_torch_x2 = self.torch_stage2[0](stage2_torch_x1)
        stage1_torch_x2 = self.torch_stage2[1](stage0_2_torch_x2)
        stage2_torch_x2 = self.torch_stage2[2](stage1_torch_x2)

        stage0_0_torch_x3, stage0_1_torch_x3, stage0_2_torch_x3 = self.torch_stage3[0](stage2_torch_x2)
        stage1_torch_x3 = self.torch_stage3[1](stage0_2_torch_x3)
        stage2_torch_x3 = self.torch_stage3[2](stage1_torch_x3)

        stage0_0_torch_x4, stage0_1_torch_x4, stage0_2_torch_x4 = self.torch_stage4[0](stage2_torch_x3)
        stage1_torch_x4 = self.torch_stage4[1](stage0_2_torch_x4)
        stage2_torch_x4 = self.torch_stage4[2](stage1_torch_x4)


        print(f'unique(stage0_0_x1.F - stage0_0_torch_x1.F): '
              f'{np.unique(stage0_0_x1.F.asnumpy() - stage0_0_torch_x1.F.detach().cpu().numpy())}')
        print(f'unique(stage0_0_x1.F - stage0_0_torch_x1.F).shape: '
              f'{np.unique(stage0_0_x1.F.asnumpy() - stage0_0_torch_x1.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_1_x1.F - stage0_1_torch_x1.F): '
              f'{np.unique(stage0_1_x1.F.asnumpy() - stage0_1_torch_x1.F.detach().cpu().numpy())}')
        print(f'unique(stage0_1_x1.F - stage0_1_torch_x1.F).shape: '
              f'{np.unique(stage0_1_x1.F.asnumpy() - stage0_1_torch_x1.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_2_x1.F - stage0_2_torch_x1.F): '
              f'{np.unique(stage0_2_x1.F.asnumpy() - stage0_2_torch_x1.F.detach().cpu().numpy())}')
        print(f'unique(stage0_2_x1.F - stage0_2_torch_x1.F).shape: '
              f'{np.unique(stage0_2_x1.F.asnumpy() - stage0_2_torch_x1.F.detach().cpu().numpy()).shape}')

        print(f'unique(stage0_0_x2.F - stage0_0_torch_x2.F): '
              f'{np.unique(stage0_0_x2.F.asnumpy() - stage0_0_torch_x2.F.detach().cpu().numpy())}')
        print(f'unique(stage0_0_x2.F - stage0_0_torch_x2.F).shape: '
              f'{np.unique(stage0_0_x2.F.asnumpy() - stage0_0_torch_x2.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_1_x2.F - stage0_1_torch_x2.F): '
              f'{np.unique(stage0_1_x2.F.asnumpy() - stage0_1_torch_x2.F.detach().cpu().numpy())}')
        print(f'unique(stage0_1_x2.F - stage0_1_torch_x2.F).shape: '
              f'{np.unique(stage0_1_x2.F.asnumpy() - stage0_1_torch_x2.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_2_x2.F - stage0_2_torch_x2.F): '
              f'{np.unique(stage0_2_x2.F.asnumpy() - stage0_2_torch_x2.F.detach().cpu().numpy())}')
        print(f'unique(stage0_2_x2.F - stage0_2_torch_x2.F).shape: '
              f'{np.unique(stage0_2_x2.F.asnumpy() - stage0_2_torch_x2.F.detach().cpu().numpy()).shape}')

        print(f'unique(stage0_0_x3.F - stage0_0_torch_x3.F): '
              f'{np.unique(stage0_0_x3.F.asnumpy() - stage0_0_torch_x3.F.detach().cpu().numpy())}')
        print(f'unique(stage0_0_x3.F - stage0_0_torch_x3.F).shape: '
              f'{np.unique(stage0_0_x3.F.asnumpy() - stage0_0_torch_x3.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_1_x3.F - stage0_1_torch_x3.F): '
              f'{np.unique(stage0_1_x3.F.asnumpy() - stage0_1_torch_x3.F.detach().cpu().numpy())}')
        print(f'unique(stage0_1_x3.F - stage0_1_torch_x3.F).shape: '
              f'{np.unique(stage0_1_x3.F.asnumpy() - stage0_1_torch_x3.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_2_x3.F - stage0_2_torch_x3.F): '
              f'{np.unique(stage0_2_x3.F.asnumpy() - stage0_2_torch_x3.F.detach().cpu().numpy())}')
        print(f'unique(stage0_2_x3.F - stage0_2_torch_x3.F).shape: '
              f'{np.unique(stage0_2_x3.F.asnumpy() - stage0_2_torch_x3.F.detach().cpu().numpy()).shape}')

        print(f'unique(stage0_0_x4.F - stage0_0_torch_x4.F): '
              f'{np.unique(stage0_0_x4.F.asnumpy() - stage0_0_torch_x4.F.detach().cpu().numpy())}')
        print(f'unique(stage0_0_x4.F - stage0_0_torch_x4.F).shape: '
              f'{np.unique(stage0_0_x4.F.asnumpy() - stage0_0_torch_x4.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_1_x4.F - stage0_1_torch_x4.F): '
              f'{np.unique(stage0_1_x4.F.asnumpy() - stage0_1_torch_x4.F.detach().cpu().numpy())}')
        print(f'unique(stage0_1_x4.F - stage0_1_torch_x4.F).shape: '
              f'{np.unique(stage0_1_x4.F.asnumpy() - stage0_1_torch_x4.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_2_x4.F - stage0_2_torch_x4.F): '
              f'{np.unique(stage0_2_x4.F.asnumpy() - stage0_2_torch_x4.F.detach().cpu().numpy())}')
        print(f'unique(stage0_2_x4.F - stage0_2_torch_x4.F).shape: '
              f'{np.unique(stage0_2_x4.F.asnumpy() - stage0_2_torch_x4.F.detach().cpu().numpy()).shape}')
        print(f'unique(stage0_2_x4.C - stage0_2_torch_x4.C): '
              f'{np.unique(stage0_2_x4.C.asnumpy() - stage0_2_torch_x4.C.detach().cpu().numpy())}')
        print(f'stage0_2_x4.S: {stage0_2_x4.s}')
        print(f'stage0_2_torch_x4.S: {stage0_2_torch_x4.s}')


        print(f'-------------------------------------------')

        print(f'------------------voxel to point--------------------')

        z1, idx_query, weights, new_feat = voxel_to_point(stage2_x4, z0, nearest=False)
        z1_torch, idx_query_torch, weights_torch, new_feat_torch = voxel_to_point_torch(stage2_torch_x4, z0_torch, nearest=False)

        print(f'unique(z1.F - z1_torch.F): '
              f'{np.unique(z1.F.asnumpy() - z1_torch.F.detach().cpu().numpy())}')
        print(f'unique(z0.C - z0_torch.C): '
              f'{np.unique(z0.C.asnumpy() - z0_torch.C.detach().cpu().numpy())}')
        print(f'unique(z1.F - z1_torch.F).shape: '
              f'{np.unique(z1.F.asnumpy() - z1_torch.F.detach().cpu().numpy()).shape}')

        print(f'unique(idx_query - idx_query_torch): '
              f'{np.unique(idx_query.asnumpy() - idx_query_torch.detach().cpu().numpy())}')
        print(f'unique(weights - weights_torch): '
              f'{np.unique(weights.asnumpy() - weights_torch.detach().cpu().numpy())}')
        print(f'unique(new_feat - new_feat_torch): '
              f'{np.unique(new_feat.asnumpy() - new_feat_torch.detach().cpu().numpy())}')


        print(f'----------------------------------------------------')

        print(f'------------------point_transforms--------------------')

        # z1 = z1_torch.detach().cpu().numpy()

        transforms0_z1 = self.point_transforms[0][0](z0.F)
        transforms1_z1 = self.point_transforms[0][1](transforms0_z1)
        transforms2_z1 = self.point_transforms[0][2](transforms1_z1)

        transforms0_z1_torch = self.torch_point_transforms[0][0](z0_torch.F)
        transforms1_z1_torch = self.torch_point_transforms[0][1](transforms0_z1_torch)
        transforms2_z1_torch = self.torch_point_transforms[0][2](transforms1_z1_torch)

        z1.F = z1.F + transforms2_z1
        z1_torch.F = z1_torch.F + transforms2_z1_torch

        print(f'unique(transforms0_z1 - transforms0_z1_torch): '
              f'{np.unique(transforms0_z1.asnumpy() - transforms0_z1_torch.detach().cpu().numpy())}')
        print(f'unique(transforms1_z1 - transforms1_z1_torch): '
              f'{np.unique(transforms1_z1.asnumpy() - transforms1_z1_torch.detach().cpu().numpy())}')
        print(f'unique(transforms2_z1 - transforms2_z1_torch): '
              f'{np.unique(transforms2_z1.asnumpy() - transforms2_z1_torch.detach().cpu().numpy())}')

        print(f'unique(z1.F - z1_torch.F): '
              f'{np.unique(z1.F.asnumpy() - z1_torch.F.detach().cpu().numpy())}')
        print(f'unique(z1.F - z1_torch.F).shape: '
              f'{np.unique(z1.F.asnumpy() - z1_torch.F.detach().cpu().numpy()).shape}')

        print(f'------------------------------------------------------')


        exit()

        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        print(f'z0.F: {z0.F}')
        print(f'z0.F.shape: {z0.F.shape}')







        x1 = point_to_voxel(x0, z0)
        # print(f'x1.F: {x1.F}')
        print(f'x1.F.shape: {x1.F.shape}')
        x1 = self.stage1(x1)
        # print(f'x1.2.F: {x1.F}')
        print(f'x1.2.F.shape: {x1.F.shape}')

        x2 = self.stage2(x1)
        # print(f'x2.F: {x2.F}')
        print(f'x2.F.shape: {x2.F.shape}')
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        # print(f'z1.F: {z1.F}')
        print(f'z1.F.shape: {z1.F.shape}')

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse_ms.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse_ms.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        # print(f'z2.F: {z2.F}')
        print(f'z2.F.shape: {z2.F.shape}')

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse_ms.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse_ms.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        # print(f'z3.F: {z3.F}')
        print(f'z3.F.shape: {z3.F.shape}')

        out = self.classifier(z3.F)
        return out

# class SPVCNN_MS(nn.Cell):
#     def __init__(self, **kwargs):
#         super().__init__()
#
#         cr = kwargs.get('cr', 1.0)
#         cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
#         cs = [int(cr * x) for x in cs]
#
#         if 'pres' in kwargs and 'vres' in kwargs:
#             self.pres = kwargs['pres']
#             self.vres = kwargs['vres']
#
#         self.net = nn.SequentialCell(
#             spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
#             spnn.BatchNorm(cs[0]),
#             spnn.ReLU())
#
#         self.classifier = nn.SequentialCell([nn.Dense(cs[0], kwargs['num_classes'])])
#
#     def construct(self, x):
#         print(f"net.input.data: {x.F}")
#         print(f"net.input.data.shape: {x.F.shape}, net.input.data.dtype: {x.F.dtype}")
#
#
#         z = PointTensor(x.F, x.C.astype('float32'))
#
#         print(f"net.input.pointtensor: {z.F}")
#         print(f"net.input.pointtensor.shape: {z.F.shape}")
#
#         print(f"before initial_voxelize")
#         x0 = initial_voxelize(z, self.pres, self.vres)
#         print(f"iniial_voxelize success")
#
#         print(f"net.voxelize: {x0.F}")
#         print(f"net.voxelize.shape: {x0.F.shape}")
#
#         x1 = self.net(x0)
#         z0 = voxel_to_point(x1, z, nearest=False)
#
#         # print(f"net.conv3d: {z0.F}")
#         # print(f"net.conv3d.shape: {z0.F.shape}")
#         # print(f"conv3d success")
#
#         out = self.classifier(z0.F)
#         return out