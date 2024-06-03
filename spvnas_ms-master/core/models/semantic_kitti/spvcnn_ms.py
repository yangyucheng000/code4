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

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN_MS']

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
        out = self.net(x)
        return out


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
        out = self.net(x)
        return out


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
        out = self.relu(self.net(x) + self.downsample(x))
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
        out = self.net(x)
        return out

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
        return self.net(x)


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
            spnn.ReLU(),
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
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN_MS(nn.Cell):

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


        # self.stem = nn.SequentialCell(
        #     spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
        #     spnn.BatchNorm(cs[0]),
        #     spnn.ReLU(),
        #     spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
        #     spnn.BatchNorm(cs[0]),
        #     spnn.ReLU())
        #
        # self.stage1 = nn.SequentialCell(
        #     BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
        #     ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
        #     ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        # )
        #
        # self.stage2 = nn.SequentialCell(
        #     BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
        #     ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
        #     ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        # )
        #
        # self.stage3 = nn.SequentialCell(
        #     BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
        #     ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
        #     ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        # )
        #
        # self.stage4 = nn.SequentialCell(
        #     BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
        #     ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
        #     ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        # )
        #
        # self.up1 = nn.CellList([
        #     BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
        #     nn.SequentialCell(
        #         ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
        #         ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
        #     )
        # ])
        #
        # self.up2 = nn.CellList([
        #     BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
        #     nn.SequentialCell(
        #         ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
        #         ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
        #     )
        # ])
        #
        # self.up3 = nn.CellList([
        #     BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
        #     nn.SequentialCell(
        #         ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
        #         ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
        #     )
        # ])
        #
        # self.up4 = nn.CellList([
        #     BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
        #     nn.SequentialCell(
        #         ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
        #         ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
        #     )
        # ])
        #
        # self.classifier = nn.SequentialCell(nn.Dense(in_channels=cs[8], out_channels=kwargs['num_classes'], weight_init=None, bias_init=None))
        #
        # self.point_transforms = nn.CellList([
        #     nn.SequentialCell(
        #         nn.Dense(cs[0], cs[4], weight_init=None, bias_init=None),
        #         nn.BatchNorm1d(cs[4]),
        #         nn.ReLU(),
        #     ),
        #     nn.SequentialCell(
        #         nn.Dense(cs[4], cs[6], weight_init=None, bias_init=None),
        #         nn.BatchNorm1d(cs[6]),
        #         nn.ReLU(),
        #     ),
        #     nn.SequentialCell(
        #         nn.Dense(cs[6], cs[8], weight_init=None, bias_init=None),
        #         nn.BatchNorm1d(cs[8]),
        #         nn.ReLU(),
        #     )
        # ])
        #
        # self.weight_initialization()
        # self.dropout = nn.Dropout(p=0.3)

        self.stem = nn.SequentialCell(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1, weight=self.torch_stem[0].kernel.data),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1, weight=self.torch_stem[3].kernel.data),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU())

        self.dense1 = nn.Dense(4, cs[0])

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
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, weight=self.torch_up4[0]),
            nn.SequentialCell(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1, weight=self.torch_up4[1][0]),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, weight=self.torch_up4[1][1]),
            )
        ])

        self.classifier = nn.SequentialCell(
            nn.Dense(in_channels=cs[8], out_channels=kwargs['num_classes'], weight_init=None, bias_init=None))

        self.point_transforms = nn.CellList([
            nn.SequentialCell(
                nn.Dense(cs[0], cs[4],
                         weight_init=Tensor(self.torch_point_transforms[0][0].weight.detach().cpu().numpy()),
                         bias_init=Tensor(self.torch_point_transforms[0][0].bias.detach().cpu().numpy())),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(),
            ),
            nn.SequentialCell(
                nn.Dense(cs[4], cs[6],
                         weight_init=Tensor(self.torch_point_transforms[1][0].weight.detach().cpu().numpy()),
                         bias_init=Tensor(self.torch_point_transforms[1][0].bias.detach().cpu().numpy())),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(),
            ),
            nn.SequentialCell(
                nn.Dense(cs[6], cs[8],
                         weight_init=Tensor(self.torch_point_transforms[2][0].weight.detach().cpu().numpy()),
                         bias_init=Tensor(self.torch_point_transforms[2][0].bias.detach().cpu().numpy())),
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

    def construct(self, x):

        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        # print(f'before stem: ')
        # print(f'x0.F: {x0.F.shape}')
        # print(f'x0.C: {x0.C.shape}')
        x0 = self.stem(x0)
        # x0.F = self.dense1(x0.F)

        # print(f'after stem: ')
        # print(f'x0.F: {x0.F.shape}')
        # print(f'x0.C: {x0.C.shape}')

        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        # print(f'z0.F: {z0.F}')
        # print(f'z0.F.shape: {z0.F.shape}')
        if (np.isnan(z0.F.asnumpy()).any()):  # Checking for nan values
            print("np.isnan name: z0.F",
                  '\n-------------------------------------------\n',
                  z0.F)  # Prints an index of samples with nan values
            exit()

        x1 = point_to_voxel(x0, z0)
        # print(f'x1.F: {x1.F}')
        # print(f'x1.F.shape: {x1.F.shape}')
        x1 = self.stage1(x1)
        # print(f'x1.2.F: {x1.F}')
        # print(f'x1.2.F.shape: {x1.F.shape}')
        if (np.isnan(x1.F.asnumpy()).any()):  # Checking for nan values
            print("np.isnan name: stage1.F",
                  '\n-------------------------------------------\n',
                  x1.F)  # Prints an index of samples with nan values
            exit()

        x2 = self.stage2(x1)
        # print(f'x2.F: {x2.F}')
        # print(f'x2.F.shape: {x2.F.shape}')
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        # print(f'z1.F: {z1.F}')
        # print(f'z1.F.shape: {z1.F.shape}')
        if (np.isnan(z1.F.asnumpy()).any()):  # Checking for nan values
            print("np.isnan name: z1.F",
                  '\n-------------------------------------------\n',
                  z1.F)  # Prints an index of samples with nan values
            exit()

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
        # print(f'z2.F.shape: {z2.F.shape}')
        if (np.isnan(z2.F.asnumpy()).any()):  # Checking for nan values
            print("np.isnan name: z2.F",
                  '\n-------------------------------------------\n',
                  z2.F)  # Prints an index of samples with nan values
            exit()

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
        # print(f'z3.F.shape: {z3.F.shape}')
        if (np.isnan(z3.F.asnumpy()).any()):  # Checking for nan values
            print("np.isnan name: z3.F",
                  '\n-------------------------------------------\n',
                  z3.F)  # Prints an index of samples with nan values
            exit()

        out = self.classifier(z3.F)

        if (np.isnan(out.asnumpy()).any()):  # Checking for nan values
            print("np.isnan name: out",
                  '\n-------------------------------------------\n',
                  out)  # Prints an index of samples with nan values
            exit()

        return out
