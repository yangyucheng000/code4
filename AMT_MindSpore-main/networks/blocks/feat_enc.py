# import torch
# import torch.nn as nn
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops as ops
import math
# class BottleneckBlock(nn.Module):
class BottleneckBlock(nn.Cell):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        # self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        # self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        # self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0, pad_mode='pad', has_bias=True)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride, pad_mode='pad', has_bias=True)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0, pad_mode='pad', has_bias=True)
        self.relu = nn.ReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            # self.norm1 = nn.Sequential()
            # self.norm2 = nn.Sequential()
            # self.norm3 = nn.Sequential()
            # if not stride == 1:
            #     self.norm4 = nn.Sequential()
            self.norm1 = nn.SequentialCell()
            self.norm2 = nn.SequentialCell()
            self.norm3 = nn.SequentialCell()
            if not stride == 1:
                self.norm4 = nn.SequentialCell()

        if stride == 1:
            self.downsample = None
        
        else:    
            # self.downsample = nn.SequentialCell(
            #     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)
            self.downsample = nn.SequentialCell(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, has_bias=True, pad_mode='valid'), self.norm4)


    # def forward(self, x):
    def construct(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


# class ResidualBlock(nn.Module):
class ResidualBlock(nn.Cell):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, pad_mode='pad', has_bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu = nn.ReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            # self.norm1 = nn.Sequential()
            # self.norm2 = nn.Sequential()
            # if not stride == 1:
            #     self.norm3 = nn.Sequential()
            self.norm1 = nn.SequentialCell()
            self.norm2 = nn.SequentialCell()
            if not stride == 1:
                self.norm3 = nn.SequentialCell()

        if stride == 1:
            self.downsample = None
        
        else:    
            # self.downsample = nn.Sequential(
            #     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
             self.downsample = nn.SequentialCell(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, pad_mode='pad', has_bias=True), self.norm3)


    # def forward(self, x):
    def construct(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


# class SmallEncoder(nn.Module):
class SmallEncoder(nn.Cell):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            # self.norm1 = nn.Sequential()
            self.norm1 = nn.SequentialCell()

        # self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
            
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=True)
        self.relu1 = nn.ReLU()

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        # self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1, pad_mode='valid', has_bias=True)
# Reference: https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/sample_code.html?highlight=kaiming_normal_
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if cell.gamma is not None:
                    cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                if cell.beta is not None:
                    cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            # elif isinstance(cell, (nn.Dense)):
            #     if cell.weight is not None:
            #         cell.weight.set_data(ms.common.initializer.initializer(
            #             ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
            #             cell.weight.shape, cell.weight.dtype))
            #     if cell.bias is not None:
            #         cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        # return nn.Sequential(*layers)
        return nn.SequentialCell(*layers)


    # def forward(self, x):
    def construct(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            # x = torch.cat(x, dim=0)
            x = ops.cat(x, axis=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            # x = torch.split(x, [batch_dim, batch_dim], dim=0)
            x = ops.split(x, [batch_dim, batch_dim], axis=0)

        return x

# class BasicEncoder(nn.Module):
class BasicEncoder(nn.Cell):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            # self.norm1 = nn.Sequential()
            self.norm1 = nn.SequentialCell()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=True)
        self.relu1 = nn.ReLU() 

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(72, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1, pad_mode='valid', has_bias=True)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # for m in self.modules():
        for m in self.cells():
            # if isinstance(m, nn.Conv2d):
            #     nn.init._normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            #     if m.weight is not None:
            #         nn.init.constant_(m.weight, 1)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)

            for _, cell in self.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                        cell.weight.shape, cell.weight.dtype))
                    
                elif isinstance(cell, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                    cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                    cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
                    
                elif isinstance(cell, (nn.Dense)):
                    if cell.weight is not None:
                        cell.weight.set_data(ms.common.initializer.initializer(
                            ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                            cell.weight.shape, cell.weight.dtype))
                    if cell.bias is not None:
                        cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        # return nn.Sequential(*layers)
        return nn.SequentialCell(*layers)


    # def forward(self, x):
    def construct(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            # x = torch.cat(x, dim=0)
            x = ops.cat(x, axis=0)


        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            # x = torch.split(x, [batch_dim, batch_dim], dim=0)
            x = ops.split(x, [batch_dim, batch_dim], axis=0)
        return x

# class LargeEncoder(nn.Module):
class LargeEncoder(nn.Cell):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(LargeEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
            
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=True)
        self.relu1 = nn.ReLU()

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(112, stride=2)
        self.layer3 = self._make_layer(160, stride=2)
        self.layer3_2 = self._make_layer(160, stride=1)

        # output convolution
        # self.conv2 = nn.Conv2d(self.in_planes, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_planes, output_dim, kernel_size=1, pad_mode='valid', has_bias=True)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # for m in self.modules():
        for m in self.cells():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            #     if m.weight is not None:
            #         nn.init.constant_(m.weight, 1)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)

            for _, cell in self.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                        cell.weight.shape, cell.weight.dtype))
                elif isinstance(cell, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                    if cell.gamma is not None:
                        cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                    if cell.beta is not None: 
                        cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
                    
                # elif isinstance(cell, (nn.Dense)):
                #     if cell.weight is not None:
                #         cell.weight.set_data(ms.common.initializer.initializer(
                #             ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                #             cell.weight.shape, cell.weight.dtype))
                #     if cell.bias is not None:
                #         cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        # return nn.Sequential(*layers)
        return nn.SequentialCell(*layers)



    # def forward(self, x):
    def construct(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            # x = torch.cat(x, dim=0)
            x = ops.cat(x, axis=0)
            

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_2(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            # x = torch.split(x, [batch_dim, batch_dim], dim=0)
            x = ops.split(x, [batch_dim, batch_dim], axis=0)

        return x
