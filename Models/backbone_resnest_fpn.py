'''
@Author: Yawen
@Date: 2020-05-08 02:53:25
@LastEditTime: 2020-05-29 06:51:47
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /AI_magic_cube/Models/pytorch/backbone_resnest2.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU

from header_fpn import FPN

config_list = [
    {'radix':2, 'cardinality':2},
    {'layer_count':3, 'in_channels':64, 'out_channels':128},
    {'layer_count':4, 'in_channels':128, 'out_channels':256},
    {'layer_count':6, 'in_channels':256, 'out_channels':512},
    # {'layer_count':3, 'in_channels':512, 'out_channels':1024},
]

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class SplAtConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(1, 1),
                 dilation=(1, 1), cardinality=1, radix=2, bias=True, 
                 reduction_factor=4, norm_layer=None):
        super(SplAtConv2d, self).__init__()
        
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = cardinality
        self.channels = out_channels
        
        self.conv = Conv2d(in_channels, out_channels*radix, kernel_size, stride, padding, 
                            groups=cardinality*radix, bias=bias)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(out_channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(out_channels, inter_channels, 1, groups=cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, out_channels*radix, 1, groups=cardinality)
        self.rsoftmax = rSoftMax(radix, cardinality)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, radix=1, cardinality=1, 
                    bottleneck_width=64, norm_layer=None, down=False, is_first=False, channel_list=[]):
        super(Bottleneck, self).__init__()

        self.channel_list = channel_list
        group_width = int(out_channels * (bottleneck_width / 64.)) * cardinality
        self.radix = radix
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, cardinality=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if is_first:
            kernel_size = 2
            stride = 2
        else:
            kernel_size = 1
            stride = 1
        if down:
            down_layers = []
            down_layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                                            ceil_mode=True, count_include_pad=False))
            down_layers.append(nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, stride=1, bias=False))
            down_layers.append(norm_layer(out_channels))                                     
            self.downsample = nn.Sequential(*down_layers)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print('out size : \n', out.size())
        # print('residual size 02: \n', residual.size())

        out += residual
        out = self.relu(out)
        
        return out

class ResNeSt(nn.Module):
    def __init__(self, config_list):
        super(ResNeSt, self).__init__()

        stem_width = 32
        num_classes = 2
        
        self.avd = False
        self.avd_first = False
        self.channel_list = []
        self.norm_layer=nn.BatchNorm2d
        self.radix = config_list[0]['radix']
        self.cardinality = config_list[0]['cardinality']
        self.in_channels = config_list[1]['in_channels']

        # self.channel_list.append(config_list[1]['in_channels'])
        # ===================== model front ======================== 
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                self.norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                self.norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ===================== model front ======================== 


        # =================== config controller ==================== 
        layers = []
        for i in range(1, len(config_list)):
            layers.append(self._make_layer(config_list[i]))   
        self.layers = nn.Sequential(*layers)
        # =================== config controller ==================== 


        # ===================== model header ======================= 
        self.avgpool = GlobalAvgPool2d()
        self.sigmoid = nn.Sigmoid()
        layers_out = config_list[-1]['out_channels']
        self.fc = nn.Linear(layers_out, num_classes)
        # ===================== model header ======================= 

        # ================= init model weights =====================
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # ================= init model weights =====================

    def _make_layer(self, config):
        layers = []
        layer_count = config['layer_count']
        in_channels = config['in_channels']
        out_channels = config['out_channels']

        self.channel_list.append(out_channels)

        # layer中第一个为下采样
        layers.append(Bottleneck(in_channels, out_channels, stride=2, 
                            radix=self.radix, cardinality=self.cardinality,
                            norm_layer=self.norm_layer, down=True, is_first=True))
        # 后续layer中的输入与第一个不同
        # in_channels = in_channels * 4
        for i in range(1,layer_count):
            layers.append(Bottleneck(out_channels, out_channels, stride=1, 
                                radix=self.radix, cardinality=self.cardinality,
                                norm_layer=self.norm_layer))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_ls_1 = self.layers[0](x)
        x_ls_2 = self.layers[1](x_ls_1)
        x_ls_3 = self.layers[2](x_ls_2)
        x_list.append(x_ls_1)
        x_list.append(x_ls_2)
        x_list.append(x_ls_3)

        x = self.layers(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x, x_list

if __name__ == '__main__':
    x = torch.rand(1,3,640,384)
    model = ResNeSt(config_list)
    with open('temp.txt', 'w+') as f:
        print(model, file = f)
    model.eval()
    channel_list = model.channel_list
    y, y_list = model(x)
    print(channel_list)
    print(y_list[0].size())

    # fpn = FPN(channel_list)
    # c = fpn(y_list)
    # print('c size is : ',c.size())
    # print(y)
    # print(y.size())
