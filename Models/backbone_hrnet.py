# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml

from Models.pytorch.blocks import Mobile3Basic, ResidualBasic, Bottleneck
from Models.pytorch.blocks import SeparableConv2d, MixedConv2d


BN_MOMENTUM = 0.01



class HRNetBranch(nn.Module):
    """
    构建一个HRNet的branch

        Args:
            block：block的类型，可以是 residual、revert residual、等等
            block_count：block的数量
            conv_model：block使用的卷积类型：Conv2d、Seperate Conv、MixConv
            in_channels：输入通道数目
            inside_channels：内部block通道数目
            out_channels：输出通道数目
    """
    def __init__(self, block, block_count, conv_model, in_channels, inside_channels, out_channels, stride=1):
        super(HRNetBranch, self).__init__()

        layers = []

        if block_count == 1:
            layers.append(block(conv_model, in_channels, out_channels, stride))
        else:
            layers.append(block(conv_model, in_channels, inside_channels, stride))
            for i in range(1, block_count - 1):
                layers.append(block(conv_model, inside_channels, inside_channels, stride))
            layers.append(block(conv_model, inside_channels, out_channels, stride))

        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        x = self.branch(x)
        return x


class HRNetTransition(nn.Module):
    """
    承接之前的分支，融合，并根据下一级的分支数量要求构造下一级分支

        Args:
            conv_model：block的类型，可以是 residual、revert residual、等等
            channel_list_pre：上一级HRNet的所有分支通道列表
            channel_list_out：下一级HRNet的所有分支通道列表
    """
    def __init__(self, conv_model, channel_list_pre, channel_list_out):
        super(HRNetTransition, self).__init__()

        num_branches_pre = len(channel_list_pre)
        num_branches_out = len(channel_list_out)
        self.num_branches_pre = num_branches_pre
        self.num_branches_out = num_branches_out

        transition_layers = []
        for i in range(num_branches_out):
            if i < num_branches_pre:  # 前后相同层级的 直线传播 通道不同用卷积变换一下

                if channel_list_out[i] != channel_list_pre[i]:
                    transition_layers.append(nn.Sequential(
                        conv_model(channel_list_pre[i],
                                   channel_list_out[i],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False),
                        nn.BatchNorm2d(
                            channel_list_out[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:  # 后面没有的，用前面的最下层，通过stride=2卷积 循环扩展
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = channel_list_pre[-1]
                    if num_branches_pre+j == i:  # 扩展时只有最后一层用out的卷积数
                        out_channels = channel_list_out[i]
                    else:
                        out_channels = in_channels
                    conv3x3s.append(nn.Sequential(
                        conv_model(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        self.transition = nn.ModuleList(transition_layers)

    def forward(self, x_list_in):
        x_list_out = []
        num_branches_pre = self.num_branches_pre
        num_branches_out = self.num_branches_out
        for i in range(num_branches_out):
            if i < num_branches_pre:  # 若之前已经有的层，直接传播
                if self.transition[i] is not None:
                    x_list_out.append(self.transition[i](x_list_in[i]))
                else:
                    x_list_out.append(x_list_in[i])
            else: # 之前没有的层，用之前的最下一层来创建
                x_list_out.append(self.transition[i](x_list_in[-1]))

        return x_list_out


class HRNetFuseBranch(nn.Module):
    """
    将之前的所有通道，融合到一个指定的通道上：向上融合——调通道+上采样 向下融合——步长为2的卷积
    即：输入 —— 所有通道， 输出 —— 指定通道

        Args:
            conv_model：block的类型，可以是 residual、revert residual、等等
            channel_list：前一段HRNet的所有分支通道列表
            out_index：融合到第几个通道，最上为高分通道，每向下一层，分辨率减半
    """
    def __init__(self, conv_model, channel_list, out_index):
        super(HRNetFuseBranch, self).__init__()
        num_branches = len(channel_list)
        self.num_branches = num_branches
        self.out_index = out_index
        assert out_index < num_branches, 'index index out of range !'

        transition_layers = []

        for i in range(num_branches):
            if i == out_index:
                transition_layers.append(None)
            elif i < out_index:
                j = i
                conv3x3s = []
                in_channels = channel_list[j]
                while j != out_index:
                    j += 1
                    if j == out_index:
                        conv3x3s.append(
                            nn.Sequential(
                                conv_model(in_channels,
                                           channel_list[out_index],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           bias=False),
                                nn.BatchNorm2d(channel_list[out_index], momentum=BN_MOMENTUM))
                        )
                    else:
                        conv3x3s.append(
                            nn.Sequential(
                                conv_model(in_channels,
                                           in_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           bias=False),
                                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True))
                        )
                transition_layers.append(nn.Sequential(*conv3x3s))
            else:
                conv3x3s = []
                in_channels = channel_list[i]
                out_channels = channel_list[out_index]
                conv3x3s.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
                )
                transition_layers.append(nn.Sequential(*conv3x3s))
        self.transition_layers = nn.ModuleList(transition_layers)

    def forward(self, x_list_in):
        x_out = x_list_in[self.out_index]

        for i in range(self.num_branches):
            if i < self.out_index:
                x_out += self.transition_layers[i](x_list_in[i])
            if i > self.out_index:

                j = i
                x_temp = x_list_in[i]
                while j > self.out_index:
                    x_temp = F.interpolate(x_temp,
                                          size=[x_temp.shape[2]*2, x_temp.shape[3]*2],
                                          mode='nearest')
                    j -= 1
                x_out += self.transition_layers[i](x_temp)

        return x_out


class HRNetFuse(nn.Module):
    """
    融合所有通道
    即：输入 —— 所有通道， 输出 —— 所有通道

        Args:
            conv_model：block的类型，可以是 residual、revert residual、等等
            channel_list：前一段HRNet的所有分支通道列表
    """
    def __init__(self, conv_model, channel_list):
        super(HRNetFuse, self).__init__()

        fuse_layers = []
        num_branches = len(channel_list)
        self.num_branches = num_branches

        for i in range(num_branches):
            fuse_layers.append(HRNetFuseBranch(conv_model, channel_list, i))

        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x_list_in):
        x_list_out = []
        for i in range(self.num_branches):
            x_list_out.append(self.fuse_layers[i](x_list_in))

        return x_list_out


class HRNetModule(nn.Module):
    """
    HRNet的Module：一个HRNet的Module包含多个branch，每个branch包含多个block，
    在module的最后包含一个fuse

        Args:
            config：Module的配置
    """
    def __init__(self, config):
        super(HRNetModule, self).__init__()
        conv_model_dict = {
            'CONV_2D': nn.Conv2d,
            'CONV_DW': SeparableConv2d,
            'CONV_MIX': MixedConv2d
        }

        block_dict = {
            'MOBILE': Mobile3Basic,
            'RESIDUAL': ResidualBasic,
            'BOTTLENECK': Bottleneck
        }
        self.num_branches = len(config['BRANCHES_NUM_BLOCKS'])

        inside_channel = config['BRANCHES_CHANNELS'][0]

        branches = []
        for i in range(self.num_branches):
            branches.append(
                HRNetBranch(
                    block_dict[config['BLOCK']],
                    config['BRANCHES_NUM_BLOCKS'][i],
                    conv_model_dict[config['CONV_MODEL']],
                    config['BRANCHES_CHANNELS'][i],
                    inside_channel,
                    config['BRANCHES_CHANNELS'][i]
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = HRNetFuse(
            conv_model_dict[config['CONV_MODEL']],
            config['BRANCHES_CHANNELS']
        )

    def forward(self, x_in_list):
        x_out_list = []
        for i in range(self.num_branches):
            x_out_list.append(self.branches[i](x_in_list[i]))
        x_out_list = self.fuse(x_out_list)
        return x_out_list


class HRNetStages(nn.Module):
    """
    HRNet的Stage：一个Stage包含多个Module

        Args:
            config：Module的配置
            in_channels：输入特征图的通道数量
    """
    def __init__(self, config, in_channels):
        super(HRNetStages, self).__init__()

        conv_model_dict = {
            'CONV_2D': nn.Conv2d,
            'CONV_DW': SeparableConv2d,
            'CONV_MIX': MixedConv2d
        }

        config_structs = list(config['STRUCT'].values())
        stages=[]
        transition_pre = [in_channels, ]

        for config_stage in config_structs:
            transition_out = config_stage['BRANCHES_CHANNELS']
            stages.append(
                HRNetTransition(conv_model_dict[config_stage['CONV_MODEL']],
                                transition_pre,
                                transition_out)
            )

            for i in range(config_stage['NUM_MODULES']):
                stages.append(
                    HRNetModule(config_stage)
                )
            transition_pre = transition_out

        self.stages = nn.Sequential(*stages)

    def forward(self, x_list_in):
        x_list_out = self.stages(x_list_in)
        return x_list_out


class HRNetFront(nn.Module):
    def __init__(self):
        super(HRNetFront, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.layer1 = Bottleneck(nn.Conv2d, 64, 256)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x





if __name__ == '__main__':
    config_path = "../../Projects/face_detect_landmark/face_detect_landmark.yaml"
    with open(config_path) as f:
        data = f.read()
    cfg = yaml.load(data, Loader=yaml.FullLoader)
    cfg_hrnet = cfg['MODEL']['BACKBONE_HRNET']
    # cfg_struct = list(cfg['MODEL']['BACKBONE_HRNET']['STRUCT'].values())

    x_in_0 = torch.rand(16,3, 48, 80)
    net5 = HRNetStages(cfg_hrnet, 3)
    out5 = net5([x_in_0])
    # torch.onnx.export(net5, [x_in_0], "/Users/xie/Test/KK_2020/AI_magic_cube/temp/hrstage.onnx", opset_version=11)
    for out in out5:
        print(out.size())





