# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml

from Models.pytorch.header_fpn import FPN


class RetinaFaceHeader(nn.Module):
    """
    RetinaFace的Header，用于从最终的feature map中提取框、关键点、以及人脸confidence

        Args:
            conv_model：提取特征所用卷积类型
            anchor_scale_list: anchor 的 scale 列表（二维）
            anchor_scale_list: anchor 的 ratio 列表（二维）
            feature_channel_list：每个特征图的通道数量列表（从最大特征图到最小特征图）
            conv_kernel 提取特征所用的卷积核大小
    """
    def __init__(self, conv_model, anchor_scale_list, anchor_ratio_list, feature_channel_list, conv_kernel=1 ):
        super(RetinaFaceHeader, self).__init__()

        # num_anchors = config['NUM_ANCHORS']
        # feature_channels = config['FEATURE_CHANNELS']

        num_anchor_list = []
        for scale, ratio in zip(anchor_scale_list, anchor_ratio_list):
            num_anchor_list.append( len(scale)*len(ratio) )

        self.num_anchor_list = num_anchor_list
        feature_channels = feature_channel_list

        header_class = []
        header_bbox = []
        header_landmark = []

        if conv_kernel == 1:
            kernel = 1
            pad = 0
        else:
            kernel = conv_kernel
            pad = kernel//2

        for chn, num_anchor in zip(feature_channels, num_anchor_list):
            header_class.append(
                conv_model(chn, num_anchor * 2, kernel_size=kernel, stride=1, padding=pad, bias=True)
            )
            header_bbox.append(
                conv_model(chn, num_anchor * 4, kernel_size=kernel, stride=1, padding=pad, bias=True)
            )
            header_landmark.append(
                conv_model(chn, num_anchor * 212, kernel_size=kernel, stride=1, padding=pad, bias=True)
            )

        self.header_class = nn.ModuleList(header_class)
        self.header_bbox = nn.ModuleList(header_bbox)
        self.header_landmark = nn.ModuleList(header_landmark)
        self.fpn = FPN(feature_channels)

    def forward(self, x_list_in):
        num_anchor_list = self.num_anchor_list

        x_list_in = self.fpn(x_list_in)

        out_class = [self.header_class[i](x_list_in[i]) for i in range(len(x_list_in))]
        out_bbox = [self.header_bbox[i](x_list_in[i]) for i in range(len(x_list_in))]
        out_landmark = [self.header_landmark[i](x_list_in[i]) for i in range(len(x_list_in))]

        batch_size = out_class[0].size()[0]

        out_class = [out_class[i].reshape(batch_size, num_anchor_list[i]*2, -1)
                     for i in range(len(out_class))]

        out_class = [out.reshape(batch_size, num_anchor * 2, -1)
                     for out, num_anchor in zip(out_class, num_anchor_list)]
        out_class = [out.transpose(1, 2) for out in out_class]
        out_class = [out.reshape(batch_size, -1, 2) for out in out_class]
        out_class = torch.cat(out_class, dim=1)

        out_bbox = [out.reshape(batch_size, num_anchor * 4, -1)
                    for out, num_anchor in zip(out_bbox, num_anchor_list)]
        out_bbox = [out.transpose(1, 2) for out in out_bbox]
        out_bbox = [out.reshape(batch_size, -1, 4) for out in out_bbox]
        out_bbox = torch.cat(out_bbox, dim=1)

        out_landmark = [out.reshape(batch_size, num_anchor * 212, -1)
                        for out, num_anchor in zip(out_landmark, num_anchor_list)]
        out_landmark = [out.transpose(1, 2) for out in out_landmark]
        out_landmark = [out.reshape(batch_size, -1, 212) for out in out_landmark]
        out_landmark = torch.cat(out_landmark, dim=1)

        return out_class, out_bbox, out_landmark


if __name__ == '__main__':
    config_path = "../../Projects/face_detect_landmark/face_detect_landmark.yaml"
    with open(config_path) as f:
        data = f.read()
    cfg = yaml.load(data, Loader=yaml.FullLoader)
    cfg_retinaface_header = cfg['MODEL']['HEADER_RETINAFACE']
    cfg_anchor = cfg['CORE']['ANCHOR']

    x_in_0 = torch.ones(22, 16, 80, 48)
    x_in_1 = torch.ones(22, 32, 40, 24)
    x_in_2 = torch.ones(22, 64, 20, 12)

    net = RetinaFaceHeader(
        nn.Conv2d,
        cfg_anchor['SCALE_LIST'],
        cfg_anchor['RATIO_LIST'],
        cfg_retinaface_header['FEATURE_CHANNELS'],
        conv_kernel=3
    )

    out_class, out_bbox, out_landmark = net([x_in_0, x_in_1, x_in_2])

    print(out_class.size())
    print(out_bbox.size())
    print(out_landmark.size())

    # x_in_0 = torch.rand(1,32, 320,192)
    # net5 = HRNetStages(cfg_hrnet, 32)
    # out5 = net5([x_in_0])
    # torch.onnx.export(net5, [x_in_0], "/Users/xie/Test/KK_2020/AI_magic_cube/temp/hrstage.onnx", opset_version=11)
    # for out in out5:
    #     print(out.size())



