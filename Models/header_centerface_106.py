import torch
import torch.nn as nn


class _conv_3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(_conv_3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class _conv_1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(_conv_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out


class CernterFaceHeader(nn.Module):
    """
    基于centerface的多任务输出

        Args:
            in_channel：最大特征金字塔通道数量
    """
    def __init__(self, in_channel, hm_cls_conv=_conv_3x3, offset_conv=_conv_3x3,
                 scale_conv=_conv_3x3, landmark_conv=None, hm_landmark_conv=None):
        super(CernterFaceHeader, self).__init__()
        assert landmark_conv is not None or hm_landmark_conv is not None, 'landmark can not empty!'
        header = []
        header.append(hm_cls_conv(in_channel, 1))
        header.append(offset_conv(in_channel, 2))
        header.append(scale_conv(in_channel, 2))
        if landmark_conv is not None:
            header.append(landmark_conv(in_channel, 212))
        if hm_landmark_conv is not None:
            header.append(hm_landmark_conv(in_channel, 106))
        self.header = nn.ModuleList(header)

    def forward(self, x):
        header = self.header
        out = [conv(x) for conv in header]
        return out


if __name__ == '__main__':
    x = torch.rand(1, 24, 128, 128)
    header = CernterFaceHeader(24, hm_landmark_conv=_conv_1x1)
    out = header(x)
    for o in out:
        print(o.size())

    torch.onnx.export(header, x, "header.onnx")



