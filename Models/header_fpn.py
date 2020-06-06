import torch.nn as nn
import math
import torch.nn.functional as F
import torch


class FPN(nn.Module):
    """
    多尺度融合模块 bigger = (little->upsample)+bigger
    输入的feature map list[0] 尺度最大
    每一层feature map

        Args:
            in_channels：输入特征图的通道数量
    """
    def __init__(self, channel_list, use_ssh=False):
        super(FPN, self).__init__()
        feature_channels = channel_list
        fpn = []
        fpn.append(None)
        for i in range(0, len(feature_channels)-1):
            if feature_channels[i] == feature_channels[i+1]:
                fpn.append(None)
            else:
                fpn.append(
                    nn.Conv2d(
                        in_channels=feature_channels[i+1],
                        out_channels=feature_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
        self.fpn = nn.ModuleList(fpn)

    def forward(self, x_list_in):
        x_list_w = [ x_in.size()[2] for x_in in x_list_in]
        x_list_h = [ x_in.size()[3] for x_in in x_list_in]
        x_list_scale_w = [ x_list_w[i+1]/x_list_w[i] for i in range(len(x_list_w)-1) ]
        x_list_scale_h = [ x_list_h[i+1]/x_list_h[i] for i in range(len(x_list_h)-1) ]
        x_list_scale_w = [ math.log(w, 2) for w in x_list_scale_w ]
        x_list_scale_h = [ math.log(h, 2) for h in x_list_scale_h ]
        for i in range(len(x_list_scale_w)):
            assert x_list_scale_w[i] - round(x_list_scale_w[i]) == 0, 'feature size are not growing by 2 !'
            assert x_list_scale_h[i] - round(x_list_scale_h[i]) == 0, 'feature size are not growing by 2 !'

        x_list_last = x_list_in[-1]
        for i in range(len(x_list_in)-2, -1, -1):
            while x_list_last.size()[3] < x_list_in[i].size()[3]:
                x_list_last = F.interpolate(
                    x_list_last, size=[x_list_last.shape[2]*2, x_list_last.shape[3]*2], mode='nearest'
                )
            if self.fpn[i+1] is None:
                x_list_in[i] += x_list_last
            else:
                x_list_in[i] = self.fpn[i+1](x_list_last)

            x_list_last = x_list_in[i]

        return x_list_last



if __name__ == '__main__':
    x1 = torch.ones(1, 16, 128, 128)
    x2 = torch.ones(1, 32, 64, 64)
    x3 = torch.ones(1, 64, 32, 32)

    fpn = FPN([16, 32, 64])
    out = fpn([x1, x2, x3])

    print(out[0].size())


