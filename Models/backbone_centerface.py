import torch.nn as nn
import torch
from Models.pytorch.blocks import MBBlock


# in_channels, out_channels, kernel_size, stride, padding, expansion, bn_momentum, layer_count
config_list = [
    {'in_channels':16, 'out_channels':24, 'kernel_size':3, 'stride':2,
     'padding':1, 'expansion':6, 'bn_momentum':0.01, 'layer_count':2},

    {'in_channels':24, 'out_channels':32, 'kernel_size':3, 'stride':2,
     'padding':1, 'expansion':6, 'bn_momentum':0.01, 'layer_count':3},

    {'in_channels':32, 'out_channels':64, 'kernel_size':3, 'stride':2,
     'padding':1, 'expansion':6, 'bn_momentum':0.01, 'layer_count':4},

    {'in_channels':64, 'out_channels':96, 'kernel_size':3, 'stride':1,
     'padding':1, 'expansion':6, 'bn_momentum':0.01, 'layer_count':3},

    {'in_channels':96, 'out_channels':160, 'kernel_size':3, 'stride':2,
     'padding':1, 'expansion':6, 'bn_momentum':0.01, 'layer_count':3},

    {'in_channels':160, 'out_channels':320, 'kernel_size':3, 'stride':1,
     'padding':1, 'expansion':6, 'bn_momentum':0.01, 'layer_count':1},
]


class MBStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 expansion=1, bn_momentum=0.01, layer_count=1):
        super(MBStage, self).__init__()

        layers = []

        layers.append(
            MBBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    expansion=expansion, bn_momentum=bn_momentum)
        )

        for i in range(1, layer_count):
            layers.append(
                MBBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                        expansion=expansion, bn_momentum=bn_momentum)
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class CenterFace(nn.Module):
    def __init__(self, cfg_list):
        super(CenterFace, self).__init__()

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 1, 1, 0, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        )

        for i in range(len(cfg_list)):
            layers.append(
                MBStage(
                    config_list[i]['in_channels'],
                    config_list[i]['out_channels'],
                    config_list[i]['kernel_size'],
                    config_list[i]['stride'],
                    config_list[i]['padding'],
                    config_list[i]['expansion'],
                    config_list[i]['bn_momentum'],
                    config_list[i]['layer_count'])
            )

        layers.append(
            nn.Sequential(
                nn.Conv2d(320, 24, 1, 1, 0),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True)
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


if __name__ == '__main__':
    x_in = torch.rand(1, 3, 640, 384)
    net = CenterFace(config_list)
    # with open('centerFace.txt', 'w+') as f:
        # print(net, file = f)
    # y_out = net(x_in)
    # print(y_out.size())
    # torch.onnx.export(net, x_in, "test.onnx", verbose=True, input_names=['in'],
    #                   output_names=['out'])
