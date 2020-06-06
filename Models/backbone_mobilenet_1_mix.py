import torch
import torch.nn as nn
from Models.pytorch.blocks import MixedConv2d


config_list = [
    {'in_channels':8, 'out_channels':16, 'stride':1, 'bn_momentum':0.01, 'layer_count':1},
    {'in_channels':16, 'out_channels':32, 'stride':2, 'bn_momentum':0.01, 'layer_count':2},
    {'in_channels':32, 'out_channels':64, 'stride':2, 'bn_momentum':0.01, 'layer_count':2},
    {'in_channels':64, 'out_channels':128, 'stride':2, 'bn_momentum':0.01, 'layer_count':6},
    {'in_channels':128, 'out_channels':256, 'stride':2, 'bn_momentum':0.01, 'layer_count':2},
]


class MBBlock1_Mix(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn_momentum=0.01):
        super(MBBlock1_Mix, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MixedConv2d(in_channels, in_channels,
                               stride=stride ,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

    def forward(self, x):

        # print('-'*80)

        # print(out.size())

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        # print(out.size())

        out = self.conv3(out)
        out = self.bn3(out)

        # print(out.size())

        return out


class MBStage_Mix(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, layer_count=1):
        super(MBStage_Mix, self).__init__()

        layers = []

        layers.append(
            MBBlock1_Mix(in_channels, out_channels, stride=stride)
        )

        for i in range(1, layer_count):
            layers.append(
                MBBlock1_Mix(out_channels, out_channels, stride=1 )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class MobileNet1_Mix(nn.Module):
    def __init__(self, cfg_list):
        super(MobileNet1_Mix, self).__init__()

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, 8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            )
        )

        for i in range(len(cfg_list)):
            layers.append(
                MBStage_Mix(
                    in_channels = config_list[i]['in_channels'],
                    out_channels = config_list[i]['out_channels'],
                    stride = config_list[i]['stride'],
                    layer_count = config_list[i]['layer_count'])
            )

        layers.append(
            nn.Sequential(
                nn.Conv2d(config_list[-1]['out_channels'], 64, 1, 1, 0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


if __name__ == '__main__':
    x_in = torch.rand(1, 3, 640, 384)
    net = MobileNet1_Mix(config_list)
    y_out = net(x_in)
    print(y_out.size())
    torch.onnx.export(net, x_in, "test.onnx", verbose=True, input_names=['in'],
                      output_names=['out'])


