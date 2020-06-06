import torch.nn as nn
import torch.nn.functional as F
import torch


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d ,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=bias)

    def forward(self ,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class HSwish(nn.Module):
    def __init__(self):
        super(HSwish ,self).__init__()

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class MixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(MixedConv2d, self).__init__()

        self.in_channels = in_channels
        self.n_chunks = self.calculate_group()
        self.split_out_channels = self.split_layer()

        self.layers = nn.ModuleList()
        for i in range(self.n_chunks):
            sub_kernel_size = 2 * i + 3
            padding = (sub_kernel_size - 1) // 2
            self.layers.append(
                nn.Conv2d(in_channels = self.split_out_channels[i],
                          out_channels = self.split_out_channels[i],
                          kernel_size = sub_kernel_size,
                          padding = padding,
                          groups = self.split_out_channels[i],
                          stride=stride,
                          bias=bias))
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   dilation=1,
                                   groups=1,
                                   bias=bias)

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], 1)
        out = self.pointwise(out)
        return out

    def calculate_group(self):
        if self.in_channels < 8:
            g_i = 1
        elif self.in_channels < 16:
            g_i = 2
        elif self.in_channels < 32:
            g_i = 3
        else:
            g_i = 4
        return g_i

    def split_layer(self):
        split = []
        rest = self.in_channels
        for i in range(self.n_chunks-1):
            add = rest//2
            rest -= add
            split.append(add)
        split.append(rest)

        split[self.n_chunks - 1] += self.in_channels - sum(split)
        return split


class Mobile3Basic(nn.Module):

    def __init__(self, conv_model, in_channels, out_channels, stride=1, expansion=1, bn_momentum=0.01):
        super(Mobile3Basic, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels*expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels*expansion)
        self.nolinear1 = HSwish()
        self.conv2 = nn.Conv2d(in_channels*expansion, in_channels*expansion, kernel_size=3, stride=stride,
                               padding=1, groups=in_channels*expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels*expansion)
        self.nolinear2 = HSwish()
        self.conv3 = nn.Conv2d(in_channels*expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nolinear1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nolinear2(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out


class ResidualBasic(nn.Module):
    def __init__(self, conv_model, in_channels, out_channels, stride=1, expansion=1, bn_momentum=0.01):
        super(ResidualBasic, self).__init__()

        self.conv1 = conv_model(in_channels, out_channels, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv_model(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, conv_model, in_channels, out_channels, stride=1, compress=4, bn_momentum=0.01):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels//compress, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//compress, momentum=bn_momentum)
        self.conv2 = conv_model(out_channels//compress, out_channels//compress,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//compress, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(out_channels//compress, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expansion=1, bn_momentum=0.01):
        super(MBBlock, self).__init__()

        if expansion == 1:
            self.conv1 = None
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels*expansion, kernel_size=1, stride=1,
                                padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels*expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels*expansion, in_channels*expansion, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=in_channels*expansion ,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels*expansion, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(in_channels*expansion, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        if in_channels==out_channels and stride==1:
            self.add_resdual = True
        else:
            self.add_resdual = False

    def forward(self, x):
        residual = x

        # print('-'*80)

        if self.conv1 is not None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = x

        # print(out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # print(out.size())

        out = self.conv3(out)
        out = self.bn3(out)

        # print(out.size())

        if self.add_resdual:
            out = out + residual

        return out


if __name__ == '__main__':
    model = Bottleneck(MixedConv2d, 33, 68)
    print(model)

