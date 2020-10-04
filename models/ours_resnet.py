'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['our_resnet18', 'our_resnet34', 'our_resnet50', 'our_resnet101',
           'our_resnet152']


class OurLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1,
                 dilation=1, groups=1, bias=True, spatial_size=4, reduction=16):
        super(OurLayer, self).__init__()
        # convolutional parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # parameters for our design

        # define particular operations
        self.resizeLayer = nn.AdaptiveAvgPool2d(4)
        self.GAPLayer = nn.AdaptiveAvgPool2d(1)
        self.spatial_extractor = nn.Linear(spatial_size * spatial_size, kernel_size * kernel_size)
        self.out_channel_extractor = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels)
        )
        self.in_channel_extractor = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        n = x.shape[0]
        x_spatial = self.resizeLayer(x)
        x_spatial = x_spatial.view(n, x_spatial.shape[1], -1)
        x_spatial = x_spatial.mean(dim=1, keepdim=False)
        x_spatial = self.spatial_extractor(x_spatial)
        x_spatial = x_spatial.view(n, self.kernel_size, self.kernel_size)  # [n,w,h]

        x_gap = self.GAPLayer(x).view(n, -1)
        x_channel_out = self.out_channel_extractor(x_gap)
        x_channel_in = self.in_channel_extractor(x_gap)

        # reshape

        x_channel_out = x_channel_out.view(n, self.out_channels, 1, 1, 1)
        x_channel_in = x_channel_in.view(n, 1, self.in_channels, 1, 1)
        x_spatial = x_spatial.view(n, 1, 1, self.kernel_size, self.kernel_size)
        weight = x_channel_out * x_channel_in * x_spatial  # [n, out_channels,in_channels,k,k]

        # run 1
        weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, -1, x.shape[2], x.shape[3])
        y = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=n)
        y = y.view(n, -1, y.shape[2], y.shape[3])

        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = OurLayer(in_planes,planes,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = OurLayer(planes,planes,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = OurLayer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = OurLayer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.conv3 = OurLayer(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                OurLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def our_resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def our_resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def our_resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def our_resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def our_resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def demo():
    net = our_resnet18(num_classes=100)
    y = net(torch.randn(10, 3, 32, 32))
    print(y.size())


# demo()
