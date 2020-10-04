'''ResNet in PyTorch.


Reference:
[1]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['weight_resnet18', 'weight_resnet34', 'weight_resnet50', 'weight_resnet101',
           'weight_resnet152']


class WeightNet(nn.Module):
    def __init__(self, inp, oup, ksize, stride):
        super().__init__()
        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = max(16, inp//16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.reduce = nn.Conv2d(inp, max(16, inp // 16), 1, 1, 0, bias=True)
        self.wn_fc1 = nn.Conv2d(inp_gap, self.M * oup, 1, 1, 0, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = nn.Conv2d(self.M * oup, oup * inp * ksize * ksize, 1, 1, 0, groups=self.G * oup, bias=False)

    def forward(self, x):
        b = x.size()[0]
        x_gap = F.adaptive_avg_pool2d(x,1)
        x_gap = self.reduce(x_gap)
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        if x.shape[0] == 1:  # case of batch size = 1
            x_w = x_w.reshape(self.oup, self.inp, self.ksize, self.ksize)
            x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad)
            return x

        x = x.reshape(1, -1, x.shape[2], x.shape[3])
        x_w = x_w.reshape(-1, self.inp, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad, groups=b)
        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = WeightNet(in_planes, planes,3,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = WeightNet(planes, planes, 3, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                WeightNet(in_planes, self.expansion*planes, 1, stride),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.conv1 = WeightNet(in_planes, planes, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = WeightNet(planes, planes, 3, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3 = WeightNet(planes, self.expansion*planes, 1, 1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                WeightNet(in_planes,self.expansion*planes,1,stride),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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


def weight_resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2],num_classes)

def weight_resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3],num_classes)

def weight_resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3],num_classes)

def weight_resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3],num_classes)

def weight_resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3],num_classes)


def demo():
    net = weight_resnet18(num_classes=100)
    y = net(torch.randn(8,3,32,32))
    print(y.size())

# demo()
