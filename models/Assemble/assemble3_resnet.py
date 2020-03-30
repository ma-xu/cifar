'''ResNet in PyTorch.
Assemble: conventional + group
Only replace 3x3 Conv
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ass3_resnet18', 'ass3_resnet34', 'ass3_resnet50', 'ass3_resnet101',
           'ass3_resnet152']


class AssConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=[1],bias=False):
        super(AssConv, self).__init__()
        self.convList = nn.ModuleList()
        self.groups = groups
        for i in groups:
            self.convList.insert(i,
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups=i,bias=bias))

    def forward(self, input):
        out = 0
        for i, conv in enumerate(self.convList):
            out = out+ conv(input)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = AssConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,groups=[1,4,8])
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AssConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,groups=[1,4,8])
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AssConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,groups=[1,4,8])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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


def ass3_resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2],num_classes)

def ass3_resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3],num_classes)

def ass3_resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3],num_classes)

def ass3_resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3],num_classes)

def ass3_resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3],num_classes)


def demo():
    net = ass3_resnet18(num_classes=100)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

demo()

def demoAssConv():
    data = torch.rand(2,8,8,8)
    ass = AssConv(8,16,3,1,1,1,[1,2,4])
    out = ass(data)
    print(out.shape)

# demoAssConv()
