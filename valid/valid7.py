"""
Validate:
Test the gap for convolution like GE.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


in_channel = 3;
width = 8;
out_channel = 6;
kernel_size=3;
batch_size = 2;

a = torch.rand(batch_size,in_channel,width,width)
globalAP=nn.AdaptiveAvgPool2d(1)
gap = globalAP(a)
print(gap.shape)

weight1 = gap[0,:,:,:].expand(out_channel,in_channel,width,width)
weight2 = gap[1,:,:,:].expand(out_channel,in_channel,width,width)
out1 = F.conv2d(a[0,:,:,:].unsqueeze(dim=0),weight1,padding=1)
out2 = F.conv2d(a[1,:,:,:].unsqueeze(dim=0),weight2,padding=1)
out = torch.cat((out1,out2),dim=0)
print(out)



exit(0)








# try remove batch dimension

in_channel = 3;
width = 8;
out_channel = 6;
kernel_size=3;
batch_size = 2;

a = torch.rand(batch_size,in_channel,width,width)
globalAP=nn.AdaptiveAvgPool2d(1)
gap = globalAP(a)
print(gap.shape)



weight1 = gap[0,:,:,:].expand(6,3,3,3)
weight2 = gap[1,:,:,:].expand(6,3,3,3)
out1 = F.conv2d(a[0,:,:,:].unsqueeze(dim=0),weight1,padding=1)
out2 = F.conv2d(a[1,:,:,:].unsqueeze(dim=0),weight2,padding=1)
out = torch.cat((out1,out2),dim=0)
print(out)

# 2
weight = gap.view(1,6,1,1).expand(12,6,3,3)
input = a.view(1,6,8,8)
out = F.conv2d(input,weight,padding=1,groups=2)
out = out.reshape(2,6,8,8)
print(out)


exit(0)


a = torch.rand(2,8,7,7)
conv1 = nn.Conv2d(8,16,3,padding=1,bias=False)
weight1 = conv1.weight
b = conv1(a)

gap = nn.AdaptiveAvgPool2d(1)
weight2 = gap(a)
weight2  = weight2.unsqueeze(dim=1).expand(2,16,8,3,3)
F.conv2d()
print(weight2.shape)
