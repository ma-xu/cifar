"""
Validate:
Can be combine a group-conv and a traditional conv?
Yes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


in_channel = 8
out_channel = 16
group = 4
a = torch.rand(2,in_channel,7,7)
conv1 = nn.Conv2d(in_channel,out_channel,3,padding=1, groups=group,bias=False)
param1 = conv1.weight
print(conv1(a).shape)

param_full = torch.zeros((out_channel,in_channel,3,3))
param_list = torch.chunk(param1,group,dim=0)
in_step = in_channel//group
out_step = out_channel//group
for i in range(0,group):
    param_full[i*out_step:(i+1)*out_step,i*in_step:(i+1)*in_step,:,:] = param_list[i]


conv2 = nn.Conv2d(in_channel,out_channel,3,padding=1, bias=False)
conv2.weight = nn.Parameter(param_full)
r1 = conv1(a)
r2 = conv1(a)
print(r1)
print(r2)
