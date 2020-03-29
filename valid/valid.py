"""
Validate:
if a group convolution can be considered as two subconv?
Yes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2,8,7,7)
conv1 = nn.Conv2d(8,16,3,padding=1, groups=2,bias=False)
b = conv1(a)


param = conv1.weight

param1 = param[0:8,:,:,:]
conv1_1 = nn.Conv2d(4,8,3,padding=1,bias=False)
conv1_1.weight = torch.nn.Parameter(param1)

param2 = param[8:16,:,:,:]
conv1_2 = nn.Conv2d(4,8,3,padding=1,bias=False)
conv1_2.weight = torch.nn.Parameter(param2)

b1 = conv1_1(a[:,0:4,:,:])
b2 = conv1_2(a[:,4:8,:,:])
print(b)
print(torch.cat([b1,b2],dim=1))
