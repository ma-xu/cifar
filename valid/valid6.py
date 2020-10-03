"""
Validate:
Test the gap for convolution.

X doesnt work
Yes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2,8,7,7)
conv1 = nn.Conv2d(8,16,3,padding=0,bias=False)
weight1 = conv1.weight
b = conv1(a)

gap = nn.AdaptiveAvgPool2d(1)
a_gap = gap(a)
print(a_gap.shape)
conv2 = nn.Conv2d(8,16,1,bias=False)
weight2 = conv2.weight
c = conv2(a_gap)
print(c.shape)
print("weight 1.shape: {}".format(weight1.shape))
print("weight 2.shape: {}".format(weight2.shape))
OUT1 = b+c


conv3 = nn.Conv2d(8,16,3,padding=0,bias=False)
conv3.weight = nn.Parameter(weight1+weight2)
OUT2 = conv3(a)


print(OUT1)
print(OUT2)
