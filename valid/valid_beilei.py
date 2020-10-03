"""
Validate:
Test the gap for convolution like GE.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([[0.0,0.1,0.2,0.3,2.1],
                  [0.4,0.5,0.6,0.7,2.0],
                  [0.8,0.9,1.1,1.5,2.2],
                  [1.2,1.3,1.4,1.0,2.3],
                  [2.4,2.5,2.6,2.7,3.0],
                  ])
x = x.view(1,1,5,5)
print(x.shape)
weight = torch.tensor([[0.2,0.4],[-1,0.3]])
weight = weight.view(1,1,2,2)
print(weight.shape)
bias = torch.tensor([0.1])
avg = nn.AvgPool2d(2,2)
out  = torch.conv2d(x,weight,bias=bias)
out = avg(out)
print(out)


