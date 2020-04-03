"""
Validate:
Test the coefficients for conv weights
Yes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2,8,7,7)
conv1 = nn.Conv2d(8,16,3,padding=1,bias=False)
weight = conv1.weight
b = conv1(a)

### its OK for real-value coefficient.
# coff = torch.rand(1)
# weight1 = coff*weight
# conv1.weight = nn.Parameter(weight1)
# b1 = conv1(a)
# print(b*coff)
# print(b1)


### It works for output channel ...
coff = torch.rand(16)
weight1 = coff.view(16,1,1,1)*weight
conv1.weight = nn.Parameter(weight1)
b1 = conv1(a)
print(b1)
print(b*coff.view(1,16,1,1))



