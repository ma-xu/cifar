"""
Validate:
Dilate Conv
Yes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PadWithin(nn.Module):
    def __init__(self, stride=2):
        super(PadWithin, self).__init__()
        self.stride = stride
        self.w = torch.zeros(2,2)
        self.w[0,0] = 1

    def forward(self, feats):
        out =  F.conv_transpose2d(feats, self.w.expand(feats.size(1), 1, self.stride, self.stride), stride=self.stride, groups=feats.size(1))
        c_in, c_out, k_w, k_h = out.shape
        out = out[:,:,0:k_w-1,0:k_h-1]
        return out
a = torch.rand(16,3,2,2)

# print(a)
# pad = PadWithin()
# print(pad(a))
# exit(0)


in_channel = 3
out_channel = 16
kernel_size = 3
a = torch.rand((3,3,40,40))
conv1 = torch.nn.Conv2d(in_channel,out_channel,kernel_size,bias=False,padding=1)
conv2 = torch.nn.Conv2d(in_channel,out_channel,math.ceil(kernel_size/2),dilation=2,bias=False,padding=1)
convMix = torch.nn.Conv2d(in_channel,out_channel,kernel_size,bias=False,padding=1)
out1 = conv1(a)
out2 = conv2(a)
weight1 = conv1.weight
weight2 = conv2.weight

print("The first out: {}".format(out1+out2))
pad = PadWithin()
weightMix = weight1+pad(weight2)
convMix.weight = nn.Parameter(weightMix)
print("The second out: {}".format(convMix(a)))
exit(0)





# validate dilate for kernel_size=5: True
a = torch.tensor([[[[1.,2., 3.,4.,5.],
          [6., 7., 8., 9., 10.],
          [11., 12., 13., 14., 15.],
          [16.,17.,18.,19.,20.],
          [21.,22.,23.,24.,25.]
        ]]])
conv = torch.nn.Conv2d(1,1,3,dilation=2,bias=False)
conv.weight=nn.Parameter(torch.tensor([[[[1.,  2., 3.],
          [ 4., 5., 6.],
          [ 7., 8., 9.]]]]))
print(conv(a))
exit(0)


conv = torch.nn.Conv2d(1,1,2,dilation=2,bias=False)
conv.weight=nn.Parameter(torch.tensor([[[[1.,  2.],
          [ 3., 4.]]]]))
print(conv.weight.sum())
print(conv(a))



# validate dilate for kernel_size=3: True
a = torch.tensor([[[[1.,2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]]]])

conv = torch.nn.Conv2d(1,1,2,dilation=2,bias=False)
conv.weight=nn.Parameter(torch.tensor([[[[1.,  2.],
          [ 3., 4.]]]]))
print(conv.weight.sum())
print(conv(a))
