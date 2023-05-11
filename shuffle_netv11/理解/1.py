from torch import nn
import torch

net1 = nn.Conv2d(in_channels=12,out_channels=6,kernel_size=3,groups=3,bias=False)
net2 = nn.Conv2d(in_channels=12,out_channels=6,kernel_size=3,bias=False)
print(sum([param.numel() for param in net1.parameters()]))
print(sum([param.numel() for param in net2.parameters()]))