from torch import nn
import torch
# 一维卷积中也存在这分组的情况
in_channel = 32
out_channel = 32
net = nn.Conv1d(in_channel,out_channel,kernel_size=1, groups=4,bias=False)
x = torch.randn(size=(1,32,4))
print(sum([i.numel() for i in net.parameters()])) # 32 * 32 / 4 = 256
print(net(x).shape)