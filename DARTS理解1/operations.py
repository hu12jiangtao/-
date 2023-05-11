# 对于每个连接的所有的搜索架构的定义
import torch
from torch import nn

OPS = {
    'none':lambda C, stride, affine: Zero(stride),  # 返回一个全0的矩阵,如果stride=2时宽高减半同样也是返回全0的矩阵（说明之间不存在连接）
    'avg_pool_3x3':lambda C, stride, affine: nn.AvgPool2d(kernel_size=3,padding=1,stride=stride,count_include_pad=False), # 平均池化
    'max_pool_3x3': lambda C, stride, affine:nn.MaxPool2d(kernel_size=3,padding=1,stride=stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride==1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
      }

class Zero(nn.Module):
    def __init__(self,stride):
        super(Zero, self).__init__()
        self.stride = stride
    def forward(self,x):
        if self.stride == 1:
            return x.mul(0)
        else:
            return x[:,:,::self.stride,::self.stride].mul(0)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x

class FactorizedReduce(nn.Module):
    def __init__(self,C_in,C_out,affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out//2,kernel_size=1,stride=2,padding=0,bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out//2,kernel_size=1,padding=0,stride=2,bias=False)
        self.bn = nn.BatchNorm2d(C_out,affine=affine)

    def forward(self,x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x),self.conv2(x[:,:,1:,1:])],dim=1) # 这一步是保证当输入的宽高为奇数的情况下，输入和输出的宽高保证不变
        return self.bn(out)

class SepConv(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False),
                                nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,stride=stride,padding=padding,
                                          groups=in_channel,bias=False),
                                nn.Conv2d(in_channel,in_channel,kernel_size=1,bias=False),
                                nn.BatchNorm2d(in_channel, affine=affine),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(in_channel,in_channel,kernel_size,stride=1,padding=padding,
                                          groups=in_channel,bias=False),
                                nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False),
                                nn.BatchNorm2d(out_channel,affine=affine))
    def forward(self,x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False),
                                nn.Conv2d(in_channel,in_channel,kernel_size,stride,padding,dilation=dilation,
                                          groups=in_channel,bias=False),
                                nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False),
                                nn.BatchNorm2d(out_channel,affine=affine))
    def forward(self,x):
        return self.op(x)


class ReLUConvBN(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding,affine=False):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(),nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding,bias=False),
                                nn.BatchNorm2d(out_channel,affine=affine))

    def forward(self,x):
        return self.op(x)




