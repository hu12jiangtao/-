from torch import nn
import torch
from torch.nn import functional as F
# 对于shuffle net v2的基本网络结构应当分为stride=1的和stride=2的两个结构
# 对于stride=1的基本模块来说 输入的特征图首先分为了两个部分x1 x2, x2经过一系列操作后与x1 concat起来后进行通道重排得到这个block的输出
# 对于stride=2的基本模块来说 输入的特征图首先分为了两个部分x1 x2， x1 x2经过相应的操作后在通道上concat起来得到这个block的输出

def channel_split(x, split): # 将通道分为两部分,split应保证为输入x通道数量的一半
    assert x.shape[1] == split * 2
    return torch.split(x,split,dim=1)

def channel_shuffle(x, groups):
    batch, channel, h, w = x.shape
    assert channel % groups == 0
    per_group_num = channel // groups
    x = x.reshape(batch, groups, per_group_num, h, w)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batch, -1, h, w)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(ShuffleUnit, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        if in_channel != out_channel or stride == 2:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel,in_channel,kernel_size=1),
                nn.BatchNorm2d(in_channel),nn.ReLU(inplace=True),
                nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,stride=stride,groups=in_channel),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel,int(out_channel / 2),kernel_size=1),
                nn.BatchNorm2d(int(out_channel / 2)),nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,stride=stride,groups=in_channel),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel,int(out_channel / 2),kernel_size=1),
                nn.BatchNorm2d(int(out_channel / 2)),nn.ReLU(inplace=True)
            )

        else:
            self.shortcut = nn.Sequential()
            in_channel = int(in_channel / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=1),
                nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=stride, groups=in_channel),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, in_channel, kernel_size=1),
                nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True)
            )

    def forward(self,x):
        if self.in_channel == self.out_channel and self.stride == 1:
            shortcut, residual = channel_split(x, int(self.in_channel / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        out = torch.cat([shortcut,residual],dim=1)
        out = channel_shuffle(out,groups=2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, ratio, class_num=100):
        super(ShuffleNetV2, self).__init__()
        if ratio == 1:
            self.out_channels = [116, 232, 464, 1024]
        elif ratio == 0.5:
            self.out_channels = [48, 96, 192, 1024]
        elif ratio == 1.5:
            self.out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            self.out_channels = [244, 488, 976, 2048]
        # 针对于输入层
        self.pre = nn.Sequential(nn.Conv2d(3, 24, kernel_size=3, padding=1), nn.BatchNorm2d(24))
        # 针对于中间的shuffle net v2 模块
        self.stage2 = self.make_block(24, self.out_channels[0], 3) # [batch,116,16,16]
        self.stage3 = self.make_block(self.out_channels[0],self.out_channels[1], 7) # [batch,232,8,8]
        self.stage4 = self.make_block(self.out_channels[1],self.out_channels[2], 3) # [batch,464,4,4]
        self.conv5 = nn.Sequential(nn.Conv2d(self.out_channels[2],self.out_channels[3],kernel_size=1),
                                   nn.BatchNorm2d(self.out_channels[3]),nn.ReLU(inplace=True)) # [batch,1024,4,4]
        # 输出层
        self.fc = nn.Linear(self.out_channels[-1], class_num)

    def make_block(self,in_channel,out_channel,repeat):
        layer = [ShuffleUnit(in_channel,out_channel,2)]
        for _ in range(repeat):
            layer.append(ShuffleUnit(out_channel,out_channel,1))
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.pre(x) # [batch, 24, 32, 32]
        x = self.stage2(x) # [batch, 116, 16, 16]
        x = self.stage3(x) # [batch, 232, 8, 8]
        x = self.stage4(x) # [batch, 464, 4, 4]
        x = self.conv5(x) # [batch, 1024, 4, 4]
        x = F.adaptive_avg_pool2d(x,1) # [batch, 1024, 1, 1]
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    net = ShuffleNetV2(ratio=1).to(device)
    x = torch.randn(size=(1,3,32,32),device=device)
    print(net(x))





