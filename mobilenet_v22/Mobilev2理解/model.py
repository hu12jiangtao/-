# 此时的alpha系数并不控制最后一层的通道数
from torch import nn
import torch
from torch.nn import functional as F

class LinearBottleNeck(nn.Module):
    def __init__(self,in_channel,out_channel,stride,t):
        super(LinearBottleNeck, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * t, kernel_size=1),
            nn.BatchNorm2d(in_channel * t), nn.ReLU6(inplace=True),

            nn.Conv2d(in_channel * t, in_channel * t, kernel_size=3, padding=1, stride=stride, groups=in_channel * t),
            nn.BatchNorm2d(in_channel * t), nn.ReLU6(inplace=True),

            nn.Conv2d(in_channel * t, out_channel, kernel_size=1), # 降维的时候需要使用线性激活(即没有非线性激活函数)
            nn.BatchNorm2d(out_channel),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

    def forward(self,x):
        out = self.net1(x)
        if self.in_channel == self.out_channel and self.stride == 1:
            out += x
        return out

class MobileNetV2(nn.Module):  # 由于输入的较小，因此对原文中的部分结构进行修改
    def __init__(self,num_class=100):
        super(MobileNetV2, self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(3,32,kernel_size=1,padding=1),
                                 nn.BatchNorm2d(32),nn.ReLU6(inplace=True))  # 此处于论文保证相同
        self.stage1 = self._make_stage(repeat=1, in_channels=32, out_channels=16, stride=1, t=1)
        self.stage2 = self._make_stage(repeat=2, in_channels=16, out_channels=24, stride=2, t=6)
        self.stage3 = self._make_stage(repeat=3, in_channels=24, out_channels=32, stride=2, t=6)
        self.stage4 = self._make_stage(repeat=4, in_channels=32, out_channels=64, stride=2, t=6)
        self.stage5 = self._make_stage(repeat=3, in_channels=64, out_channels=96, stride=1, t=6)
        self.stage6 = self._make_stage(repeat=3, in_channels=96, out_channels=160, stride=1, t=6) # 原文中是stride=2，示例上是1
        self.stage7 = self._make_stage(repeat=1, in_channels=160, out_channels=320, stride=1, t=6)
        self.conv1 = nn.Sequential(nn.Conv2d(320,1280,kernel_size=1),
                                   nn.BatchNorm2d(1280),nn.ReLU6(inplace=True))
        self.conv2 = nn.Conv2d(1280,num_class,kernel_size=1)

    def forward(self,x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = self.conv2(x)
        x = x.reshape(x.shape[0],-1)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        lst = []
        strides = [stride] + [1] * (repeat - 1)
        for stride in strides:
            lst.append(LinearBottleNeck(in_channels,out_channels,stride,t))
            in_channels = out_channels
        return nn.Sequential(*lst)

if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=(1,3,32,32),device=device)
    net = MobileNetV2().to(device)
    print(net(x))
