from torch import nn
import torch


class BasicConv2d(nn.Module): # 输入层
    def __init__(self,in_channel,out_channel,kernel_size,padding,stride):
        super(BasicConv2d, self).__init__()
        self.op = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
                                nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True))
    def forward(self,x):
        return self.op(x)

# ChannelShuffle + DepthwiseConv2d + PointwiseConv2d得到shuffle_net v1的基本模块
class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch, channel, height, weight = x.shape
        channel_per_group = int(channel // self.groups)
        x = x.reshape(batch, self.groups, channel_per_group, height, weight)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(batch, -1, height, weight)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,groups):
        super(DepthwiseConv2d, self).__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,
                                          stride=stride,padding=padding,groups=groups),
                                nn.BatchNorm2d(out_channel))
    def forward(self,x):
        return self.op(x)

class PointwiseConv2d(nn.Module): # 1x1的分组卷积
    def __init__(self,in_channel,out_channel,groups):
        super(PointwiseConv2d, self).__init__()
        self.op = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=1,groups=groups),
                                nn.BatchNorm2d(out_channel))

    def forward(self,x):
        return self.op(x)

class ShuffleNetUnit(nn.Module):
    def __init__(self,in_channel, out_channel, stride, groups):
        super(ShuffleNetUnit, self).__init__()
        # 首先升维到out_channel的0.25后进行depth-wise之后在此升维到out_channel
        self.bottlneck = nn.Sequential(PointwiseConv2d(in_channel, out_channel // 4, groups),nn.ReLU(inplace=True))
        # 之后进行通道重拍操作
        self.channel_shuffle = ChannelShuffle(groups)
        # 之后利用depth-wise进行处理
        self.depth_wise = DepthwiseConv2d(out_channel // 4, out_channel // 4,
                                          kernel_size=3, stride=stride, groups=out_channel // 4)

        # 进行残差连接
        if stride == 2 or in_channel != out_channel:
            self.fusion = self.concat_
            self.short_cup = nn.Sequential(nn.AvgPool2d(kernel_size=3,stride=stride,padding=1))
            self.expand = PointwiseConv2d(out_channel // 4, out_channel - in_channel, groups)
        else:
            self.fusion = self.add_
            self.short_cup = nn.Sequential()
            # 之后进行point-wise处理
            self.expand = PointwiseConv2d(out_channel // 4, out_channel, groups)
        # 残差连接之后的Relu操作
        self.relu = nn.ReLU(inplace=True)

    def concat_(self,x,y):
        return torch.cat([x,y],dim=1)

    def add_(self,x,y):
        return x + y

    def forward(self,x):
        short_cut = self.short_cup(x)
        x = self.bottlneck(x)
        x = self.channel_shuffle(x)
        x = self.depth_wise(x)
        x = self.expand(x)
        x = self.fusion(short_cut,x)
        return self.relu(x)

# 构建整一个的shuffle_net v1网络
class ShuffleNet(nn.Module):
    def __init__(self,num_block,num_classes=100,groups=3):
        super(ShuffleNet, self).__init__()
        out_channels = [24,240,480,960]
        # 模型的输入层
        self.inputs = BasicConv2d(in_channel=3,out_channel=out_channels[0],kernel_size=3,stride=1,padding=1)
        self.in_channel = out_channels[0]
        # 构建shuffle net的第一个stage
        self.stage2 = self._make_stage(ShuffleNetUnit,num_block[0],out_channels[1],stride=2,groups=groups)
        # 构建shuffle net的第二个stage
        self.stage3 = self._make_stage(ShuffleNetUnit, num_block[1], out_channels[2], stride=2, groups=groups)
        # 构建shuffle net的第三个stage
        self.stage4 = self._make_stage(ShuffleNetUnit, num_block[2], out_channels[3], stride=2, groups=groups)
        # 最后的输出层
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels[-1], num_classes)

    def forward(self,x):
        x = self.inputs(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

    def _make_stage(self, block, num_blocks, output_channels, stride, groups):
        strides = [stride] + [1] * (num_blocks - 1)
        stage = []
        for stride in strides:
            stage.append(block(self.in_channel, output_channels, stride, groups))
            self.in_channel = output_channels
        return nn.Sequential(*stage)


if __name__ == '__main__':
    device = torch.device('cuda')
    torch.manual_seed(1)
    x = torch.randn(size=(1,3,32,32))
    net = ShuffleNet([4,8,4])
    print(sum(param.numel() for param in net.parameters()))




