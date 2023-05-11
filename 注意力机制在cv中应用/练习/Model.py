from torch import nn
import torch
from torch.nn import functional as F
import math
import torchvision

class Se_net(nn.Module):
    # 对于每个像素的不同通道分配不同的权重(对于所有的像素点的通道权重都是相同的)
    def __init__(self,in_channel,ratio):
        super(Se_net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channel, in_channel // ratio, bias=False),
                                nn.ReLU(),nn.Linear(in_channel // ratio, in_channel, bias=False),
                                nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x): # x = [b,c,h,w]
        b, c, h, w = x.shape
        # 此时的fc相当于weight权重
        avg = self.avg_pool(x).reshape(b,c)
        fc = self.fc(avg)
        fc = fc.reshape(b,c,1,1)
        # x相当于key
        return fc * x  # 相当于给每个通道分配一个权重

class ECA_NET(nn.Module):
    # 此时将用来求解权重的全连接层变换为一维的卷积，减少了训练参数的个数，加快了训练速度
    # 此时的卷积核的个数是根据输入的通道数确定的
    def __init__(self,in_channel,gamma = 2,beta = 1):
        super(ECA_NET, self).__init__()
        kernel_size = int(abs((math.log(in_channel,2) + beta) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=padding,bias=False)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()  # 针对于输入的最后一个维度

    def forward(self,x):
        avg = self.avg_pool(x) # [batch,c,1,1]
        avg = avg.permute(0,2,1,3).squeeze(-1) # [batch,1,c]
        weight = self.conv(avg)
        weight = self.sigmoid(weight) # [batch,1,c]
        weight = weight.permute(0,2,1).unsqueeze(-1)
        return weight * x

# 通道注意力机制 + 空间注意力机制
class ChannelAttention(nn.Module): # 通道注意力机制
    def __init__(self,in_channel,ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # [b, c, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel // ratio, in_channel, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y1 = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        y2 = self.conv2(self.relu(self.conv1(self.max_pool(x)))) # [b, c, 1, 1]
        y = y1 + y2
        weight = self.sigmoid(y)
        return weight  # 返回每个通道的权重

class SpatialAttention(nn.Module): # 空间注意力机制
    def __init__(self,kernel_size):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x): # [batch,c,h,w]
        max_value = torch.max(x,dim=1,keepdim=True) # [batch,1,h,w]
        mean_value = torch.mean(x,dim=1,keepdim=True) # [batch,1,h,w]
        value = torch.cat([max_value,mean_value],dim=1) # [batch,2,h,w]
        weight = self.conv(value)
        weight = self.sigmoid(weight) # [batch,1,h,w]
        return weight

class cbam_block(nn.Module):
    def __init__(self,in_channel,ratio=8,kernel_size=7):
        super(cbam_block, self).__init__()
        # 此时分为了通道注意力机制 + 空间注意力机制
        # 通道注意力机制:对于输入的x 经过最大池化和平均池化得到x_1,x_2后,经过一个全连接层得到y_1,y_2，之后将两者相加后进行sigmoid得到每个通道的权重
        # 空间注意力机制:对于输入的x 经过每个像素的通道的求解最大值和均值得到x_1([1,h,w]),x_2,将两者concat后经过一个卷积后得到y，将y进行sigmoid得到每个像素的权重
        self.channelattention = ChannelAttention(in_channel,ratio)
        self.spatialattention = SpatialAttention(kernel_size)

    def forward(self,x):
        out1 = self.channelattention(x) * x
        out2 = self.spatialattention(out1) * out1
        return out2

# 创建Resnet18架构的网络
class CifarSEBasicBlock(nn.Module): # 每个残差块的架构
    def __init__(self,in_channel,out_channel,stride,is_attention=True):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if is_attention:
            self.se = ECA_NET(out_channel)
        else:
            self.se = lambda x:x

        if stride != 1 or out_channel != in_channel:
            self.downsample = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride),
                                            nn.BatchNorm2d(out_channel))
        else:
            self.downsample = lambda x:x

    def forward(self,x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out3 = self.se(out2)
        out = F.relu(out3 + self.downsample(x))
        return out


class CifarSEResNet(nn.Module):  # 此时相当于一个resnet20的网络
    def __init__(self,block, in_channel,basic_channel,is_attention=True):
        super(CifarSEResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,basic_channel,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(basic_channel)
        if is_attention is True:
            self.layer1 = self._make_layer(block,basic_channel,basic_channel, stride=1, n_block=3, is_attention=True)
            self.layer2 = self._make_layer(block,basic_channel, basic_channel * 2, stride=2, n_block=3, is_attention=True)
            self.layer3 = self._make_layer(block,basic_channel * 2, basic_channel * 4, stride=2,n_block=3, is_attention=True)
        else:
            self.layer1 = self._make_layer(block,basic_channel,basic_channel, stride=1, n_block=3, is_attention=False)
            self.layer2 = self._make_layer(block,basic_channel, basic_channel * 2, stride=2, n_block=3, is_attention=False)
            self.layer3 = self._make_layer(block,basic_channel * 2, basic_channel * 4, stride=2,n_block=3, is_attention=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(basic_channel * 4, 10)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self,block,in_channel,out_channel,stride,n_block,is_attention):
        layers = []
        strides = [stride] + [1] * (n_block - 1)
        for stride in strides:
            layers.append(block(in_channel,out_channel,stride,is_attention))
            in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        return self.fc(x)

if __name__ == '__main__':
    device = torch.device('cuda')
    net = CifarSEResNet(CifarSEBasicBlock, in_channel=3,basic_channel=16).to(device)
    x = torch.randn(size=(1,3,224,224),device=device)
    print(net(x).shape)


