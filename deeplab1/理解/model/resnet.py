from torch import nn
import torch
from torch.nn import functional as F
import torchvision

# 此时的resnet50的layer3是没有进行改变的(下采样的倍率为16),layer4中的每个blk的膨胀系数都是2
# 在原文中根据Multi_grad的策略得到的layer4的膨胀系数为[2,4,8],且layer3中的宽高没有下降一倍(下采样的倍率为8)

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self):
        super(ResNet_Bottleneck_OS16, self).__init__()
        resnet = torchvision.models.resnet50()
        # 针对于resnet50来说，根据论文前面3个layer1,layer2,layer3中没有使用膨胀卷积
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        # self.layer5即是resnet中的layer4，在原文中膨胀系数为[2,4,8],此时则是全部为2
        self.layer5 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self,x):
        out = self.resnet(x)
        return self.layer5(out)


class Bottleneck(nn.Module): # 对于其中的一个残差块(三个残差块构成了一个layer4)
    expansion = 4
    def __init__(self,in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        # 每个layer最终输出的通道数应当是channels * 4， 中间变化的通道数为channels
        # dilation针对的是3*3的卷积，stride也是针对3*3的卷积
        out_channel = channels * 4
        self.conv1 = nn.Conv2d(in_channels,channels,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,stride=stride,dilation=dilation,padding=dilation,bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels,out_channel,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        if stride != 1 or in_channels != out_channel:
            self.conv = nn.Conv2d(in_channels,out_channel,kernel_size=1,stride=stride,bias=False)
            self.bn = nn.BatchNorm2d(out_channel)
            self.down_sample = nn.Sequential(self.conv,self.bn)
        else:
            self.down_sample = nn.Sequential()

    def forward(self,x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        return F.relu(x3 + self.down_sample(x))

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    stride = [stride] + [1] * (num_blocks - 1)
    blocks = []
    for i in range(len(stride)):
        blocks.append(block(in_channels, channels, stride[i], dilation))
        in_channels = block.expansion * channels
    return nn.Sequential(*blocks)




if __name__ == "__main__":
    if __name__ == '__main__':
        torch.manual_seed(1)
        device = torch.device('cuda')
        net = ResNet_Bottleneck_OS16().to(device)
        x = torch.randn(size=(1, 3, 224, 224), device=device)
        print(net(x))
