from torch import nn
import torch
# 在模型结构中需要体现α
class DepthwiseSeparableConv(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,alpha=1.):
        super(DepthwiseSeparableConv, self).__init__()
        # stride作用与深度可分离卷积的depth-wise模块中
        in_channel = int(alpha * in_channel)
        out_channel = int(alpha * out_channel)
        # depth-wise
        self.conv1 = nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,stride=stride,bias=False,groups=in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        # Point-wise(1x1卷积)
        self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        out1 = self.relu1(self.bn1(self.conv1(x)))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        return out2

class MobileNet(nn.Module):
    def __init__(self,alpha = 1.):
        super(MobileNet, self).__init__()
        # 第一个模块使用标准卷积
        self.conv = nn.Conv2d(3,int(alpha * 32),kernel_size=3,padding=1,stride=2,bias=False)
        self.bn = nn.BatchNorm2d(int(alpha * 32))
        self.relu = nn.ReLU(inplace=True)
        # 深度可分离卷积模块
        self.ds_conv_1 = DepthwiseSeparableConv(in_channel=32,out_channel=64,stride=1,alpha=alpha)
        self.ds_conv_2 = DepthwiseSeparableConv(in_channel=64, out_channel=128, stride=2, alpha=alpha)
        self.ds_conv_3 = DepthwiseSeparableConv(in_channel=128, out_channel=128, stride=1, alpha=alpha)
        self.ds_conv_4 = DepthwiseSeparableConv(in_channel=128, out_channel=256, stride=2, alpha=alpha)
        self.ds_conv_5 = DepthwiseSeparableConv(in_channel=256, out_channel=256, stride=1, alpha=alpha)
        self.ds_conv_6 = DepthwiseSeparableConv(in_channel=256, out_channel=512, stride=2, alpha=alpha)

        self.ds_conv_7_1 = DepthwiseSeparableConv(in_channel=512, out_channel=512, stride=1, alpha=alpha)
        self.ds_conv_7_2 = DepthwiseSeparableConv(in_channel=512, out_channel=512, stride=1, alpha=alpha)
        self.ds_conv_7_3 = DepthwiseSeparableConv(in_channel=512, out_channel=512, stride=1, alpha=alpha)
        self.ds_conv_7_4 = DepthwiseSeparableConv(in_channel=512, out_channel=512, stride=1, alpha=alpha)
        self.ds_conv_7_5 = DepthwiseSeparableConv(in_channel=512, out_channel=512, stride=1, alpha=alpha)

        self.ds_conv_8 = DepthwiseSeparableConv(in_channel=512, out_channel=1024, stride=2, alpha=alpha)
        self.ds_conv_9 = DepthwiseSeparableConv(in_channel=1024, out_channel=1024, stride=2, alpha=alpha)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(alpha * 1024), 2)

    def get_param_num(self):
        param_count = 0
        for param in self.parameters():
            param_count += param.numel()
        return param_count


    def forward(self,x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.ds_conv_1(x)
        x = self.ds_conv_2(x)
        x = self.ds_conv_3(x)
        x = self.ds_conv_4(x)
        x = self.ds_conv_5(x)
        x = self.ds_conv_6(x)

        x = self.ds_conv_7_1(x)
        x = self.ds_conv_7_2(x)
        x = self.ds_conv_7_3(x)
        x = self.ds_conv_7_4(x)
        x = self.ds_conv_7_5(x)

        x = self.ds_conv_8(x)
        x = self.ds_conv_9(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

class Normal_Conv(nn.Module):
    def __init__(self):
        super(Normal_Conv, self).__init__()
        self.blk1 = blk(3,32,stride=2)
        self.blk2 = nn.Sequential(blk(32,64,stride=1),blk(64,128,stride=2))
        self.blk3 = nn.Sequential(blk(128,128,stride=1),blk(128,256,stride=2))
        self.blk4 = nn.Sequential(blk(256, 256, stride=1), blk(256, 512, stride=2))
        self.blk5 = nn.Sequential(blk(512, 512, stride=1),blk(512, 512, stride=1),blk(512, 512, stride=1),
                                  blk(512, 512, stride=1),blk(512, 512, stride=1))
        self.blk6 = nn.Sequential(blk(512, 1024, stride=2),blk(1024, 1024, stride=2))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 2)

    def forward(self,x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0],-1)
        return self.fc(x)

    def get_param_num(self):
        param_count = 0
        for param in self.parameters():
            param_count += param.numel()
        return param_count


class blk(nn.Module):
    def __init__(self,in_channel,ouu_channel,stride):
        super(blk, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channel,ouu_channel,kernel_size=3,padding=1,stride=stride,bias=False),
                                 nn.BatchNorm2d(ouu_channel),nn.ReLU())
    def forward(self,x):
        return self.net(x)



if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=(1,3,224,224),device=device)
    net1 = MobileNet(1).to(device)
    print(net1.get_param_num())
    net = Normal_Conv().to(device)
    print(net.get_param_num())




