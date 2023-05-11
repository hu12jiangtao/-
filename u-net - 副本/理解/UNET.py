import torch
from torch import nn
import config
from torch.nn import functional as F

class DoubleConv1(nn.Sequential):
    # 此时的DoubleConv(3,6)相当于nn.Sequential(nn.Conv2d(3,6,3,padding=1,bias=True),nn.BatchNorm2d(6),nn.ReLU(inplace=True),
    #                                        nn.Conv2d(6,6,3,padding=1,bias=True),nn.BatchNorm2d(6),nn.ReLU(inplace=True))
    def __init__(self,in_channel,out_channel,mid_channel=None):
        if mid_channel is None:
            mid_channel = out_channel
        super(DoubleConv1, self).__init__(nn.Conv2d(in_channel,mid_channel,kernel_size=3,padding=1,bias=True),
                                          nn.BatchNorm2d(mid_channel),nn.ReLU(inplace=True),
                                          nn.Conv2d(mid_channel,out_channel,kernel_size=3,padding=1),
                                          nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True))


class DoubleConv2(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel=None):
        super(DoubleConv2, self).__init__()
        if mid_channel is None:
            mid_channel = out_channel
        self.part1 = nn.Sequential(nn.Conv2d(in_channel,mid_channel,kernel_size=3,padding=1,bias=False),
                                   nn.BatchNorm2d(mid_channel),nn.ReLU(inplace=True))
        self.part2 = nn.Sequential(nn.Conv2d(mid_channel,out_channel,kernel_size=3,padding=1,bias=False),
                                   nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True))
    def forward(self,x):
        return self.part2(self.part1(x))


class Down1(nn.Sequential): # 和Down2得到的效果相同
    # 此时encode的一个模块的部分等价于nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),DoubleConv(in_channel,out_channel))
    def __init__(self,in_channel,out_channel):
        super(Down1, self).__init__(nn.MaxPool2d(kernel_size=2,stride=2),
                                    DoubleConv1(in_channel,out_channel))

class Down2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Down2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv = DoubleConv2(in_channel,out_channel)

    def forward(self,x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    # 先反卷积之后进行concat最后进行两次的卷积
    def __init__(self,in_channel, out_channel):
        super(Up, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2(in_channel, out_channel)
    def forward(self,x1,x2):
        # 此时x1=[batch, in_channel, h1/2, w1/2],x2=[batch, in_channel/2, h2, w2]
        x1 = self.up_sample(x1) # [batch, in_channel/2, h1, w1]
        # 当h1 * 2 和 h2 不相同时(造成这种问题是由于在encode的池化时宽高下采样)，此时需要进行padding
        # pad的填充顺序为左右上下
        diff_h = x2.shape[2] - x1.shape[2] # 高
        diff_w = x2.shape[3] - x1.shape[3] # 宽
        x1 = F.pad(x1,[diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        out = torch.cat([x2,x1],dim=1) # [batch,in_channel,h2,w2]
        return self.conv(out) # [batch, out_channel, h2, w2]

class Unet(nn.Module):
    def __init__(self,in_channel,num_class,base_channel):
        super(Unet, self).__init__()
        self.encode_part1 = DoubleConv2(in_channel,base_channel)
        encode_channel_lst = [base_channel, base_channel * 2, base_channel * 2**2, base_channel * 2**3, base_channel * 2**4]
        # encode部分
        self.down1 = Down2(encode_channel_lst[0], encode_channel_lst[1])
        self.down2 = Down2(encode_channel_lst[1], encode_channel_lst[2])
        self.down3 = Down2(encode_channel_lst[2], encode_channel_lst[3])
        self.down4 = Down2(encode_channel_lst[3], encode_channel_lst[4])
        # decode部分
        decode_channel_lst = [base_channel * 2**4, base_channel * 2**3, base_channel * 2**2, base_channel * 2, base_channel]
        self.up1 = Up(decode_channel_lst[0], decode_channel_lst[1])
        self.up2 = Up(decode_channel_lst[1], decode_channel_lst[2])
        self.up3 = Up(decode_channel_lst[2], decode_channel_lst[3])
        self.up4 = Up(decode_channel_lst[3], decode_channel_lst[4])
        # 输出
        self.out = nn.Conv2d(base_channel,num_class,kernel_size=1)

    def forward(self,x): # 若x的输入为[batch,1,480,480]
        # 解码器的部分
        x = self.encode_part1(x) # [batch,64,480,480]

        encode_out1 = self.down1(x) # [batch,128,240,240]
        encode_out2 = self.down2(encode_out1) # [batch,256,120,120]
        encode_out3 = self.down3(encode_out2) # [batch,512,60,60]
        encode_out4 = self.down4(encode_out3) # [batch,1024,30,30]

        # 解码器的部分
        decode_out1 = self.up1(encode_out4, encode_out3) # [batch,512,60,60]
        decode_out2 = self.up2(decode_out1, encode_out2) # [batch,256,120,120]
        decode_out3 = self.up3(decode_out2, encode_out1) # [batch,128,240,240]
        decode_out4 = self.up4(decode_out3, x) # [batch,64,480,480]
        # 输出
        return self.out(decode_out4)

if __name__ == '__main__':
    torch.manual_seed(1)
    device = config.device
    x = torch.randn(size=(1,3,584,564),device=device)
    net = Unet(in_channel=3,num_class=2,base_channel=64).to(config.device)
    print(net(x))


