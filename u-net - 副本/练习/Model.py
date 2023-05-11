from torch import nn
import torch
from torch.nn import functional as F

class DoubleConv(nn.Module): # 不改变宽高，只改变通道数
    def __init__(self,in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        if mid_channel is None:
            mid_channel = out_channel
        self.net = nn.Sequential(nn.Conv2d(in_channel,mid_channel,kernel_size=3,padding=1,bias=False),
                                 nn.BatchNorm2d(mid_channel),nn.ReLU(inplace=True),
                                 nn.Conv2d(mid_channel,out_channel,kernel_size=3,padding=1,bias=False),
                                 nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True))
    def forward(self,x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Down, self).__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
                                 DoubleConv(in_channel,out_channel))
    def forward(self,x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Up, self).__init__()
        self.transpose = nn.ConvTranspose2d(in_channel ,in_channel // 2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channel,out_channel)

    def forward(self,x1,x2):
        # x1.shape=[b,in_channel, h/2, w/2], x2.shape=[b,in_channel // 2, h, w]
        x1 = self.transpose(x1) # [b,in_channel // 2, h, w]
        # 确保经过转秩卷积的x1和x2的宽高是相同的
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1,(diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2)) # 填充应为左右上下
        out = torch.cat([x2,x1],dim=1) # [b,in_channel, h, w]
        out = self.conv(out) # [b,out_channel,h,w]
        return out

class Unet(nn.Module):
    def __init__(self,in_channel,num_classes,base_channel):
        super(Unet, self).__init__()
        self.encode_in = DoubleConv(in_channel,base_channel)
        channel_lst = [base_channel, base_channel * 2, base_channel * 2**2, base_channel * 2**3, base_channel * 2**4]
        # 进行下采样
        self.down1 = Down(channel_lst[0], channel_lst[1])
        self.down2 = Down(channel_lst[1], channel_lst[2])
        self.down3 = Down(channel_lst[2], channel_lst[3])
        self.down4 = Down(channel_lst[3], channel_lst[4]) # 此时的输出应当为[1024,h/16,w/16]
        # 进行上采样
        self.up1 = Up(channel_lst[4], channel_lst[3]) # 经过后输出[512,h/8,w/8]
        self.up2 = Up(channel_lst[3], channel_lst[2]) # 经过后输出[256,h/4,w/4]
        self.up3 = Up(channel_lst[2], channel_lst[1]) # 经过后输出[128,h/2,w/2]
        self.up4 = Up(channel_lst[1], channel_lst[0]) # 经过后输出[64,h,w]
        # 类别的分类
        self.out = nn.Conv2d(base_channel,num_classes,kernel_size=1)

    def forward(self,x): # x.shape=[batch,3,h,w]
        encode_in = self.encode_in(x) # [batch,64,h,w]
        # down sample
        score1 = self.down1(encode_in) # [batch,128,h/2,w/2]
        score2 = self.down2(score1) # [batch,256,h/4,w/4]
        score3 = self.down3(score2) # [batch,512,h/8,w/8]
        score4 = self.down4(score3) # [batch,1024,h/16,w/16]
        # up sample
        decode_1 = self.up1(score4,score3) # [batch,512,h/8,w/8]
        decode_2 = self.up2(decode_1,score2) # [batch,256,h/4,w/4]
        decode_3 = self.up3(decode_2,score1) # [batch,128,h/2,w/2]
        decode_4 = self.up4(decode_3,encode_in) # [batch,64,h,w]
        # output
        out = self.out(decode_4)
        return out

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=[1,3,584,564],device=device)
    net = Unet(3,2,64).to(device)
    print(net(x))

