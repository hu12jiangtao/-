# 用来存储学生网络的模型（此时不在利用残差模块以及正则化的手段）
from torch import nn
import torch

class res_block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(res_block, self).__init__()
        self.forward_blk = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=2),
                             nn.BatchNorm2d(out_channel),nn.ReLU())

    def forward(self,x):
        return self.forward_blk(x)


class Conv_net(nn.Module):
    def __init__(self,blk_num=3):
        super(Conv_net, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2),nn.BatchNorm2d(64),
                                    nn.ReLU(),nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
        conv_blk = nn.ModuleList()
        in_channel = 64
        out_channel = in_channel * 2
        for i in range(blk_num):
            conv_blk.append(res_block(in_channel,out_channel))
            in_channel = out_channel
            out_channel = in_channel * 2
        self.block2 = nn.Sequential(*conv_blk,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        self.fc = nn.Linear(512, 10)

    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        return self.fc(out)

if __name__ == '__main__':
    net = Conv_net()
    net_num = sum([i.numel() for i in net.parameters()])
    print(net_num)