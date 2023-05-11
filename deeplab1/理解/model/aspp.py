from torch import nn
import torch
from torch.nn import functional as F

# 此时输入aspp模块的特征图的形状为[batch,2048,h/16,w/16]
# 此时相对于原论文中concat后面缺少了dropout的操作，同时少了一个3x3的卷积模块
class ASPP_Bottleneck(nn.Module):
    def __init__(self,num_classes):
        super(ASPP_Bottleneck, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(2048, 256, kernel_size=3,stride=1,padding=6,dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(2048, 256, kernel_size=3,stride=1,padding=12,dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(2048, 256, kernel_size=3,stride=1,padding=18,dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self,feature_map):
        feature_map_h = feature_map.shape[2]
        feature_map_w = feature_map.shape[3]
        out_1x1_1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # [batch,256,h/16,w/16]
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # [batch,256,h/16,w/16]
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # [batch,256,h/16,w/16]
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # [batch,256,h/16,w/16]

        out_image = self.avg_pool(feature_map)
        out_image = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_image)))
        out_image = F.upsample(out_image,(feature_map_h,feature_map_w),mode='bilinear') # [batch,256,h/16,w/16]

        out = torch.cat([out_1x1_1,out_3x3_1,out_3x3_2,out_3x3_3,out_image],dim=1) # [batch,1024,h/16,w/16]
        out = F.relu(self.bn_conv_1x1_2(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)
        return out



if __name__ =='__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=(2,2048,120,120),device=device)
    net = ASPP_Bottleneck(20).to(device)
    print(net(x))





