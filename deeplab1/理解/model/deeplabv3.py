import torch
from torch import nn
from model import aspp
from model import resnet
from torch.nn import functional as F

class DeepLabV3(nn.Module):
    def __init__(self,num_classes):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.resnet = resnet.ResNet_Bottleneck_OS16()
        self.aspp = aspp.ASPP_Bottleneck(self.num_classes)

    def forward(self,image):
        h = image.shape[2]
        w = image.shape[3]
        out1 = self.resnet(image)
        out2 = self.aspp(out1)
        out = F.upsample(out2,size=(h,w),mode='bilinear')
        return out

if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=(2,3,480,480),device=device)
    net = DeepLabV3().to(device)
    print(net(x))