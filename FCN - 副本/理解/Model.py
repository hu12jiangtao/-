from torch import nn
import torch
import torchvision
import numpy as np

# 此时利用双线性差值法给转秩卷积的卷积核赋初值，转秩卷积的卷积核的形状为[out_channel,in_channel,h,w]
def bilinear_kernel(in_channels, out_channels, kernel_size): # 此时的宽高都是kernel_size
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    # 分别用于两个数组的生成，第一个数组np.arrange(0,kernel_size),第二个数组np.arrange(0,kernel_size)
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)  # 赋了初值
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)

class FCN(nn.Module):
    def __init__(self,num_classes):
        super(FCN, self).__init__()
        pretrain_model = torchvision.models.vgg16_bn(pretrained=True) # 每一个最大池化层后面最为特征输出
        self.stage1 = pretrain_model.features[:7] # 此时输出的特征图:[batch,64,112,112]
        self.stage2 = pretrain_model.features[7:14] # 此时输出的特征图:[batch,128,56,56]
        self.stage3 = pretrain_model.features[14:24] # 此时输出的特征图:[batch,256,28,28]
        self.stage4 = pretrain_model.features[24:34] # 此时输出的特征图:[batch,512,14,14]
        self.stage5 = pretrain_model.features[34:] # 此时输出的特征图:[batch,512,7,7]
        # s4,s5的反卷积层(宽高都扩大两倍)
        self.upsample_2x_1 = nn.ConvTranspose2d(512,512,kernel_size=4,padding=1,stride=2,bias=False) # 针对于宽高为原先的1/32
        self.upsample_2x_1.weight.data = bilinear_kernel(in_channels=512, out_channels=512, kernel_size=4)
        self.upsample_2x_2 = nn.ConvTranspose2d(256,256,kernel_size=4,padding=1,stride=2,bias=False) # 针对于宽高为原先的1/16
        self.upsample_2x_2.weight.data = bilinear_kernel(in_channels=256, out_channels=256, kernel_size=4)
        self.upsample_8x = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=16,stride=8,padding=4,bias=False) # 还原至原先输入图片的尺寸大小
        self.upsample_8x.weight.data = bilinear_kernel(in_channels=num_classes, out_channels=num_classes, kernel_size=16)
        # 用来改变通道数的卷积层(宽高不发生变化，通道数减小)
        self.conv_trans1 = nn.Conv2d(512,256,kernel_size=1)
        self.conv_trans2 = nn.Conv2d(256,num_classes,kernel_size=1)

    def forward(self,x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        s5 = self.upsample_2x_1(s5) # [batch,512,14,14]
        add1 = s5 + s4 # [batch,512,14,14]
        add1 = self.conv_trans1(add1) # [batch,256,14,14]

        add1 = self.upsample_2x_2(add1) # [batch,256,28,28]
        add2 = add1 + s3 # [batch,256,28,28]
        add2 = self.conv_trans2(add2) # [batch,num_classes,28,28]

        # 此时需要扩大8倍还原至原先图片的大小
        output = self.upsample_8x(add2)
        return output


if __name__ == '__main__':
    torch.manual_seed(1)
    model = FCN(12)
    device = torch.device('cuda')
    model.to(device)
    x = torch.randn(size=(1,3,224,224),device=device)
    y = model(x)
    print(y)