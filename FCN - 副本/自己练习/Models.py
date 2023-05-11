from torch import nn
import torch
import torchvision
import cfg
import numpy as np

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
        pretrain_model = torchvision.models.vgg16_bn(pretrained=True) # 导入预训练模型
        self.stage1 = pretrain_model.features[:7]
        self.stage2 = pretrain_model.features[7:14]
        self.stage3 = pretrain_model.features[14:24]
        self.stage4 = pretrain_model.features[24:34]
        self.stage5 = pretrain_model.features[34:]
        # 反卷积的部分(不改变通道数)
        self.upsample_2x_1 = nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernel(512, 512, 4)
        self.upsample_2x_2 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4)
        self.upsample_8x = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=16,stride=8,padding=4,bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        # 修改特征图通道的卷积
        self.conv_trans1 = nn.Conv2d(512,256,kernel_size=1)
        self.conv_trans2 = nn.Conv2d(256,num_classes,kernel_size=1)

    def forward(self,x):
        score1 = self.stage1(x) # [batch, 64, h/2, w/2]
        score2 = self.stage2(score1) # [batch, 128, h/4, w/4]
        score3 = self.stage3(score2) # [batch, 256, h/8, w/8]
        score4 = self.stage4(score3) # [batch, 512, h/16, w/16]
        score5 = self.stage5(score4) # [batch, 512, h/32, w/32]

        score5 = self.upsample_2x_1(score5) # [batch,512, h/16, w/16]
        add1 = score5 + score4 # [batch,512, h/16, w/16]
        add1 = self.conv_trans1(add1) # [batch,256, h/16, w/16]
        add1 = self.upsample_2x_2(add1) # [batch,256, h/8, w/8]
        add2 = add1 + score3
        add2 = self.conv_trans2(add2) # [batch,num_classes, h/8, w/8]
        out = self.upsample_8x(add2) # [batch,num_classes, h, w]
        return out


if __name__ == '__main__':
    torch.manual_seed(1)
    model = FCN(12).to(cfg.device)
    model_param_path = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\理解\\param.params'
    model.load_state_dict(torch.load(model_param_path))
    x = torch.randn(size=[1,3,224,224],device=cfg.device)
    y = model(x)
    print(y)