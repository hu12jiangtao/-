import torch
import torchvision
from torch import nn
from torch.nn import functional as F
# seg-net并没有像FCN一样将encode部分的输出 输入到 decode内容中
# 此时同样是取出vgg-16中每个的max_pool的输出
# 前面两个max_pool之前有2个blk(一个blk有一个conv层和卷积层构成)，之后的三个max_pool之前有3个blk
class SegNet(nn.Module):
    def __init__(self,in_channel,output_channel):
        super(SegNet, self).__init__()
        self.pretrain_model = torchvision.models.vgg16(pretrained=True)
        # print(self.pretrain_model)
        # encode的第一个模块(含有两个blk)
        self.encoder_conv_00 = nn.Sequential(nn.Conv2d(in_channel,64,kernel_size=3,padding=1),nn.BatchNorm2d(64))
        self.encoder_conv_01 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64))
        # encode的第二个模块(含有两个blk)
        self.encoder_conv_10 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128))
        self.encoder_conv_11 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128))
        # encode的第三个模块(含有三个blk)
        self.encoder_conv_20 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256))
        self.encoder_conv_21 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256))
        self.encoder_conv_22 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256))
        # encode的第四个模块(含有三个blk)
        self.encoder_conv_30 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512))
        self.encoder_conv_31 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512))
        self.encoder_conv_32 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512))
        # encode的第五个模块(含有三个blk)
        self.encoder_conv_40 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512))
        self.encoder_conv_41 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512))
        self.encoder_conv_42 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512))
        self.init_vgg_weigts()
        # 解码器的第一个部分(一个和encode部分完全对称的解码器)
        self.decode_convtr_42 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.decode_convtr_41 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.decode_convtr_40 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        # 解码器的第二个部分
        self.decode_convtr_32 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.decode_convtr_31 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512))
        self.decode_convtr_30 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256))
        # 解码器的第三个部分
        self.decode_convtr_22 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256))
        self.decode_convtr_21 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256))
        self.decode_convtr_20 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128))
        # 解码器的第四个部分
        self.decode_convtr_11 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128))
        self.decode_convtr_10 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))
        # 解码器的第五个部分
        self.decode_convtr_01 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))
        self.decode_convtr_00 = nn.Sequential(nn.ConvTranspose2d(64, output_channel, kernel_size=3, padding=1))


    def forward(self,input_img):
        # 第一个模块
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, index_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True) # [batch,64,112,112]
        # 第二个模块
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, index_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True) # [batch,128,56,56]
        # 第三个模块
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, index_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True) # [batch,256,28,28]
        # 第四个模块
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, index_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True) # [batch,512,14,14]
        # 第五个模块
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, index_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)  # [batch,512,7,7]
        # decode
        # 第一个模块
        x_4d = F.max_unpool2d(x_4,index_4,kernel_size=2,stride=2)
        x_42d = F.relu(self.decode_convtr_42(x_4d))
        x_41d = F.relu(self.decode_convtr_41(x_42d))
        x_40d = F.relu(self.decode_convtr_40(x_41d))
        # 第二个模块
        x_3d = F.max_unpool2d(x_40d, index_3, kernel_size=2, stride=2)
        x_32d = F.relu(self.decode_convtr_32(x_3d))
        x_31d = F.relu(self.decode_convtr_31(x_32d))
        x_30d = F.relu(self.decode_convtr_30(x_31d))
        # 第三个模块
        x_2d = F.max_unpool2d(x_30d, index_2, kernel_size=2, stride=2)
        x_22d = F.relu(self.decode_convtr_22(x_2d))
        x_21d = F.relu(self.decode_convtr_21(x_22d))
        x_20d = F.relu(self.decode_convtr_20(x_21d))
        # 第四个模块
        x_1d = F.max_unpool2d(x_20d, index_1, kernel_size=2, stride=2)
        x_11d = F.relu(self.decode_convtr_11(x_1d))
        x_10d = F.relu(self.decode_convtr_10(x_11d))
        # 第五个模块
        x_0d = F.max_unpool2d(x_10d, index_0, kernel_size=2, stride=2)
        x_01d = F.relu(self.decode_convtr_01(x_0d))
        x_00d = self.decode_convtr_00(x_01d)

        x_softmax = F.softmax(x_00d,dim=1)
        return x_00d,x_softmax


    def init_vgg_weigts(self):
        # 第一个模块
        assert self.encoder_conv_00[0].weight.shape == self.pretrain_model.features[0].weight.shape
        self.encoder_conv_00[0].weight.data = self.pretrain_model.features[0].weight.data
        assert self.encoder_conv_00[0].bias.shape == self.pretrain_model.features[0].bias.shape
        self.encoder_conv_00[0].bias.data = self.pretrain_model.features[0].bias.data

        assert self.encoder_conv_01[0].weight.shape == self.pretrain_model.features[2].weight.shape
        self.encoder_conv_01[0].weight.data = self.pretrain_model.features[2].weight.data
        assert self.encoder_conv_01[0].bias.shape == self.pretrain_model.features[2].bias.shape
        self.encoder_conv_01[0].bias.data = self.pretrain_model.features[2].bias.data
        # 第二个模块
        assert self.encoder_conv_10[0].weight.shape == self.pretrain_model.features[5].weight.shape
        self.encoder_conv_10[0].weight.data = self.pretrain_model.features[5].weight.data
        assert self.encoder_conv_10[0].bias.shape == self.pretrain_model.features[5].bias.shape
        self.encoder_conv_10[0].bias.data = self.pretrain_model.features[5].bias.data

        assert self.encoder_conv_11[0].weight.shape == self.pretrain_model.features[7].weight.shape
        self.encoder_conv_11[0].weight.data = self.pretrain_model.features[7].weight.data
        assert self.encoder_conv_11[0].bias.shape == self.pretrain_model.features[7].bias.shape
        self.encoder_conv_11[0].bias.data = self.pretrain_model.features[7].bias.data
        # 第三个模块
        assert self.encoder_conv_20[0].weight.shape == self.pretrain_model.features[10].weight.shape
        self.encoder_conv_20[0].weight.data = self.pretrain_model.features[10].weight.data
        assert self.encoder_conv_20[0].bias.shape == self.pretrain_model.features[10].bias.shape
        self.encoder_conv_20[0].bias.data = self.pretrain_model.features[10].bias.data

        assert self.encoder_conv_21[0].weight.shape == self.pretrain_model.features[12].weight.shape
        self.encoder_conv_21[0].weight.data = self.pretrain_model.features[12].weight.data
        assert self.encoder_conv_21[0].bias.shape == self.pretrain_model.features[12].bias.shape
        self.encoder_conv_21[0].bias.data = self.pretrain_model.features[12].bias.data

        assert self.encoder_conv_22[0].weight.shape == self.pretrain_model.features[14].weight.shape
        self.encoder_conv_22[0].weight.data = self.pretrain_model.features[14].weight.data
        assert self.encoder_conv_22[0].bias.shape == self.pretrain_model.features[14].bias.shape
        self.encoder_conv_22[0].bias.data = self.pretrain_model.features[14].bias.data
        # 第四个模块
        assert self.encoder_conv_30[0].weight.shape == self.pretrain_model.features[17].weight.shape
        self.encoder_conv_30[0].weight.data = self.pretrain_model.features[17].weight.data
        assert self.encoder_conv_30[0].bias.shape == self.pretrain_model.features[17].bias.shape
        self.encoder_conv_30[0].bias.data = self.pretrain_model.features[17].bias.data

        assert self.encoder_conv_31[0].weight.shape == self.pretrain_model.features[19].weight.shape
        self.encoder_conv_31[0].weight.data = self.pretrain_model.features[19].weight.data
        assert self.encoder_conv_31[0].bias.shape == self.pretrain_model.features[19].bias.shape
        self.encoder_conv_31[0].bias.data = self.pretrain_model.features[19].bias.data

        assert self.encoder_conv_32[0].weight.shape == self.pretrain_model.features[21].weight.shape
        self.encoder_conv_32[0].weight.data = self.pretrain_model.features[21].weight.data
        assert self.encoder_conv_32[0].bias.shape == self.pretrain_model.features[21].bias.shape
        self.encoder_conv_32[0].bias.data = self.pretrain_model.features[21].bias.data
        # 第五个模块
        assert self.encoder_conv_40[0].weight.shape == self.pretrain_model.features[24].weight.shape
        self.encoder_conv_40[0].weight.data = self.pretrain_model.features[24].weight.data
        assert self.encoder_conv_40[0].bias.shape == self.pretrain_model.features[24].bias.shape
        self.encoder_conv_40[0].bias.data = self.pretrain_model.features[24].bias.data

        assert self.encoder_conv_41[0].weight.shape == self.pretrain_model.features[26].weight.shape
        self.encoder_conv_41[0].weight.data = self.pretrain_model.features[26].weight.data
        assert self.encoder_conv_41[0].bias.shape == self.pretrain_model.features[26].bias.shape
        self.encoder_conv_41[0].bias.data = self.pretrain_model.features[26].bias.data

        assert self.encoder_conv_42[0].weight.shape == self.pretrain_model.features[28].weight.shape
        self.encoder_conv_42[0].weight.data = self.pretrain_model.features[28].weight.data
        assert self.encoder_conv_42[0].bias.shape == self.pretrain_model.features[28].bias.shape
        self.encoder_conv_42[0].bias.data = self.pretrain_model.features[28].bias.data



if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=(1,3,224,224),device=device)
    net = SegNet(3,10).to(device)
    x_00d, x_softmax = net(x)
    print(x_00d)
