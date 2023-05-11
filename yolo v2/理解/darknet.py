from torch import nn
import torch

model_urls = {
    "darknet19": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth",
}

# class Conv_BN_LeakyReLU(nn.Module):
#     def __init__(self,in_channel,out_channel,k,p=0,s=1,d=1):
#         super(Conv_BN_LeakyReLU, self).__init__()
#         self.net = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=k,padding=p,stride=s,dilation=d),
#                                  nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.1,inplace=True))
#
#     def forward(self,x):
#         return self.net(x)

# class DarkNet_19(nn.Module):  # 已经过验证，证明其是正确的
#     # 自己写的，如是需要导入权重参数时则会发生错误
#     def __init__(self):
#         super(DarkNet_19, self).__init__()
#         # part1
#         self.part1 = nn.Sequential(Conv_BN_LeakyReLU(3, 32, 3, 1), nn.MaxPool2d((2,2),2))
#         # part2
#         self.part2 = nn.Sequential(Conv_BN_LeakyReLU(32, 64, 3, 1), nn.MaxPool2d((2,2),2))
#         # part3
#         self.part3 = nn.Sequential(Conv_BN_LeakyReLU(64, 128, 3, 1), Conv_BN_LeakyReLU(128, 64, 1),
#                                    Conv_BN_LeakyReLU(64, 128, 3, 1), nn.MaxPool2d((2,2),2))
#         # part4
#         self.part4 = nn.Sequential(Conv_BN_LeakyReLU(128, 256, 3, 1), Conv_BN_LeakyReLU(256, 128, 1),
#                                    Conv_BN_LeakyReLU(128, 256, 3, 1),nn.MaxPool2d((2,2),2))
#         # part5
#         self.part5 = nn.Sequential(Conv_BN_LeakyReLU(256, 512, 3, 1), Conv_BN_LeakyReLU(512, 256, 1),
#                                    Conv_BN_LeakyReLU(256, 512, 3, 1), Conv_BN_LeakyReLU(512, 256, 1),
#                                    Conv_BN_LeakyReLU(256, 512, 3, 1)) # 此时得到的结果作为细粒度特征(浅层语意特征和之后的深层语意特征进行融合)
#         self.part5_maxpool = nn.MaxPool2d((2,2),2)
#         # part6
#         self.part6 = nn.Sequential(Conv_BN_LeakyReLU(512, 1024, 3, 1), Conv_BN_LeakyReLU(1024, 512, 1),
#                                    Conv_BN_LeakyReLU(512, 1024, 3, 1), Conv_BN_LeakyReLU(1024, 512, 1),
#                                    Conv_BN_LeakyReLU(512, 1024, 3, 1))
#     def forward(self,x):
#         c1 = self.part1(x) # [b, 32, h/2, w/2]
#         c2 = self.part2(c1) # [b, 64, h/4, w/4]
#         c3 = self.part3(c2) # [b, 128, h/8, w/8]
#         c3 = self.part4(c3) # [b, 256, h/16, w/16]
#         c4 = self.part5(c3) # [b, 512, h/16, w/16]
#         c5 = self.part6(self.part5_maxpool(c4)) # [b, 1024, h/32, w/32]
#         outputs = {'c3':c3, 'c4':c4, 'c5':c5}
#         return outputs


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

class DarkNet_19(nn.Module):
    def __init__(self):
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

    def forward(self, x):
        c1 = self.conv_1(x)  # [B, C1, H/2, W/2]
        c2 = self.conv_2(c1)  # [B, C2, H/4, W/4]
        c3 = self.conv_3(c2)  # [B, C3, H/8, W/8]
        c3 = self.conv_4(c3)  # [B, C3, H/8, W/8]
        c4 = self.conv_5(self.maxpool_4(c3))  # [B, C4, H/16, W/16]
        c5 = self.conv_6(self.maxpool_5(c4))  # [B, C5, H/32, W/32]

        output = {
            'c3': c3,
            'c4': c4,
            'c5': c5
        }
        return output



def build_darknet19(pretrained=False):
    # model
    model = DarkNet_19()
    feat_dims = [256, 512, 1024]

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet19']
        # checkpoint state dict
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model, feat_dims




if __name__ == '__main__':
    # torch.manual_seed(1)
    # device = torch.device('cuda')
    # x = torch.randn(size=(1,3,64,64),device=device)
    # net = DarkNet_19().to(device)
    # out = net(x)
    # print(out['c3'])
    # print(out['c4'])
    # print(out['c5'])
    device = torch.device('cuda')
    model, feat_dim = build_darknet19(pretrained=True)
    x = torch.randn(size=(1,3,224,224),device=device)
    model.to(device)
    outputs = model(x)
    print(outputs['c4'].shape)


