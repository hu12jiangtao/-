from torch import nn
import torch
import datasets
import cfg
from torch.utils import data
import Model
import pandas as pd
import numpy as np
from torchvision.utils import save_image

class Config:
    def __init__(self):
        self.num_classes = 12
        self.device = torch.device('cuda')


if __name__ == '__main__':
    config = Config()
    # 导入测试的数据集
    Load_test = datasets.LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
    loader_test = data.DataLoader(Load_test,batch_size=2,shuffle=True)
    # 导入已经训练好的FCN模型
    net = Model.FCN(config.num_classes)
    net.load_state_dict(torch.load('param.params'))
    net.to(config.device)
    # 导入每一类所对应的颜色
    color_lst = []
    pd_label_color = pd.read_csv(cfg.class_dict_path,sep=',',engine='python')
    for i in range(len(pd_label_color.index)):
        tmp = pd_label_color.iloc[i]
        color_lst.append([tmp['r'],tmp['g'],tmp['b']])
    color_lst = torch.tensor(color_lst)
    # 进行前向传播得到预测的类别
    for idx,sample in enumerate(loader_test):
        image,label = sample['img'].to(config.device), sample['label'].to(config.device)
        net_out = net(image) # [batch,num_class,h,w]
        pred_label = torch.argmax(net_out,dim=1).cpu() # [batch,h,w]
        # 将预测的类别图片转换为颜色图片(得到了像素图)
        pred = color_lst[pred_label] # [batch,h,w,3]
        fact = color_lst[label] # [batch,h,w,3]
        save_tensor = torch.cat([pred,fact],dim=0).type(torch.float32)
        save_tensor = save_tensor.permute(0,3,1,2) # [batch,3,h,w]
        save_image(save_tensor,f'test_images/{idx}.png',nrow=2,normalize=True) # 第一行是预测的像素点，第二行是真实的像素点

