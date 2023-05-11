# 此时需要得到一张图片经过FCN网络后对每一个像素点的预测情况，并且给出最终得到的图片和标签图片的交占比，以及处理每张图片所需要的时间
import numpy as np
import torch
import time
import Models
import datasets
import cfg
from torch.utils import data
import pandas as pd
import main
from torchvision.utils import save_image

if __name__ == '__main__':
    # 导入测试的数据集
    test_dataset = datasets.LoadDataset([cfg.TEST_ROOT,cfg.TEST_LABEL],crop_size=cfg.crop_size)
    test_loader = data.DataLoader(test_dataset,batch_size=cfg.test_batch_size,shuffle=True)
    # 导入FCN模型
    net = Models.FCN(cfg.num_classes).to(cfg.device)
    load_net_path = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\理解\\param(备份).params'
    net.load_state_dict(torch.load(load_net_path))
    # 得到占空比以及预测点的准确率
    running_metrics_val = main.runningScore(cfg.num_classes)
    # 将标签输出变为三维的矩阵
    color_lst = []
    read_data = pd.read_csv(cfg.class_dict_path,sep=',',engine='python')
    for i in range(len(read_data.index)):
        tmp = read_data.iloc[i]
        color_lst.append([tmp['r'],tmp['g'],tmp['b']])
    color = np.array(color_lst)
   # 开始进行测试
    running_metrics_val.reset()
    start_time= time.time()
    for index,(x,y) in enumerate(test_loader):
        x,y = x.to(cfg.device), y.to(cfg.device)
        y_hat = net(x) # [batch,num_class,h,w]
        pred_label = torch.argmax(y_hat,dim=1).cpu().numpy() # [batch,h,w]
        true_label = y.cpu().numpy() # [batch,h,w]
        running_metrics_val.update(true_label,pred_label)
        pred_image = color[pred_label] # [batch,h,w,3]
        true_image = color[true_label] # [batch,h,w,3]
        image = torch.from_numpy(np.concatenate([true_image,pred_image],axis=0)).type(torch.float32)
        image = image.permute(0,3,1,2)
        save_image(image,f'out/{index}.png',nrow=2,normalize=True)
    mean_using_time = (time.time() - start_time) / len(test_dataset)
    acc, mean_iou = running_metrics_val.get_score()
    print(f'使用时间:{mean_using_time:1.3f},预测的所有像素的准确率:{acc:1.3f},测试图片的平均IOU:{mean_iou:1.3f}')


