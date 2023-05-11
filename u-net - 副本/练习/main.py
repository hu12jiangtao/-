import random

import torch
import config
import Model
import my_datasets
import transforms
from torch.utils import data
import train_and_eval
import numpy as np
import random
if __name__ == '__main__':
    # 固定随机种子
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    # 创建u-net模型
    model = Model.Unet(config.in_channel,config.num_classes,config.base_channel).to(config.device)
    # 需要对数据集进行导入(对于标签图片和输入模型的彩色图片都进行随机的数据增强)
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    train_dataset = my_datasets.DriveDataset(config.data_path,train=True,
                                             transforms=transforms.get_transform(train=True,mean=mean,std=std))
    test_dataset = my_datasets.DriveDataset(config.data_path,train=False,
                                            transforms=transforms.get_transform(train=False,mean=mean,std=std))
    # 此时由于train_dataset、test_dataset输出的宽高相同，因此collate_fn无用
    train_loader = data.DataLoader(train_dataset,batch_size=config.train_batch_size,
                                   shuffle=True,collate_fn=my_datasets.collate_fn)
    test_loader = data.DataLoader(test_dataset,batch_size=config.test_batch_size,
                                  shuffle=False,collate_fn=my_datasets.collate_fn)
    # 创建优化器和学习率的衰减器
    opt = torch.optim.SGD(model.parameters(),lr=config.lr,momentum=config.momentum,weight_decay=config.weight_decay)
    lr_scheduler = train_and_eval.create_lr_scheduler(opt,len(train_loader),config.epochs,warmup=True) # 每一次迭代学习率都发生变化
    # 开始进行么一轮迭代的训练和验证
    for epoch in range(config.epochs):
        # 需要返回当前的损失和学习率
        train_loss,lr = train_and_eval.train_one_epoch(model, opt,train_loader, config.device, config.num_classes, lr_scheduler)
        # 利用验证的数据集进行验证
        cumulative_dice,acc_global, acc_cls, iou, mean_iou = \
            train_and_eval.evaluate(model, test_loader, config.device, config.num_classes)
        # 参数的显示
        print(f'epoch:{epoch+1} lr:{lr:1.5f} train_loss:{train_loss:1.3f} acc_global:{acc_global:1.3f} '
              f'acc_cls:{[round(i,3) for i in acc_cls]} iou:{[round(i,3) for i in iou]} '
              f'mean_iou:{mean_iou:1.3f} dice:{cumulative_dice:1.3f}')



