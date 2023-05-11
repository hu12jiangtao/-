import torch
import UNET
import config
import torchvision
from torch.utils import data
import my_datasets
import train_and_eval
import transforms
from PIL import Image
import numpy as np

# 于FCN之间的区别:1.FCN模型中使用的是相加，而u-net中使用的是concat 2.u-net中额外加入了Dice-Loss
# 3.在其中加入了focal_loss(在此实验中验证影响较小)，多用于样本极度不均衡的情况，样本均衡时对模型准确率的提高影响不大

# 为什么u-net可以使用较少的数据集训练出较好的模型:1.使用了随机的数据增强(在FCN中没有使用随机增强)，
# 此时由于是全卷积的模型，因此batch内部的宽高相同即可，不需要保证batch之间的宽高也是相同的，增加了随机性

# u-net中的overlap操作，此代码中并没有体现，其目标是解决输入的图片较大，显存不够的问题
# 原理是将一张大图片分为多个patch，每个patch之间overlap(每个patch和四周的patch之间有部分是重叠的)

if __name__ == '__main__':
    # 导入训练时的数据集
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # 导入的数据集
    train_dataset = my_datasets.DriveDataset(config.root,
                                 train=True,
                                 transforms=transforms.get_transform(train=True, mean=mean, std=std))

    val_dataset = my_datasets.DriveDataset(config.root,
                               train=False,
                               transforms=transforms.get_transform(train=False, mean=mean, std=std))

    # collate_fn确保了每个批量中的所有样本的形状是相同的
    train_loader = data.DataLoader(train_dataset,batch_size=config.train_batch_size,shuffle=True,
                                   collate_fn=my_datasets.collate_fn) # 宽高为480，480
    val_loader = data.DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                                 collate_fn=my_datasets.collate_fn) #
    # val_loader中的图片原先高宽为584，564，经过resize后的宽高为496，480，宽高都可以被16整除，因此输入网络后没有问题

    # 导入训练的模型
    net = UNET.Unet(in_channel=3,num_class=2,base_channel=64).to(config.device) # 分为了背景和神经
    # 开始进行训练
    opt = torch.optim.SGD(net.parameters(),lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    # 学习率衰减器是每一次迭代都会改变其学习率
    scheduler = train_and_eval.create_lr_scheduler(opt,len(train_loader),config.epochs,warmup=True)
    # 开始进行学习
    for i in range(config.epochs):
        train_loss, lr = train_and_eval.train_one_epoch(net, opt, train_loader,
                                                        config.device, config.num_classes, scheduler)
        acc_global, acc_cls, iou, mean_iou, cumulative_dice = train_and_eval.evaluate\
            (net, val_loader, config.device, config.num_classes)
        print(f'epoch:{i+1} train_loss:{train_loss:13f} lr:{lr:1.8f} acc_global:{acc_global:1.3f} '
              f'acc_cls:{[round(k,3) for k in acc_cls]} iou:{[round(k,3) for k in iou]} mean_iou:{mean_iou:1.3f} '
              f'cumulative_dice:{cumulative_dice:1.3f}')





