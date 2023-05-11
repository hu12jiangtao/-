import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as T
from torch.utils import data

class WarmLR(_LRScheduler):
    # 以下是warm_up的作用
    # 开始的几个epoch，逐步增大学习率，如下图所示，使得模型逐渐趋于稳定，
    # 等模型相对稳定后再选择预先设置的基础学习率进行训练，使得模型收敛速度变得更快，模型效果更佳
    # warm_up一般是每个iter进行一次学习率的更改，此时的self.last_epoch代表的是当前的iter，self.base_lrs为长度为1，元素值为lr的列表
    def __init__(self,opt, total_iter, last_epoch=-1):
        self.total_iter = total_iter
        super(WarmLR, self).__init__(opt,last_epoch)
    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iter + 1e-8) for base_lr in self.base_lrs]

def get_training_dataloader(mean, std, batch_size=16, shuffle=True):
    train_transform = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.RandomRotation(15),
                                 T.ToTensor(),T.Normalize(mean=mean,std=std)])
    train_dataset = torchvision.datasets.CIFAR100(root='D:\\python\\pytorch作业\\all_data\\cifar100',
                                                  transform=train_transform,download=False,train=True)
    train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    return train_loader

def get_test_dataloader(mean, std, batch_size=16, shuffle=True):
    test_transform = T.Compose([T.ToTensor(),T.Normalize(mean=mean,std=std)])
    test_dataset = torchvision.datasets.CIFAR100(root='D:\\python\\pytorch作业\\all_data\\cifar100',
                                                  transform=test_transform,download=False,train=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return test_loader


