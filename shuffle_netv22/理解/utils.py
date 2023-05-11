from torch import nn
import torch
import torchvision
import torchvision.transforms as T
from torch.utils import data

class AverageMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.sum = 0
        self.avg = 0

    def update(self,value, num=0):
        self.cnt += num
        self.sum += value * num
        self.avg = self.sum / self.cnt


def get_training_dataloader(mean, std, batch_size=16, shuffle=True):
    train_transformer = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),
                                   T.RandomRotation(15),T.ToTensor(),T.Normalize(mean=mean,std=std)])
    train_dataset = torchvision.datasets.CIFAR100(root='D:\\python\\pytorch作业\\all_data\\cifar100',
                                                  train=True,download=False,transform=train_transformer)
    train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    return train_loader

def get_testing_dataloader(mean,std,batch_size=16,shuffle=False):
    test_transformer = T.Compose([T.ToTensor(),T.Normalize(mean=mean,std=std)])
    test_dataset = torchvision.datasets.CIFAR100(root='D:\\python\\pytorch作业\\all_data\\cifar100',
                                                  train=False,download=False,transform=test_transformer)
    test_loader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)
    return test_loader
