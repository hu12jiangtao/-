# mobile_net-v2针对于mobile_net_v1做出的改进有两点
# 参照resnet网络的结构给出逆向的残差连接
# 对于之后降维的部分采用的是线性激活函数
import os

import torch
import torchvision
from torchvision import transforms as T
from torch.utils import data
import config
import model
from torch import nn
import numpy as np
import random

def get_train_loader(root, mean, std, batch_size=16, shuffle=True):
    train_transform = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),
                                 T.RandomRotation(15),T.ToTensor(),T.Normalize(mean=mean,std=std)])
    train_dataset = torchvision.datasets.CIFAR100(root,train=True,transform=train_transform,download=False)
    train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    return train_loader

def get_test_loader(root, mean, std, batch_size=16, shuffle=True):
    test_transform = T.Compose([T.ToTensor(),T.Normalize(mean=mean,std=std)])
    test_dataset = torchvision.datasets.CIFAR100(root,train=False,transform=test_transform,download=False)
    test_loader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)
    return test_loader

class AddMachine:
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [float(i)+j for i,j in zip(args,self.data)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(data_loader,net,device):
    net.eval()
    metric = AddMachine(3)
    loss = nn.CrossEntropyLoss()
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        y_hat = net(x)
        l = loss(y_hat,y)
        metric.add(l * y.numel(),accuracy(y_hat,y),y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(train_loader,test_loader,net,lr,max_epochs,device,start_epoch):
    opt = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[60, 120, 160],gamma=0.2)
    loss = nn.CrossEntropyLoss()
    best_test_acc = 0.
    for epoch in range(start_epoch, max_epochs + 1):
        net.train()
        metric = AddMachine(2)
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(accuracy(y_hat,y), y.numel())
        lr_scheduler.step()
        train_acc = metric[0] / metric[1]
        test_loss, test_acc = evaluate_accuracy(test_loader,net,device)
        print('lr:',opt.param_groups[0]['lr'])
        print(f'epoch:{epoch} train_acc:{train_acc:1.4f}'
              f' test_loss:{test_loss:1.4f} test_acc:{test_acc:1.4f}')

        torch.save(net.state_dict(),'params.param')
        if test_acc > best_test_acc and epoch > 120:
            torch.save(net.state_dict(),os.path.join('checkpoints',f'epoch_{epoch}.params'))
            best_test_acc = test_acc



if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    # 导入数据集
    test_loader = get_test_loader(config.root, config.mean, config.std, batch_size=config.test_batch_size, shuffle=False)
    train_loader = get_train_loader(config.root, config.mean, config.std, batch_size=config.train_batch_size, shuffle=True)
    # 导入模型
    net = model.MobileNetV2().to(config.device) # 模型验证是相同的
    print('当前模型的训练参数:',sum([param.numel() for param in net.parameters()]))
    # 开始进行训练
    if os.path.exists('params.param'):
        net.load_state_dict(torch.load('params.param'))
        lr = 0.004
    else:
        lr = 0.1
    train(train_loader, test_loader, net, lr, max_epochs=200, device=config.device, start_epoch=1)







