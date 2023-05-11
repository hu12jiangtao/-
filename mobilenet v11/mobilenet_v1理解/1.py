import torch
from torch.nn import functional as F
from torch import nn
import datasets
import model
import config
from torch.utils import data
import pandas as pd
import os

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
    metric = AddMachine(2)
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        y_hat = net(x)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0] / metric[1]

if __name__ == '__main__':
    # 导入模型
    net = model.MobileNet(alpha=0.75).to(config.device)
    net.load_state_dict(torch.load('params.param'))
    # 导入数据集
    test_dataset = datasets.DogvsCatDataset(config.root_dir,train=False)
    test_loader = data.DataLoader(test_dataset,batch_size=25,shuffle=False)
    net.eval()  # 由于存在bn层因此一定会需要这句话的
    test_acc = evaluate_accuracy(test_loader, net, config.device)
    print(test_acc)


