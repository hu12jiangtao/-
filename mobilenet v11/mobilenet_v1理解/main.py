# mobile-net v1相较于普通的卷积网络的创新点有两个:(并且对其进行证明)
# 1.将正常的卷积转换为了深度可分离卷积，大大减少了参数量和计算量
# 2.提出里全局参数α(用于控制模型中所有通道数的同步衰减)和ρ(输入图片的分辨率Resize操作)进一步来降低网络的参数和计算量(ρ不减小参数量)
import os

import torch
from torch import nn
import datasets
import model
import config
from torch.utils import data
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

def train(train_loader,test_loader,net,max_epochs,device,start_epoch):
    opt = torch.optim.Adam(net.parameters(),lr=1e-4)
    loss = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(start_epoch, max_epochs + 1):
        net.train()
        metric = AddMachine(3)
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), accuracy(y_hat,y), y.numel())
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if epoch % 5 == 0:
            test_acc = evaluate_accuracy(test_loader,net,device)
            print(f'epoch:{epoch} train_loss:{train_loss:1.4f} train_acc:{train_acc:1.4f} test_acc:{test_acc:1.4f}')
        else:
            print(f'epoch:{epoch} train_loss:{train_loss:1.4f} train_acc:{train_acc:1.4f}')
        torch.save(net.state_dict(),'params.param')
        if epoch % 25 == 0:
            torch.save(net.state_dict(), f'params({epoch}).param')



if __name__ == '__main__':
    # 导入数据集
    train_dataset = datasets.DogvsCatDataset(config.root_dir,train=True,gamma=0.714)
    test_dataset = datasets.DogvsCatDataset(config.root_dir,train=False)
    train_loader = data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    test_loader = data.DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False) # 在验证的数据集上不存在标签
    # 导入模型
    net = model.MobileNet(alpha=0.75).to(config.device)
    print('当前网络模型参数:',net.get_param_num())
    if os.path.exists('params.param'):
        start_epoch = 61
        net.load_state_dict(torch.load('params.param'))
    else:
        start_epoch = 1
    # 开始进行训练
    train(train_loader, test_loader, net, config.max_epoch, config.device,start_epoch)
