# shuffle_net v1的创新点的原理就是减小在mobile_net v1中占据大部分的计算量的1x1卷积的计算量
# 使用的方法是将普通的1x1卷积替换成了分组的1x1卷积，其计算量降低了group倍，同时加入了残差连接
# 如果只是用分组的1x1卷积会导致每个组之间的信息不能进行交互，因此引入了通道重排的操作
import os

import torch
from torch import nn
import models
import utils

class Config:
    def __init__(self):
        self.mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.EPOCH = 200
        self.MILESTONES = [60, 120, 160]
        self.device = torch.device('cuda')
        self.train_batch_size = 128
        self.test_batch_size = 32
        self.lr = 0.1
        self.warm_epoch = 1

class AddMachine:
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [float(i) + j for i,j in zip(args,self.data)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(data_loader, net, loss, device):
    net.eval()
    metric = AddMachine(3)
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        y_hat = net(x)
        l = loss(y_hat,y)
        metric.add(l * y.numel(), accuracy(y_hat,y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_and_eval(train_loader,test_loader,net,config,loss,opt,warm_lr_scheduler,lr_scheduler):
    best_eval_acc = 0
    save_checkpoint = 'checkpoint'
    for epoch in range(1, config.EPOCH + 1):
        # 网络的训练
        net.train()
        metric = AddMachine(3)
        if epoch > config.warm_epoch:
            lr_scheduler.step()
        for x,y in train_loader:
            x,y = x.to(config.device),y.to(config.device)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), accuracy(y_hat,y), y.numel())
            if epoch <= config.warm_epoch:
                warm_lr_scheduler.step()
        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        # 网络的测试
        eval_loss,eval_acc = evaluate_accuracy(test_loader, net, loss, config.device)
        print(opt.param_groups[0]['lr'],end=' ')
        print(f'epoch:{epoch} train_loss:{train_loss:1.4f} train_acc:{train_acc:1.4f} eval_loss:{eval_loss:1.4f} '
              f'eval_acc:{eval_acc:1.4f}')
        # 对模型进行保存
        if eval_acc > best_eval_acc and epoch > config.MILESTONES[1]:
            best_eval_acc = eval_acc
            save_path = os.path.join(save_checkpoint,f'epoch_{epoch}.params')
            torch.save(net.state_dict(),save_path)

if __name__ == '__main__':
    config = Config()
    # 导入模型
    net = models.ShuffleNet([4,8,4]).to(config.device)
    param_num = sum([param.numel() for param in net.parameters()])
    print(f'模型训练参数个数:{param_num}')
    # 导入数据集
    train_loader = utils.get_training_dataloader(config.mean,config.std,config.train_batch_size,shuffle=True)
    test_loader = utils.get_test_dataloader(config.mean,config.std,config.test_batch_size,shuffle=False)
    # 进行训练
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(),lr=config.lr,momentum=0.9,weight_decay=5e-4)
    # 学习率衰减器
    iter_per_epoch = len(train_loader)
    warm_lr_scheduler = utils.WarmLR(opt, config.warm_epoch * iter_per_epoch) # 在预热的时候每个iter进行学习率衰减
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,config.MILESTONES,gamma=0.2)
    # 开始进行训练和预测
    train_and_eval(train_loader, test_loader, net, config, loss, opt, warm_lr_scheduler, lr_scheduler)




