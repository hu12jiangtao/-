import torch
from torch import nn
import numpy as np
import model_search
import utils
from torch.utils import data
import torchvision
import arch

# Darts的模型架构是利用相同的模块进行堆叠，因此此时的架构参数(每个操作的权重参数)是共享的，
# 因此此时的架构参数只有两种，一种是当前cell的stride=1，另一种是stride=2
class Config:
    def __init__(self):
        self.device = torch.device('cuda')
        self.init_ch = 16
        self.layers = 8
        self.data_root = 'data' # 'D:\\python\\pytorch作业\\all_data\\cifar10\\data'
        self.train_portion = 0.5 # 一半作为训练集,一半作为验证集
        self.batch_size = 50
        self.lr = 0.025
        self.momentum = 0.9
        self.wd = 3e-4
        self.epochs = 50
        self.lr_min = 0.001
        self.arch_lr = 3e-4
        self.arch_wd = 1e-3
        self.grad_clip = 5.

def train(train_loader, valid_loader, model, arch, criterion, optimizer, lr, config):
    valid_iter = iter(valid_loader)
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    for x_train, y_train in train_loader:
        model.train()
        batch_size = x_train.shape[0]

        x_train, y_train = x_train.to(config.device), y_train.to(config.device)

        x_valid, y_valid = next(valid_iter)
        x_valid, y_valid = x_valid.to(config.device), y_valid.to(config.device)
        # 对架构参数alpha进行更新
        arch.step(x_train, y_train, x_valid, y_valid, lr, optimizer)
        # 在训练集上对架构参数和操作参数都进行更新
        logits = model(x_train)
        loss = criterion(logits,y_train)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip) # 进行梯度的剪枝
        optimizer.step()
        # 计算top1 和 top5 的准确率
        result = utils.accuracy(logits,y_train,topk=(1,5))
        losses.update(loss.item(),batch_size)
        top1.update(result[0], batch_size)
        top5.update(result[1], batch_size)
    return losses.avg, top1.avg, top5.avg

def infer(valid_loader, model, criterion, config):
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    with torch.no_grad():
        for x, y in valid_loader:
            batch_size = x.shape[0]
            x, y = x.to(config.device), y.to(config.device)
            logits = model(x)
            loss = criterion(logits, y)
            result = utils.accuracy(logits, y, topk=(1,5))
            losses.update(loss.item(),batch_size)
            top1.update(result[0], batch_size)
            top5.update(result[1], batch_size)
    return losses.avg,top1.avg,top5.avg

if __name__ == '__main__':
    print(1)
    config = Config()
    # 用于卷积的加速运算
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(2)
    torch.manual_seed(2)
    # 导入模型
    criterion = nn.CrossEntropyLoss()
    model = model_search.Network(config.init_ch, 10, config.layers, criterion).to(config.device)
    print(f'模型参数数量:{sum([i.numel() for name, i in model.named_parameters() if "auxiliary" not in name])}')
    # 导入训练数据集和验证数据集
    train_transform,valid_transform = utils._data_transforms_cifar10()
    train_data = torchvision.datasets.CIFAR10(config.data_root,train=True,transform=train_transform,download=False)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(num_train * config.train_portion))
    train_loader = data.DataLoader(train_data,batch_size=config.batch_size,
                                   sampler=data.sampler.SubsetRandomSampler(indices[:split])) # 在前25000个样本中不重复的随机采样
    valid_loader = data.DataLoader(train_data,batch_size=config.batch_size,
                                   sampler=data.sampler.SubsetRandomSampler(indices[split:])) # 在后25000个样本中不重复的随机采样
    # 设计权重参数和模型架构参数的优化方式(此时在训练集上操作参数和架构参数都会更新，在验证集上只更新架构参数)
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.wd) # 训练集上更新的优化方式
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,float(config.epochs),eta_min=config.lr_min)
    arch = arch.Arch(model,config) # 验证机上架构参数的优化方式,对arch的step进行过验证，得到的与源代码相同
    # 开始进行训练
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        # 获取在cell中的所有节点所连接的路径序列，以及当前路径中所选择的操作
        genotype = model.genotype()
        print(genotype)
        # 对模型进行训练
        train_loss,top1_acc, top5_acc = train(train_loader, valid_loader, model, arch, criterion, optimizer, lr, config)
        print(f'epoch:{epoch+1} train_loss:{train_loss:1.4f} top1_acc:{top1_acc} top5_acc:{top5_acc}')
        # 对模型进行验证
        valid_loss, valid_top1_acc,valid_top5_acc = infer(valid_loader, model, criterion, config)
        print(f'valid_loss:{valid_loss:1.4f} valid_top1_acc:{valid_top1_acc} valid_top5_acc:{valid_top5_acc}')
        if (1 + epoch) % 10 == 0:
            torch.save(model.state_dict(),f'param{1+epoch}.params')
        torch.save(model.state_dict(),'param.params')















