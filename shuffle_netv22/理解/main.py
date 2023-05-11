import model
import utils
import torch
from torch import nn

class Config:
    def __init__(self):
        self.device = torch.device('cuda')
        self.train_batch_size = 48
        self.valid_batch_size = 32
        self.mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        self.ratio = 1
        self.init_lr = 0.1
        self.MILESTONES = [60, 120, 160]
        self.epochs = 200

def accuracy(outputs,target, topk=(1,)):
    result = []
    batch_size = outputs.shape[0]
    max_num = max(topk) # 5
    _, idx = torch.topk(outputs, max_num, dim=1) # [batch, 5]
    idx = idx.permute(1,0) # [5, batch]
    cmp = torch.eq(idx, target.reshape(1,-1).expand_as(idx))
    for i in topk:
        now_cmp = torch.sum(cmp[:i].reshape(-1).float(),dim=0)
        acc = 100 * now_cmp / batch_size
        result.append(acc)
    return result

def evaluate(data_loader,net,config):
    loss_metric = utils.AverageMetric()
    top1_metric = utils.AverageMetric()
    top5_metric = utils.AverageMetric()
    loss = nn.CrossEntropyLoss()
    net.eval()
    for x, y in data_loader:
        batch = x.shape[0]
        x, y = x.to(config.device), y.to(config.device)
        y_hat = net(x)
        l = loss(y_hat, y)
        result = accuracy(y_hat, y, topk=(1, 5))
        loss_metric.update(l, batch)
        top1_metric.update(result[0], batch)
        top5_metric.update(result[1], batch)
    return loss_metric.avg, top1_metric.avg, top5_metric.avg

def train(train_loader,test_loader,net,opt,lr_scheduler,config):
    loss = nn.CrossEntropyLoss()
    loss_metric = utils.AverageMetric()
    top1_metric = utils.AverageMetric()
    top5_metric = utils.AverageMetric()
    acc = 0
    for epoch in range(config.epochs):
        net.train()
        # 训练
        for x,y in train_loader:
            batch = x.shape[0]
            x, y = x.to(config.device), y.to(config.device)
            y_hat = net(x)
            l = loss(y_hat, y)
            opt.zero_grad()
            l.backward()
            opt.step()
            result = accuracy(y_hat,y, topk=(1,5))
            loss_metric.update(l, batch)
            top1_metric.update(result[0], batch)
            top5_metric.update(result[1], batch)
        lr_scheduler.step()
        train_loss, train_top1, train_top5 = loss_metric.avg,top1_metric.avg,top5_metric.avg
        # 验证
        with torch.no_grad():
            valid_loss, valid_top1, valid_top5 = evaluate(test_loader, net, config)
        print('lr:',round(opt.param_groups[0]['lr'],5),end=' ')
        print(f'epoch:{epoch+1}  train_loss:{train_loss:1.3f}  train_top1:{train_top1:1.3f}%  train_top5:{train_top5:1.3f}% '
              f'valid_loss:{valid_loss:1.3f}  valid_top1:{valid_top1:1.3f}%  valid_top5:{valid_top5:1.3f}%')
        # 模型的存储
        if valid_top1 > acc and (epoch + 1) > config.MILESTONES[2]:
            torch.save(net.state_dict(),'params.param')
            acc = valid_top1

if __name__ == '__main__':
    config = Config()
    # 导入训练集数据和验证集的数据
    train_loader = utils.get_training_dataloader(config.mean,config.std,config.train_batch_size)
    valid_loader = utils.get_testing_dataloader(config.mean,config.std,config.valid_batch_size)
    # 获取模型
    model = model.ShuffleNetV2(ratio=1).to(config.device)
    # 优化器
    opt = torch.optim.SGD(model.parameters(),lr=config.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, config.MILESTONES, gamma=0.2)
    # 进行训练
    train(train_loader, valid_loader, model, opt, lr_scheduler, config)
