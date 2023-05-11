import torch
from torch import nn
import Losses
import numpy as np
from torch.nn import functional as F

def create_lr_scheduler(optimizer,num_step: int,epochs: int,warmup=True,warmup_epochs=1,warmup_factor=1e-3):
    def f(x):
        if warmup is True and x <= (num_step * warmup_epochs):
            alpha = float(x) / (num_step * warmup_epochs)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)


def train_one_epoch(model, optimizer, data_loader, device, num_classes,lr_scheduler):
    model.train()
    epoch_loss = 0
    for image,target in data_loader:
        image, target = image.to(device), target.to(device)
        outputs = model(image) # [batch,num_class,h,w]
        loss = Losses.sum_loss(outputs,target,255,num_classes,device)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    torch.save(model.state_dict(),'params.param')
    return epoch_loss / len(data_loader), optimizer.param_groups[0]['lr'] # 返回一轮中损失均值和最终的学习率


class ConfusionMatrix(object):  # 类中使用的是numpy
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros(shape=(num_classes,num_classes))

    def update(self,pred,target): # pred.shape=target.shape=[batch,h,w]
        for i in range(pred.shape[0]):
            self.mat += self.each_sample_update(pred[i].reshape(-1), target[i].reshape(-1), self.num_classes)

    def each_sample_update(self,pred,target,num_classes):  # pred.shape = target.shape = [h * w, ]
        # 此时输入的target中包含元素值有255，0，1三个元素(此时需要排除掉元素值为255的情况),此时就需要用mask
        mask = (target >= 0) & (target < num_classes) # 也可以是mask = (target != 255)
        mat = np.bincount(num_classes * target[mask] + pred[mask], minlength= self.num_classes ** 2).\
            reshape(self.num_classes,self.num_classes)
        return mat

    def reset(self):
        self.mat = np.zeros(shape=(self.num_classes,self.num_classes))

    def get_score(self):
        # 计算所有类别的平均准确率
        acc_global = np.diag(self.mat).sum() / self.mat.sum()
        # 计算每个类别的像素预测的准确率
        acc_cls = np.diag(self.mat) / self.mat.sum(1) # 输出一个列表
        # 计算每个类别的IOU
        inter = np.diag(self.mat) # 分子部分
        union = self.mat.sum(0) + self.mat.sum(1) - inter
        iou = inter / union
        # 计算平均的IOU
        mean_iou = iou.mean()
        return acc_global, acc_cls, iou, mean_iou

class DiceCoefficient(object): # 类中使用的是torch
    def __init__(self,num_classes, ignore_index):
        self.cumulative_dice = 0
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = 0

    def update(self,pred,label):
        self.count += 1
        # 获得Dice系数
        dice_target = Losses.build_target(label,self.ignore_index,self.num_classes)
        dice_pred = F.one_hot(pred,self.num_classes).permute(0,3,1,2)
        # 获得标签1(神经)的Dice系数, now_dice为这个batch中的每个样本的每个类别的Dice系数的均值
        now_dice = Losses.multiclass_dice_coeff(dice_pred[:,1:],dice_target[:,1:],self.ignore_index)
        self.cumulative_dice += now_dice

    def get_score(self):
        return self.cumulative_dice / self.count # 所有样本的Dice系数均值


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    dice = DiceCoefficient(num_classes=num_classes, ignore_index=255)
    for x,y in data_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x) # [batch,num_class,h,w]
        pred_dice = torch.argmax(outputs,dim=1).cpu()
        y_dice = y.cpu()
        pred_confuse = pred_dice.numpy() # [batch,h,w]
        y_confuse = y_dice.numpy() # [batch,h,w]
        confmat.update(pred_confuse,y_confuse)
        dice.update(pred_dice,y_dice)
    acc_global, acc_cls, iou, mean_iou = confmat.get_score()
    cumulative_dice = dice.get_score()
    return acc_global, acc_cls, iou, mean_iou, cumulative_dice


if __name__ == '__main__':
    torch.manual_seed(1)
    net = nn.Linear(4,3)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    scheduler = create_lr_scheduler(opt, 10, 200, warmup = True)




