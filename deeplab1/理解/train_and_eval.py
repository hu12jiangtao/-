from torch import nn
import torch
import numpy as np

def create_lr_scheduler(optimizer,num_step,epochs,warmup=True,warmup_epochs=1,warmup_factor=1e-3):
    # 使用deeplabv2中的ploy学习率衰减函数
    # warmup代表在使用ploy学习率衰减函数的前面是否使用其他的学习率衰减策略
    assert num_step > 0 and epochs > 0
    if warmup is not True:
        warmup_epochs = 0
    def f(x): # x为样本迭代次数，而不是批量迭代次数
        if warmup is True and x < warmup_epochs * num_step:
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)

def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler): # 此时我需要返回训练损失函数以及学习率
    model.train()
    epoch_loss = 0
    for idx,(image,mask) in enumerate(data_loader):
        image,mask = image.to(device),mask.to(device) # mask=[batch,h,w]
        y_hat = model(image) # [b,num_classes,h,w]
        loss = nn.CrossEntropyLoss(ignore_index=255)
        l = loss(y_hat.reshape(y_hat.shape[0],y_hat.shape[1],-1),mask.reshape(mask.shape[0],-1))
        epoch_loss += l.item()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        lr_scheduler.step()
    torch.save(model.state_dict(),'params.param')
    return epoch_loss / len(data_loader), optimizer.param_groups[0]['lr']

class GetScore(object):
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.confuse = np.zeros(shape=(num_classes,num_classes))

    def reset(self):
        return np.zeros(shape=(self.num_classes,self.num_classes))

    def update(self,pred, label):
        # pred = label = [batch,h,w]
        batch = pred.shape[0]
        for i in range(batch):
            self.confuse += self.update_batch(pred[i].reshape(-1),label[i].reshape(-1))

    def update_batch(self,pred,label):
        # pred = label = [h * w]
        mask = (label >= 0) & (label < self.num_classes)
        matrix = np.bincount(label[mask] * self.num_classes + pred[mask], minlength=self.num_classes**2)\
            .reshape(self.num_classes,self.num_classes)
        return matrix

    def get_score(self):
        global_acc = np.diag(self.confuse).sum() / self.confuse.sum()
        class_acc = np.diag(self.confuse) / self.confuse.sum(1)
        inter = np.diag(self.confuse)
        union = self.confuse.sum(1) + self.confuse.sum(0) - inter
        iou = inter / union
        mean_iou = np.mean(iou)
        return global_acc,class_acc,iou,mean_iou

def evaluate(model, data_loader, device, num_classes):
    # 主要就是mean_iou的指标
    model.eval()
    metric = GetScore(num_classes)
    for idx, (image,mask) in enumerate(data_loader):
        image, mask = image.to(device), mask.to(device)
        y_hat = model(image)
        confuse_pred = torch.argmax(y_hat,dim=1).cpu().numpy()
        confuse_target = mask.cpu().numpy()
        metric.update(confuse_pred,confuse_target)
    global_acc, class_acc, iou, mean_iou = metric.get_score()
    return global_acc, class_acc, iou, mean_iou

