import torch
import Models
import cfg
import datasets
from torch.utils import data
from torch import nn
import numpy as np

class AddMachine(object):
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

class runningScore(object): # 用于计算所有像素点预测正确的准确率以及mean_IOU
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.confuse_matrix = np.zeros((num_classes,num_classes)) # 初始的困惑矩阵

    def reset(self): # 每一轮迭代开始需要将困惑矩阵进行初始化
        self.confuse_matrix = np.zeros((self.num_classes,self.num_classes))

    def update(self,true_label,pred_label): # 每次模型参数更新后需要对困惑矩阵进行更新
        for t_l,p_l in zip(true_label,pred_label):
            self.confuse_matrix += self.get_confuse(t_l.reshape(-1),p_l.reshape(-1))

    def get_confuse(self,true_label,pred_label): # true_label = (w * h,) pred_label = (w * h,)
        mask = (true_label >= 0) & (true_label < self.num_classes)
        hist = np.bincount(self.num_classes * true_label[mask].astype(int) + pred_label[mask],
                           minlength=self.num_classes ** 2).reshape(self.num_classes,self.num_classes)
        return hist

    def get_score(self): # 平均交占比和预测的像素的准确率
        acc = np.diag(self.confuse_matrix).sum() / self.confuse_matrix.sum()
        intersection = np.diag(self.confuse_matrix) # 每个类别的交集
        union = self.confuse_matrix.sum(axis=0) + self.confuse_matrix.sum(axis=1) - intersection
        iu = intersection / union
        mean_iu = np.nanmean(iu)
        return acc,mean_iu

def train(train_loader,net):
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    running_metrics_val = runningScore(num_classes=cfg.num_classes)
    for i in range(cfg.num_epochs):
        metric = AddMachine(2) # 用来计算每次迭代的损失
        running_metrics_val.reset()
        for x,y in train_loader: # x=[batch,3,h,w],y=[batch,h,w]
            x,y = x.to(cfg.device),y.to(cfg.device)
            y_hat = net(x) # [batch,num_classes,h,w]
            # 用于求解平均占空比
            pred_label = torch.argmax(y_hat,dim=1).cpu().numpy() # [batch,h,w]
            true_label = y.cpu().numpy()

            y_hat_loss = y_hat.reshape(y_hat.shape[0],y_hat.shape[1],-1) # [batch,num_classes,h*w]
            y_loss = y.reshape(y.shape[0],-1)
            l = loss(y_hat_loss,y_loss)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y_loss.numel(), y_loss.numel())
            running_metrics_val.update(true_label,pred_label)
        lr_scheduler.step()
        acc,mean_iu = running_metrics_val.get_score()
        torch.save(net.state_dict(),'param.params')
        print(f'epoch:{i+1} loss:{metric[0] / metric[1]:1.3f} acc:{acc:1.3f} mean_iou:{mean_iu:1.3f}')


if __name__ == '__main__':
    # 导入整个数据集
    train_dataset = datasets.LoadDataset(file_path=[cfg.TRAIN_ROOT,cfg.TRAIN_LABEL],crop_size=cfg.crop_size)
    train_loader = data.DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True)
    # 导入整个模型
    net = Models.FCN(cfg.num_classes)
    net.to(cfg.device)
    # 开始进行训练
    train(train_loader, net)
