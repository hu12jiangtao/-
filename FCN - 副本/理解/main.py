# FCN的训练技巧
# 1.加载预训练模型 2.采用参数的衰减迭代 3.卷积核采用线性插值的方法进行赋初值 4.训练较多的epoch 5.pool3之前的输出不需要进行特征融合
# 数据集的类别是否是平衡的对实验不存在影响，模型会自动调节(背景的像素点的占比为75%对实验没有影响)
# 整一个过程:在训练集中的每一个样本存在一张图片以及其对应的像素点的图片，
#          将一张图片输入模型后得到每个像素点的预测[batch,num_class,h,w],与像素图片[batch,1,h,w]进行交叉熵预测，用损失值训练网络

# net = nn.Linear(4,3)
# opt = torch.optim.Adam(net.parameters(),lr=0.001)
# # 利用余弦的学习率衰减T_max代表1/4的余弦周期的迭代次数，当T_max为迭代次数时就是一个学习率衰减的优化算法
# sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2,eta_min=0,last_epoch=-1)

import torch
from torch import nn
import torchvision
import cfg
import datasets
from torch.utils import data
import Model
import numpy as np

class Config(object):
    def __init__(self):
        self.num_classes = 12
        self.device = torch.device('cuda')


class runningScore(object):
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes,num_classes))

    def reset(self): # 每一轮迭代后混淆矩阵都会重新进行赋值
        self.confusion_matrix = np.zeros((self.num_classes,self.num_classes))

    def update(self,label_true,label_pred): # 用于更新混淆矩阵
        # label_true.shape=label_pred.shape=[batch,h,w]
        for l_t,l_p in zip(label_true,label_pred):
            self.confusion_matrix += self._fast_hist(l_t.reshape(-1), l_p.reshape(-1), self.num_classes)

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        class_index = n_class * label_true[mask].astype(int) + label_pred[mask]
        # 此时将实际的第4类像素误判成第3类像素对应的类别索引应当为num_class * 4 + 3
        hist = np.bincount(class_index,minlength=n_class * n_class).reshape(n_class,n_class)
        return hist

    def get_scores(self): # 用于计算正确点的个数，以及平均交并比
        right_pred = np.diag(self.confusion_matrix).sum() # 所有的类别预测正确的像素点的个数(每个类别交集的像素点的总和)
        sum_num = self.confusion_matrix.sum() # 所有的像素点的个数
        pred_accuracy = right_pred / sum_num # 整个图片所有类别像素点预测正确的个数

        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) - intersection # 每个类别并集集的像素点的总和
        iu = intersection / union
        mean_iou = np.nanmean(iu)
        return pred_accuracy,mean_iou

class AddMachine:
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def train(net,train_loader,config):
    opt = torch.optim.Adam(net.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=50, gamma=0.5)
    loss = nn.CrossEntropyLoss()
    running_metrics_val = runningScore(config.num_classes)
    for i in range(cfg.EPOCH_NUMBER):
        net.train()
        metric = AddMachine(2)
        running_metrics_val.reset()
        for sample in train_loader:
            image,label_image = sample['img'].to(config.device), sample['label'].to(config.device)
            # image=[batch,3,h,w],label_image=[batch,h,w]
            y_hat = net(image) # [batch,num_class,h,w]
            pred_label = torch.argmax(y_hat,dim=1).cpu().numpy() # [batch,h,w]
            y_hat = y_hat.reshape(y_hat.shape[0],y_hat.shape[1],-1) # [batch,num_class,h*w]
            y = label_image.reshape(label_image.shape[0],-1) # [batch,h*w]
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), y.numel())

            true_label = label_image.cpu().numpy()
            running_metrics_val.update(true_label,pred_label) # 对混淆矩阵进行更新
        pred_accuracy,mean_iou = running_metrics_val.get_scores()
        print(f'epoch:{i+1} loss:{metric[0] / metric[1]:1.3f} pred_accuracy:{pred_accuracy:1.3f} mean_iou:{mean_iou:1.3f}')
        scheduler.step()
        torch.save(net.state_dict(),'param.params')





if __name__ == '__main__':
    config = Config()
    # 此时没有利用验证集进行验证，只有经过一定次数迭代的MIU和所有像素预测正确的准确率
    Load_train = datasets.LoadDataset([cfg.TRAIN_ROOT,cfg.TRAIN_LABEL],cfg.crop_size)
    train_loader = data.DataLoader(Load_train,batch_size=cfg.BATCH_SIZE,shuffle=True)
    # 导入FCN的模型
    fcn = Model.FCN(num_classes=12)
    fcn = fcn.to(config.device) # out = [batch,num_class,h,w]
    # 开始进行训练
    train(fcn, train_loader, config) # 最终得到的结果为损失0.051，所有像素点预测正确的概率为0.98，iou的指标为0.91


