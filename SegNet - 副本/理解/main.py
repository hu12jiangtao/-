# seg-net的训练过程和FCN网络相类似，都是只使用了交叉熵损失函数，在seg-net中加入了每个类别所对应的权重
import torch
from torch import nn
import ComVid_config
import model
import ComVid_dataset
from torch.utils import data
import numpy as np

class GetScore(object):
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.confuse = np.zeros(shape=(num_classes,num_classes))

    def reset(self):
        self.confuse = np.zeros(shape=(self.num_classes,self.num_classes))

    def update(self,pred,label): # pred = [batch,h,w],label = [batch,h,w]
        batch_size = pred.shape[0]
        for i in range(batch_size):
            self.confuse += self.update_batch(pred[i].reshape(-1),label[i].reshape(-1))

    def update_batch(self,pred,label): # pred=label=[h * w, ]
        mask = (label >= 0) & (label < self.num_classes)
        matrix = np.bincount(label[mask] * self.num_classes + pred[mask],minlength=self.num_classes ** 2)\
            .reshape(self.num_classes,self.num_classes)
        return matrix

    def get_score(self):
        global_acc = np.diag(self.confuse).sum() / self.confuse.sum()
        class_acc = np.diag(self.confuse) / self.confuse.sum(axis=1)
        inter = np.diag(self.confuse)
        union = self.confuse.sum(axis=1) + self.confuse.sum(axis=0) - inter
        iou = inter / union
        mean_iou = np.mean(iou)
        return global_acc,class_acc,iou,mean_iou

class AddMachine(object):
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def train(train_loader, test_loader, net, loss, config):
    opt = torch.optim.Adam(net.parameters(),lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=50,gamma=0.5) # 每50轮下降一半的学习率
    for epoch in range(config.num_epochs):
        metric = AddMachine(2)
        net.train()
        for x,y in train_loader: # x=[batch,3,h,w], y=[batch,h,w]
            x, y = x.to(config.device),y.to(config.device)
            y_hat,_ = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), y.numel())
        lr_scheduler.step()
        torch.save(net.state_dict(),'params.param')
        # 开始进行验证
        net.eval()
        with torch.no_grad():
            running_metrics_val = GetScore(config.num_classes)
            for x, y in test_loader:
                x,y = x.to(config.device),y.to(config.device)
                y_hat,_ = net(x) # [batch,num_classes,h,w]
                pred = torch.argmax(y_hat,dim=1)
                confuse_pred = pred.cpu().numpy() # [batch, h, w]
                confuse_label = y.cpu().numpy() # [batch, h, w]
                running_metrics_val.update(confuse_pred,confuse_label)
            global_acc, class_acc, iou, mean_iou = running_metrics_val.get_score()
            print(f'epoch:{epoch+1} train_loss:{metric[0]/metric[1]:1.4f} '
                  f'global_acc:{global_acc:1.4f} '
                  f'class_acc:{[round(i,3) for i in class_acc]} '
                  f'iou:{[round(i,3) for i in iou]} '
                  f'mean_iou:{mean_iou:1.4f}')






if __name__ == "__main__":
    # 导入模型
    model = model.SegNet(ComVid_config.in_channel,ComVid_config.out_channel).to(ComVid_config.device)
    # 导入数据集
    train_dataset = ComVid_dataset.LoadDataset([ComVid_config.train_image_path,
                                                ComVid_config.train_label_path], ComVid_config.crop_size)
    test_dataset = ComVid_dataset.LoadDataset([ComVid_config.test_image_path,
                                               ComVid_config.test_label_path], ComVid_config.crop_size)
    train_loader = data.DataLoader(train_dataset,batch_size=4,shuffle=True)
    test_loader = data.DataLoader(train_dataset,batch_size=1,shuffle=False)
    # 设置每一类的权重(根据训练的图片)
    labels = [i[1] for i in train_dataset]
    part_prob = ComVid_dataset.compute_class_probability(labels,ComVid_config.num_classes)
    class_weights = (1.0 / part_prob).type(torch.float32).to(ComVid_config.device)
    loss = nn.CrossEntropyLoss(weight=class_weights)
    # 此时每轮应当为训练集训练，测试集进行测试
    train(train_loader, test_loader, model, loss, ComVid_config)



