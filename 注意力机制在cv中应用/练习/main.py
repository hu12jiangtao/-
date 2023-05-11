import Model
from torch import nn
import torchvision
import torch
from torch.utils import data

# 作者在原文中进行了实验
# 1.在设计注意力机制模块的时候使用全局平均池化得到的准确率比全局最大池化的效果好
# 2.在设计注意力机制模块时的全连接层的中间层的神经元个数为输入的1/16倍(1/2，1/4，1/8，1/16的准确率非常近似，但是取1/16需要训练的权重参数的训练参数值较小)
# 3.在设计注意力机制模块的全连接层最后使用的激活函数采用sigmoid的效果比tanh和relu要好
# 4.在resnet网络中的每个stage位置加入SeNet的准确率都不不加入时高
# 5.加入通道注意力机制一般是放在残差分支后，聚合之前

class AddMachine(object):
    def __init__(self,n):
        self.data = [0.] * n

    def add(self,*args):
        self.data = [float(i)+j for i,j in zip(args,self.data)]

    def __getitem__(self, item):
        return self.data[item]

def evaluate(y_hat,y):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(data_loader,net,device):
    net.eval()
    metric = AddMachine(2)
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        y_hat = net(x)
        metric.add(evaluate(y_hat,y),y.numel())
    return metric[0] / metric[1]

def train(train_loader,test_loader,net,device):
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4, betas=(0.9,0.999))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=100)
    loss = nn.CrossEntropyLoss()
    for i in range(100):
        net.train()
        metric = AddMachine(3)
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), evaluate(y_hat,y), y.numel())
        lr_scheduler.step()
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(test_loader,net,device)
        print(f'epoch:{i+1} train_loss:{train_loss:1.3f} train_acc:{train_acc:1.3f} test_acc:{test_acc:1.3f}')






if __name__ =='__main__':
    device = torch.device('cuda')
    # 导入数据集
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=0.5,std=0.5)])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(mean=0.5,std=0.5)])
    train_dataset = torchvision.datasets.CIFAR10(root='D:\\python\\pytorch作业\\all_data\\cifar10\\data',train=True,
                                                 transform=train_transform,download=False)
    test_dataset = torchvision.datasets.CIFAR10(root='D:\\python\\pytorch作业\\all_data\\cifar10\\data',train=False,
                                                 transform=test_transform,download=False)
    train_loader = data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = data.DataLoader(test_dataset,batch_size=64,shuffle=False)
    # 导入模型
    # 加入通道注意力机制后的验证准确率为0.751
    # net = Model.CifarSEResNet(Model.CifarSEBasicBlock, in_channel=3,basic_channel=16).to(device)
    # 加入通道注意力机制后的验证准确率为0.761(没有加注意力机制效果好)
    net = Model.CifarSEesNet(Model.CifarSEBasicBlock, in_channel=3, basic_channel=16,is_attention=False).to(device)
    # 开始训练
    train(train_loader, test_loader, net, device)
