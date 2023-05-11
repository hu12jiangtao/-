import torch
import torchvision
from torch.utils import data
from PIL import Image
from torch import nn

def load_cifir_data(path,train_batch_size,test_batch_size):
    # RandomCrop中的padding代表的含义为首先将整个图片进行padding之后在进行随机的裁剪
    train_trans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                  torchvision.transforms.RandomHorizontalFlip(0.5),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                   [0.2023, 0.1994, 0.2010])])
    test_trans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                  [0.2023, 0.1994, 0.2010])])
    train_data = torchvision.datasets.CIFAR10(path,train=True,download=False,transform=train_trans)
    test_data = torchvision.datasets.CIFAR10(path,train=False,download=False,transform=test_trans)
    train_loader = data.DataLoader(train_data,batch_size=train_batch_size,shuffle=True)
    test_loader = data.DataLoader(test_data,batch_size=test_batch_size,shuffle=False)
    return train_loader,test_loader

class add_machine(object):
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y,y_hat):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate(net,data_loader,device):
    metric = add_machine(2)
    net.eval()
    net.to(device)
    for X,y in data_loader:
        X,y = X.to(device),y.to(device)
        y_hat = net(X)
        metric.add(accuracy(y,y_hat),y.numel())
    return metric[0] / metric[1]

def train_epoch(epoch,net,train_loader,test_loader,loss,opt,config):
    net.train()
    metric = add_machine(3)
    for index,(X,y) in enumerate(train_loader):
        X,y = X.to(config.device),y.to(config.device)
        y_hat = net(X)
        l = loss(y_hat,y)
        opt.zero_grad()
        l.backward()
        opt.step()
        metric.add(l * y.numel(), accuracy(y,y_hat), y.numel())
    test_acc = evaluate(net,test_loader,config.device)
    print(f'[epoch:{epoch + 1} \t train_loss:{metric[0] / metric[2]:1.3f} \t train_acc:{metric[1] / metric[2]:1.3f} \t test_acc:{test_acc}')