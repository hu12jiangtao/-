# 此时对于训练的数据集会存在多种的数据增强操作
import torch
from torch import nn
import torchvision
from torch.utils import data
from d2l import torch as d2l
import argparse
d2l.use_svg_display()

parse = argparse.ArgumentParser()
parse.add_argument('--rule_mode',type=str,choices=['relu','P_relu','LeakyRelu'])
opt = parse.parse_args()


transf_train = torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                         torchvision.transforms.CenterCrop((28,28)),
                                         torchvision.transforms.RandomRotation(45), # 45角中的随即翻转
                                         torchvision.transforms.RandomVerticalFlip(p=0.5), # 水平翻转
                                         torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 水平的翻转
                                         torchvision.transforms.ToTensor()
                                         ])
transf_test = torchvision.transforms.Compose([torchvision.transforms.Resize((28,28)),
                                              torchvision.transforms.ToTensor()])
# 此时的卷积是固定的输入大小
torch.backends.cudnn.benchmark = True

# 数据集
path = '../data/MNIST'
train_dataset = torchvision.datasets.MNIST(path,train=True,transform=transf_train,download=True)
test_dataset = torchvision.datasets.MNIST(path,train=False,transform=transf_test,download=True)
train_iter = data.DataLoader(train_dataset,batch_size=16,shuffle=True)
test_iter = data.DataLoader(test_dataset,shuffle=False,batch_size=4)


'''
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)
'''

# 模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(7 * 7 * 32,10)
        self.flatten = nn.Flatten()
        if opt.rule_mode == 'relu':
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
        elif opt.rule_mode == 'P_relu':
            self.relu1 = nn.PReLU()
            self.relu2 = nn.PReLU()
        elif opt.rule_mode == 'LeakyRelu':
            self.relu1 = nn.LeakyReLU()
            self.relu2 = nn.LeakyReLU()

    def forward(self,x):
        out = self.maxPool1(self.relu1(self.conv1(x)))
        out = self.maxPool2(self.relu2(self.conv2(out)))
        out = self.linear(self.flatten(out))
        return out # [batch,10]

class add_machine(object):
    def __init__(self,n):
        self.data = [0.0] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    prob = torch.argmax(y_hat,dim=-1)
    cmp = (prob == y).type(y.dtype)
    return cmp.type(y.dtype).sum().item()

def evaluate_acc(net,test_loader,device):
    net.eval()
    net.to(device)
    metric = add_machine(2)
    for X,y in test_loader:
        X,y = X.to(device),y.to(device)
        y_hat = net(X)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0] / metric[1]

def train(train_iter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.CrossEntropyLoss()
    train_net = Model().to(device)
    opt = torch.optim.Adam(train_net.parameters())
    for epoch in range(10):
        train_net.train()
        metric = add_machine(3)
        for index,(X,y) in enumerate(train_iter):
            X,y = X.to(device),y.to(device)
            y_hat = train_net(X)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(accuracy(y_hat,y),l * y.numel(),y.numel())
        train_acc = metric[0] / metric[2]
        train_loss = metric[1] / metric[2]
        test_acc = evaluate_acc(train_net,test_iter,device)
        print(f'[{epoch + 1}/10] train_acc:{train_acc:1.3f} test_acc:{test_acc:1.3f},loss:{train_loss:1.3f}')

train(train_iter)

