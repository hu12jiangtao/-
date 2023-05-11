import torch
from torch import nn
import torchvision
from torch.utils import data


def load_image(batch_size):
    trans = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root="D:\\python\\gan简单版\\解决模式崩溃（模板）\\data",
                                               train=True, transform=trans, download=False)
    test_dataset = torchvision.datasets.MNIST(root="D:\\python\\gan简单版\\解决模式崩溃（模板）\\data",
                                               train=False, transform=trans, download=False)
    train_iter = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_iter = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_iter,test_iter # 训练集和测试集的数据并不相同


# 确定需要随机选择的超参数:dropout,lr,num_epochs
class Net(nn.Module):
    def __init__(self,hidden_num1, hidden_num2, dropout):  # hidden_num1=256, hidden_num2=100
        super(Net, self).__init__()
        self.net = nn.Sequential(self.blk(784,hidden_num1,dropout),self.blk(hidden_num1,hidden_num2,dropout))
        self.fc = nn.Linear(hidden_num2,10)

    def forward(self,x):
        return self.fc(self.net(x))

    def blk(self,in_channel,out_channel,dropout):
        return nn.Sequential(nn.Linear(in_channel,out_channel),nn.Dropout(dropout),nn.ReLU())

class add_machine(object):
    def __init__(self,n):
        self.data = [0.0] * n

    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]

    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    a = torch.argmax(y_hat,dim=1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(net,test_iter,device):
    metric = add_machine(2)
    for x,y in test_iter:
        x,y = x.to(device),y.to(device)
        x = x.reshape(x.shape[0],-1)
        y_hat = net(x)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0] / metric[1]

def train_model(each_give_param,train_dataset,test_dataset,device,is_print=True):
    num_epoch = each_give_param['num_epoch']
    lr = each_give_param['lr']
    dropout = each_give_param['dropout']  # dropout的值越大正则化效果越强
    net = Net(hidden_num1=256, hidden_num2=100, dropout=dropout).to(device)
    opt = torch.optim.SGD(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    batch_iter = 0
    metric = add_machine(3)
    for i in range(num_epoch):
        net.train()
        for x,y in train_dataset:
            x,y = x.to(device), y.to(device)
            x = x.reshape(x.shape[0],-1)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            batch_iter += 1
            metric.add(l * y.numel(), accuracy(y_hat,y), y.numel())
            if batch_iter % 200 == 0:
                net.eval()
                test_acc = evaluate_accuracy(net,test_dataset,device)
                if is_print:
                    print(f'epoch:{i + 1}  loss:{metric[0]/metric[1]:1.3f}  train_acc:{metric[1]/metric[2]:1.3f}'
                          f'  test_acc:{test_acc:1.3f}')
    net.eval()
    test_acc = evaluate_accuracy(net,test_dataset,device)
    return test_acc, net




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter,test_iter = load_image(batch_size=128)
    give_param = {'num_epoch':5, 'lr':0.1, 'dropout':0.9}
    train_model(each_give_param=give_param,
                train_dataset=train_iter,test_dataset=test_iter,device=device)

