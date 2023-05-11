import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()

#生成SGD使用的小批量数据
batch_size=256
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
#设置整个网络
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
#给初始的参数进行赋值
def init_params(a):
    if type(a) ==nn.Linear:
        a.weight.data.normal_(0,0.01)
        a.bias.data.fill_(0)
net.apply(init_params)
#设计损失函数
loss=nn.CrossEntropyLoss()
#设计参数更新的方式
trainer=torch.optim.SGD([{'params':net[1].weight},{'params':net[1].bias},{'params':net[3].weight},{'params':net[3].bias}],lr=0.1)
#可以简SGD的第一个参数可以写成 net.paarameters() ,也可以每一层每一层分开的写，其为一个列表，里面的元素为字典
#建立一个评估机制
def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()
class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]
def evaluate_accuracy(net,deal_data):
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(2)
    for X,y in deal_data:
        matric.add(accuracy(net(X),y),y.numel())
    return matric[0]/matric[1]

def train_epcho_ch3(net,accuracy,loss,train_iters):
    if isinstance(net,nn.Module):
        net.train()
    matric=add_machine(3)
    for X,y in train_iters:
        print(X.shape)
        print(y.shape)
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*len(y),accuracy(y_hat,y),y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]

epcho_num=10
for i in range(epcho_num):
    train_loss,train_acc=train_epcho_ch3(net,accuracy,loss,train_iters)
    test_acc=evaluate_accuracy(net,test_iters)
    print(f'当前迭代次数:{i+1}')
    print('train_loss:',train_loss)
    print('train_acc:', train_acc)
    print('test_acc:', test_acc)
    print('*'*50)
