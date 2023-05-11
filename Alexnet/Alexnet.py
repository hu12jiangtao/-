import torch
from d2l import torch as d2l
from torch import nn
d2l.use_svg_display()
import time
def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')
#得到训练数据和测试数据
batch_size=128
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size,resize=224)
#获得整个Alexnet
def Alex_net():
    net=nn.Sequential()
    net.add_module('conv1',nn.Conv2d(1,96,kernel_size=11,stride=4))
    net.add_module('activate1',nn.ReLU())
    net.add_module('max_pool1',nn.MaxPool2d(kernel_size=3,stride=2))
    net.add_module('conv2',nn.Conv2d(96,256,kernel_size=5,padding=2))
    net.add_module('activate2',nn.ReLU())
    net.add_module('max_pool2',nn.MaxPool2d(stride=2,kernel_size=3))
    net.add_module('conv3',nn.Conv2d(256,384,padding=1,kernel_size=3))
    net.add_module('activate3',nn.ReLU())
    net.add_module('conv4',nn.Conv2d(384,384,padding=1,kernel_size=3))
    net.add_module('activate4',nn.ReLU())
    net.add_module('conv5',nn.Conv2d(384,256,padding=1,kernel_size=3))
    net.add_module('activate5',nn.ReLU())
    net.add_module('max_pool3',nn.MaxPool2d(stride=2,kernel_size=3))
    net.add_module('flatten',nn.Flatten())
    net.add_module('Linear1',nn.Linear(5*5*256,4096))
    net.add_module('activate6',nn.ReLU())
    net.add_module('drop_out1',nn.Dropout(0.5))
    net.add_module('Linear2',nn.Linear(4096,4096))
    net.add_module('activate7', nn.ReLU())
    net.add_module('drop_out2', nn.Dropout(0.5))
    net.add_module('Linear3',nn.Linear(4096,10))
    return net

net=Alex_net()
'''
#用来查看各层网络的形状
X=torch.randn((1,1,224,224),dtype=torch.float32)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'\t output shape:',X.shape)
'''
class add_machine:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]
def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(net,deal_data,device):
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(2)
    for X,y in deal_data:
        if isinstance(X,list):
            X=[x.to(device) for x in X]
        else:
            X=X.to(device)
        y=y.to(device)
        y_hat=net(X)
        matric.add(accuracy(y_hat,y),y.numel())
    return matric[0]/matric[1]

def train(net,train_iters,test_iters,epoch_num,lr,device):
    def init_params(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_params)
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    net.to(device)
    print('training in :',device)
    for epoch in range(epoch_num):
        net.train()
        matric=add_machine(3)
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*y.shape[0],accuracy(y_hat,y),y.numel())
        test_acc=evaluate_accuracy(net,test_iters,device)
        train_loss=matric[0]/matric[2]
        train_acc=matric[1]/matric[2]
        print('当前迭代的次数:',epoch+1)
        print('train_loss:',train_loss)
        print('train_acc:', train_acc)
        print('test_acc', test_acc)
        print('*'*50)

t1=time.clock()
train(net,train_iters,test_iters,epoch_num=10,lr=0.015,device=try_gpu())
t2=time.clock()
print(t2-t1)