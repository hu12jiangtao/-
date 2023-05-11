import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()

#中间层的选择的输出通道应当和数据的模型复杂度相关

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

#导入数据
#输入的图片应当是num*1*28*28
batch_size=256
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size=batch_size)

def Lenet_model():
    net=nn.Sequential()
    net.add_module('conv1',nn.Conv2d(1,6,kernel_size=5,padding=2))
    net.add_module('activate1',nn.ReLU())
    net.add_module('ave_pool1',nn.AvgPool2d(stride=2,kernel_size=2))
    net.add_module('conv2',nn.Conv2d(6,16,kernel_size=5))
    net.add_module('activate2',nn.ReLU())
    net.add_module('ave_pool2',nn.AvgPool2d(stride=2,kernel_size=2))
    net.add_module('flatten',nn.Flatten())
    net.add_module('Linear1',nn.Linear(16*5*5,120))
    net.add_module('activate3',nn.ReLU())
    net.add_module('Linear2',nn.Linear(120,84))
    net.add_module('activate4',nn.ReLU())
    net.add_module('Linear3',nn.Linear(84,10))
    return net

net=Lenet_model()

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

def evaluate_accuracy(net,deal_data,device):
    if isinstance(net,nn.Module):
        net.eval()
    net.to(device)
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

def train(net,train_iters,test_iters,device,lr,epoch_num):
    def init_params(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_params)
    net.to(device)
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    print('now training:',device)
    for epcho in range(epoch_num):
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
        test_epoch_acc=evaluate_accuracy(net,test_iters, device)
        print(f'当前迭代次数为:{epcho+1}')
        print('train loss:',matric[0]/matric[2])
        print('train acc:', matric[1] / matric[2])
        print('test acc:',test_epoch_acc)
        print('*'*50)

train(net,train_iters,test_iters,device=try_gpu(),lr=0.1,epoch_num=20)


