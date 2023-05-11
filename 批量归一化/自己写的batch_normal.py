from torch import nn
from d2l import torch as d2l
import torch
from torch.nn import functional as F

d2l.use_svg_display()

def batch_norm(X,gamma,beta,moving_mean,moving_var,epsilon,momentum):
    if not torch.is_grad_enabled():  #此时在预测的模式
        X_hat=(X-moving_mean)/(moving_var+epsilon)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2:
            mean=torch.mean(X,dim=0,keepdim=True)
            var=torch.mean((X-mean)**2,dim=0,keepdim=True)
        else:
            mean=torch.mean(X,dim=(0,2,3),keepdim=True)
            var=torch.mean((X-mean)**2,dim=(0,2,3),keepdim=True)
        X_hat=(X-mean)/(var+epsilon)
        moving_mean=momentum*moving_mean+(1-momentum)*mean
        moving_var=momentum*moving_var+(1-momentum)*var
    Y=gamma*X_hat+beta
    return Y,moving_mean,moving_var


class BatchNormal(nn.Module):
    def __init__(self,in_channal,num_dims):
        super().__init__()
        assert num_dims in (2,4)
        if num_dims==2:
            shape=(1,in_channal)
        else:
            shape=(1,in_channal,1,1)
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))
        self.moving_mean=torch.zeros(shape,dtype=torch.float32)
        self.moving_var=torch.zeros(shape,dtype=torch.float32)
    def forward(self,X):
        if self.moving_mean.device != X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var=batch_norm(X,self.gamma,self.beta,self.moving_mean,self.moving_var,epsilon=1e-5,momentum=0.9)
        return Y

#此处利用的是LeNet
net=nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNormal(6, num_dims=4), nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(6, 16, kernel_size=5), BatchNormal(16, num_dims=4), nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                  nn.Linear(16*4*4, 120), BatchNormal(120, num_dims=2), nn.Sigmoid(),
                  nn.Linear(120, 84), BatchNormal(84, num_dims=2), nn.Sigmoid(),
                  nn.Linear(84, 10))

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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
    with torch.no_grad():
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
    print('is training:',device)
    matric=add_machine(3)
    for epoch in range(epoch_num):
        net.train()
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            trainer.zero_grad()
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            trainer.step()
            matric.add(l*y.shape[0],accuracy(y_hat,y),y.numel())
        test_acc=evaluate_accuracy(net,test_iters,device=try_gpu())
        print('当前迭代的次数:',epoch+1)
        print('train loss:',matric[0]/matric[2])
        print('train acc:', matric[1] / matric[2])
        print('test acc:', test_acc)
        print('*'*50)

batch_size=256
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
train(net,train_iters,test_iters,epoch_num=20,lr=1,device=try_gpu())