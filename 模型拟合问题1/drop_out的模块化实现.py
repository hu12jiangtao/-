import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt

# nn.Dropout(k),当k=0的情况下相当于没有进行dropout,k的值越大，正则化的效果越强
# nn.Dropout的原理是相当于将多个模型进行了融合(共享一套参数)
# 例如第一次丢弃了n个神经元构成了网络1(梯度下降训练网络1中的参数)，当二次丢弃了不同的n个神经元构成了网络2(梯度下降训练网络2中的参数)


class DropOut(nn.Module):   # 自己构建的Dropout层
    def __init__(self,drop_out):
        super().__init__()
        self.drop_out=drop_out
    def forward(self,X):
        if self.training==True:   # 将Module设置为evaluation mode，相当于 self.training=False =》net.eval()就是将self.training设置为False
            cmp=torch.rand_like(X)
            cmp=(cmp>=self.drop_out)
            return cmp * X / (1 - self.drop_out)
        else:
            return X


#设置网络的超参数
num_input,num_output,num_linear1,num_linear2=784,10,256,256
drop_out1,drop_out2=0.5,0.5
lr,batch_size,epcho_times=0.5,256,5
#加载数据
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
#定义网络

net=nn.Sequential(nn.Flatten(),nn.Linear(num_input,num_linear1),nn.ReLU(),DropOut(drop_out1),
                  nn.Linear(num_linear1,num_linear2),nn.ReLU(),DropOut(drop_out2),nn.Linear(num_linear2,num_output))

#初始化参数
def init_param(m):
    if type(m)==nn.Linear:
        m.weight.data.normal_(0,0.1)
        m.bias.data.fill_(0)
net.apply(init_param)
#损失函数
loss=nn.CrossEntropyLoss()
#更新参数
trainer=torch.optim.SGD(net.parameters(),lr=lr)
#模型的评估
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
        net.eval()   # 此时代表进入评估的模式，评估模式下是不会进入drop_out的
    matric=add_machine(2)
    for X,y in deal_data:
        matric.add(accuracy(net(X),y),y.numel())
    return matric[0]/matric[1]
def epcho_train_ch3(net,loss,trainer,train_iters):
    if isinstance(net,nn.Module):
        net.train() #此时代表进入训练的模式，进入drop_out
    matric=add_machine(3)
    for X,y in train_iters:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*len(y),accuracy(y_hat,y),y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]

def epcho_train(net, train_iters, test_iters, loss, num_epochs, trainer):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics=epcho_train_ch3(net,loss,trainer,train_iters)
        print('*'*50)
        test_acc = evaluate_accuracy(net, test_iters)   # 此时可以证明eval()使得其进入的自己构建的else语句，就是drop_out并没有被使用
        #animator.add(epoch + 1, train_metrics + (test_acc,))
        print(train_metrics[1],test_acc)




epcho_train(net, train_iters, test_iters, loss, epcho_times, trainer)