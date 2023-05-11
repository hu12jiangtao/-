from d2l import torch as d2l
import numpy as np
import torch
import math
from torch.utils import data
from torch import nn

#L2正则化（用来解决过拟合的问题），作用与模型的容量（1.模型变得比较少，减少参数的数量，2.缩小模型参数的取值范围），利用方式二来解决模型的容量过大
#及对于损失函数loss加入了一个惩罚项，用来限制w的取值范围，此时变为 min(loss),s.t. ||w||^2<theta 可转变为 min(loss+0.5*gamma*||w||^2)
#此时当gamma=0时没有作用，当gamma->无穷时，w的取值只可能为0，模型容量为最小，因此gamma越大，作用最强


n_train,n_test,num_inputs,batch_size=20,100,200,5  #此时n_train仅仅选择20的原因是使得数据的复杂度较低
true_w,true_b=torch.ones([num_inputs,1])*0.01,0.05   #此时的权重形状为100*1 ,num_inputs代表特征数

def synthetic_data(w,b,sample_num):  #这一步是为生成随机的特征以及与之对应的标签
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,[sample_num,1])  # [sample_num,1],这一层只有一个神经元
    return X,Y
def load_array(train_data,batch_size,is_training=True):
    dataset=data.TensorDataset(*train_data)
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=is_training)

train_data=synthetic_data(true_w,true_b,n_train)  #n_train代表样本的个数
train_iter=load_array(train_data,batch_size,is_training=True)
test_data=synthetic_data(true_w,true_b,n_test)
test_iter=load_array(test_data,batch_size,is_training=False)

def init_params(m):
    if type(m)==nn.Linear:
        m.weight.data.normal_(0,1)
        m.bias.data.fill_(0)

def train_concise(wd):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    net.apply(init_params)
    loss=nn.MSELoss()
    epcho_time,lr=200,0.003
    #trainer=torch.optim.SGD([{'params':net[0].weight,'weight_decay':wd},{'params':net[0].bias}],lr=lr)
    trainer=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=wd)    #将每一层L2正则化的weight_decay都设置为了一个相同的wd
    animator=d2l.Animator(xlabel='epcho',ylabel='loss',yscale='log',xlim=[5,epcho_time],legend=['train','test'])
    for i in range(epcho_time):
        for X,y in train_iter:
            if isinstance(net,nn.Module):
                net.train()
            y_hat=net(X)
            l=loss(y_hat,y)
            if isinstance(trainer,torch.optim.Optimizer):
                trainer.zero_grad()
                l.backward()
                trainer.step()
        if (i+1)%5==0:
            animator.add(i+1,(d2l.evaluate_loss(net,train_iter,loss),d2l.evaluate_loss(net,test_iter,loss)))
    print('w的L2范数是:',torch.norm(net[0].weight.data).item())

train_concise(3)